#!/usr/bin/env python
"""
CHE701P corrected Gaussian-path extraction and orientation QC (v3)
=============================================================

Purpose
-------
Create an authoritative, matched coordinate-energy-force dataset from the
supervisor-supplied Gaussian 16 checkpoint-continuation optimisation log.

Critical point
--------------
The log prints Standard-orientation coordinates but Gaussian force blocks in
the original Cartesian axes. In this particular log, cclib does not receive
Input/Z-matrix orientation tables, so its parsed ``grads`` cannot be assumed
to have been rotated into the Standard orientation.

This script therefore:
1. uses cclib for Standard-orientation coordinates, energies, convergence
   values and metadata;
2. independently parses the raw Gaussian force blocks;
3. reconstructs the original-axis coordinates used for every evaluated
   geometry from Gaussian's Cartesian Berny ``Old X``/``New X`` tables;
4. obtains the rigid rotation from reconstructed original coordinates to
   Standard orientation;
5. rotates every force vector into the same orientation as the coordinates;
6. performs independent count, unit, geometry, force and read-back checks;
7. writes the corrected dataset only if all mandatory QC tests pass.

The script performs no MLFF calculation and does not modify the raw files.

Usage
-----
    python 02_extract_gaussian_path_corrected_v3.py
or:
    python 02_extract_gaussian_path_corrected_v3.py "C:\\path\\to\\12_gaussian_path_benchmark"

Use --overwrite only after deliberately archiving an earlier derived output:
    python 02_extract_gaussian_path_corrected_v3.py "%STAGE%" --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ase
import cclib
from cclib.parser.utils import convertor as cclib_convertor
import numpy as np
import pandas as pd
import scipy
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols
from ase.io import read, write
from ase.units import Bohr, Hartree
from scipy.spatial.transform import Rotation


DEFAULT_STAGE = None

EXPECTED_SOURCE_SHA256 = (
    "b7d3187e26d8e5e3204054c99e1c962684dfb8886f4982fc130059151e49f6cb"
)
EXPECTED_ATOMS = 78
EXPECTED_STEPS = 244
EXPECTED_FORMULA_COUNTS = {"C": 13, "H": 47, "N": 2, "O": 16}
EXPECTED_TACRINE_COUNTS = {"C": 13, "H": 15, "N": 2}
EXPECTED_WATER_COUNTS = {"H": 32, "O": 16}

ALIGNMENT_RMSD_TOL_A = 1.0e-4
CCLIB_ENERGY_MATCH_TOL_EV = 1.0e-8
FORCE_TABLE_MATCH_TOL_AU = 1.0e-12
FORCE_CONVERGENCE_TOL_AU = 6.0e-7
COORDINATE_MATCH_TOL_A = 1.0e-12
ROTATION_METHOD_TOL = 1.0e-10
# ASE's extxyz writer serialises per-atom real-valued columns to eight
# decimal places. The maximum absolute rounding error from that text
# representation is therefore approximately 5e-9 in the stored unit.
# These tolerances apply only to write/read serialisation checks; all
# upstream coordinate, energy, force, rotation and convergence QC
# retains its original stricter tolerances.
EXTXYZ_POSITION_READBACK_TOL_A = 5.1e-9
EXTXYZ_ENERGY_READBACK_TOL_EV = 1.0e-10
EXTXYZ_FORCE_READBACK_TOL_EV_PER_A = 5.1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and validate the corrected Gaussian path."
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default=DEFAULT_STAGE,
        help="Path to 12_gaussian_path_benchmark.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing derived output files.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_structure(atomnos: np.ndarray, coordinates_a: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(atomnos, dtype="<i4").tobytes(order="C"))
    digest.update(np.asarray(coordinates_a, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def float_gaussian(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_symbols(atomnos: np.ndarray) -> list[str]:
    return [chemical_symbols[int(number)] for number in atomnos]


def count_symbols(symbols: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(symbols).items()))


def ensure_expected_outputs_absent(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Derived outputs already exist. Archive them or rerun with --overwrite:\n"
            f"{names}"
        )


def parse_standard_orientations(
    lines: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    coordinates: list[np.ndarray] = []
    atomnos_reference: np.ndarray | None = None

    index = 0
    while index < len(lines):
        if "Standard orientation:" not in lines[index]:
            index += 1
            continue

        cursor = index + 1
        separators = 0
        while cursor < len(lines):
            if lines[cursor].strip().startswith("-----"):
                separators += 1
                if separators == 2:
                    cursor += 1
                    break
            cursor += 1

        frame: list[list[float]] = []
        atomnos: list[int] = []
        while cursor < len(lines) and not lines[cursor].strip().startswith("-----"):
            parts = lines[cursor].split()
            if len(parts) >= 6 and parts[0].isdigit():
                atomnos.append(int(parts[1]))
                frame.append(
                    [
                        float_gaussian(parts[3]),
                        float_gaussian(parts[4]),
                        float_gaussian(parts[5]),
                    ]
                )
            cursor += 1

        if len(frame) == EXPECTED_ATOMS:
            current_atomnos = np.asarray(atomnos, dtype=int)
            if atomnos_reference is None:
                atomnos_reference = current_atomnos
            elif not np.array_equal(atomnos_reference, current_atomnos):
                raise ValueError("Atomic-number order changed between Standard orientations.")
            coordinates.append(np.asarray(frame, dtype=float))

        index = cursor + 1

    if atomnos_reference is None:
        raise ValueError("No complete Standard-orientation geometry was parsed.")

    return np.asarray(coordinates, dtype=float), atomnos_reference


def parse_scf_energies_hartree(lines: list[str]) -> np.ndarray:
    pattern = re.compile(
        r"SCF Done:\s+.*?=\s*([-+]?\d+\.\d+(?:[DEde][-+]?\d+)?)"
    )
    values = []
    for line in lines:
        match = pattern.search(line)
        if match:
            values.append(float_gaussian(match.group(1)))
    return np.asarray(values, dtype=float)


def parse_raw_force_blocks(
    lines: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    force_frames: list[np.ndarray] = []
    atomnos_reference: np.ndarray | None = None

    index = 0
    while index < len(lines):
        if "Forces (Hartrees/Bohr)" not in lines[index]:
            index += 1
            continue

        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].strip().startswith("-----"):
            cursor += 1
        cursor += 1

        frame: list[list[float]] = []
        atomnos: list[int] = []
        while cursor < len(lines) and not lines[cursor].strip().startswith("-----"):
            parts = lines[cursor].split()
            if len(parts) >= 5 and parts[0].isdigit():
                atomnos.append(int(parts[1]))
                frame.append(
                    [
                        float_gaussian(parts[2]),
                        float_gaussian(parts[3]),
                        float_gaussian(parts[4]),
                    ]
                )
            cursor += 1

        if len(frame) == EXPECTED_ATOMS:
            current_atomnos = np.asarray(atomnos, dtype=int)
            if atomnos_reference is None:
                atomnos_reference = current_atomnos
            elif not np.array_equal(atomnos_reference, current_atomnos):
                raise ValueError("Atomic-number order changed between force blocks.")
            force_frames.append(np.asarray(frame, dtype=float))

        index = cursor + 1

    if atomnos_reference is None:
        raise ValueError("No complete Gaussian force block was parsed.")

    return np.asarray(force_frames, dtype=float), atomnos_reference


def parse_cartesian_berny_tables(
    lines: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Old X, New X and -DE/DX tables in Gaussian original Cartesian axes.

    Coordinates are returned in Bohr, and -DE/DX in Hartree/Bohr.
    A few printed -DE/DX entries may be NaN; the authoritative raw force blocks
    are used for the force dataset, while the Berny force column is retained
    only as an auxiliary diagnostic.
    """

    row_pattern = re.compile(
        r"^\s*([XYZ])(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"
    )
    axis_to_index = {"X": 0, "Y": 1, "Z": 2}

    old_frames: list[np.ndarray] = []
    new_frames: list[np.ndarray] = []
    derivative_frames: list[np.ndarray] = []

    index = 0
    while index < len(lines):
        line = lines[index]
        if not (line.strip().startswith("Variable") and "Old X" in line):
            index += 1
            continue

        old_values: dict[tuple[int, str], float] = {}
        new_values: dict[tuple[int, str], float] = {}
        derivative_values: dict[tuple[int, str], float] = {}

        cursor = index + 1
        while cursor < len(lines):
            current = lines[cursor]
            if "Item" in current and "Threshold" in current and "Converged?" in current:
                break

            match = row_pattern.match(current)
            if match:
                axis, atom_index, old_x, derivative, _, _, _, new_x = match.groups()
                key = (int(atom_index), axis)
                old_values[key] = float_gaussian(old_x)
                new_values[key] = float_gaussian(new_x)
                derivative_values[key] = float_gaussian(derivative)

            cursor += 1

        if len(old_values) != EXPECTED_ATOMS * 3:
            raise ValueError(
                f"Incomplete Cartesian Berny table near line {index + 1}: "
                f"{len(old_values)} of {EXPECTED_ATOMS * 3} coordinate rows."
            )

        old_array = np.zeros((EXPECTED_ATOMS, 3), dtype=float)
        new_array = np.zeros((EXPECTED_ATOMS, 3), dtype=float)
        derivative_array = np.zeros((EXPECTED_ATOMS, 3), dtype=float)

        for (atom_index, axis), old_value in old_values.items():
            component = axis_to_index[axis]
            old_array[atom_index - 1, component] = old_value
            new_array[atom_index - 1, component] = new_values[(atom_index, axis)]
            derivative_array[atom_index - 1, component] = derivative_values[
                (atom_index, axis)
            ]

        old_frames.append(old_array)
        new_frames.append(new_array)
        derivative_frames.append(derivative_array)

        index = cursor + 1

    return (
        np.asarray(old_frames, dtype=float),
        np.asarray(new_frames, dtype=float),
        np.asarray(derivative_frames, dtype=float),
    )


def parse_convergence_tables(
    lines: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = (
        "Maximum Force",
        "RMS     Force",
        "Maximum Displacement",
        "RMS     Displacement",
    )

    values: list[list[float]] = []
    targets: list[list[float]] = []
    flags: list[list[bool]] = []

    for index, line in enumerate(lines):
        if not (
            "Item" in line and "Threshold" in line and "Converged?" in line
        ):
            continue

        current_values: list[float] = []
        current_targets: list[float] = []
        current_flags: list[bool] = []

        for cursor in range(index + 1, min(index + 9, len(lines))):
            current = lines[cursor]
            if any(label in current for label in labels):
                parts = current.split()
                current_values.append(float_gaussian(parts[-3]))
                current_targets.append(float_gaussian(parts[-2]))
                current_flags.append(parts[-1].upper() == "YES")

        if len(current_values) == 4:
            values.append(current_values)
            targets.append(current_targets)
            flags.append(current_flags)

    return (
        np.asarray(values, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(flags, dtype=bool),
    )


def parse_step_numbers(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pattern = re.compile(
        r"Step number\s+(\d+)\s+out of a maximum of\s+(\d+)"
    )
    steps: list[int] = []
    maxima: list[int] = []
    for line in lines:
        match = pattern.search(line)
        if match:
            steps.append(int(match.group(1)))
            maxima.append(int(match.group(2)))
    return np.asarray(steps, dtype=int), np.asarray(maxima, dtype=int)


def kabsch_row_rotation(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Find row-vector rotation R such that centred_source @ R ~= centred_target."""
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)

    u_matrix, _, vt_matrix = np.linalg.svd(source_centered.T @ target_centered)
    rotation_matrix = u_matrix @ vt_matrix

    if np.linalg.det(rotation_matrix) < 0:
        u_matrix[:, -1] *= -1
        rotation_matrix = u_matrix @ vt_matrix

    aligned = source_centered @ rotation_matrix
    rmsd = float(
        np.sqrt(np.mean(np.sum((aligned - target_centered) ** 2, axis=1)))
    )
    return rotation_matrix, rmsd


def scipy_rotation_check(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[Rotation, float]:
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    rotation, _ = Rotation.align_vectors(target_centered, source_centered)
    aligned = rotation.apply(source_centered)
    rmsd = float(
        np.sqrt(np.mean(np.sum((aligned - target_centered) ** 2, axis=1)))
    )
    return rotation, rmsd


def closest_frame_rmsd(
    query: np.ndarray,
    frames: np.ndarray,
) -> tuple[int, float]:
    best_index = -1
    best_rmsd = math.inf
    for frame_index, frame in enumerate(frames):
        _, rmsd = kabsch_row_rotation(query, frame)
        if rmsd < best_rmsd:
            best_index = frame_index
            best_rmsd = rmsd
    return best_index, best_rmsd


def write_text_report(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("CHE701P corrected Gaussian-path extraction and QC")
    lines.append("=" * 52)
    lines.append(f"Status: {report['status']}")
    lines.append(f"Timestamp UTC: {report['timestamp_utc']}")
    lines.append(f"Source log: {report['source_log']['filename']}")
    lines.append(f"Source SHA256: {report['source_log']['sha256']}")
    lines.append("")
    lines.append("Core counts")
    lines.append("-" * 11)
    for key, value in report.get("counts", {}).items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("Orientation findings")
    lines.append("-" * 20)
    for key, value in report.get("orientation", {}).items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("Energy and convergence findings")
    lines.append("-" * 31)
    for key, value in report.get("energy_convergence", {}).items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("Supplied XYZ comparison")
    lines.append("-" * 23)
    for key, value in report.get("supplied_xyz", {}).items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("Mandatory QC checks")
    lines.append("-" * 19)
    for key, value in report.get("checks", {}).items():
        lines.append(f"{key}: {value}")

    if report.get("warnings"):
        lines.append("")
        lines.append("Warnings")
        lines.append("-" * 8)
        lines.extend(f"- {warning}" for warning in report["warnings"])

    if report.get("error"):
        lines.append("")
        lines.append("Error")
        lines.append("-" * 5)
        lines.append(report["error"])
        lines.append(report.get("traceback", ""))

    lines.append("")
    lines.append("Scientific interpretation")
    lines.append("-" * 25)
    lines.append(
        "The extracted 244 structures are sequential geometries along an "
        "unconverged B3LYP/6-31G(d) checkpoint-continuation optimisation trace."
    )
    lines.append(
        "They are not independent samples and do not constitute validation of "
        "the full 100-configuration MD-derived dataset."
    )
    lines.append(
        "The force vectors in the corrected output have been rotated into the "
        "same Standard orientation as the stored coordinates."
    )
    lines.append(
        "The lowest-energy sampled geometry is not a converged or "
        "frequency-confirmed minimum."
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.stage is None:
        raise SystemExit(
            "A private 12_gaussian_path_benchmark stage path is required. "
            "Pass it explicitly as the first argument; raw Gaussian/project files are not redistributed."
        )
    stage = Path(args.stage).expanduser().resolve()
    raw_dir = stage / "00_received_raw"
    script_dir = stage / "02_reference_extraction" / "01_scripts"
    output_dir = stage / "02_reference_extraction" / "02_outputs"
    log_dir = stage / "02_reference_extraction" / "03_logs"
    qc_orientation_dir = stage / "03_reference_qc" / "02_orientation"
    manifest_dir = stage / "10_reproducibility" / "02_manifests"

    for directory in (
        script_dir,
        output_dir,
        log_dir,
        qc_orientation_dir,
        manifest_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "extxyz": output_dir / "gaussian_path_corrected.extxyz",
        "structures": output_dir / "gaussian_path_structures.csv",
        "atomic_forces": output_dir / "gaussian_path_atomic_forces.csv",
        "convergence": output_dir / "gaussian_path_convergence.csv",
        "orientation_qc": qc_orientation_dir / "gaussian_path_orientation_qc.csv",
        "metadata": output_dir / "gaussian_path_metadata.json",
        "xyz_comparison": output_dir / "supplied_xyz_comparison.json",
        "qc_report": output_dir / "gaussian_path_extraction_qc.txt",
        "run_log": log_dir / "02_extract_gaussian_path_corrected.log",
        "manifest": manifest_dir / "gaussian_path_outputs_SHA256.csv",
    }

    report: dict[str, Any] = {
        "status": "STARTED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": str(stage),
        "software": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "ase": ase.__version__,
            "cclib": cclib.__version__,
        },
        "checks": {},
        "warnings": [],
    }

    try:
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"Raw input directory not found: {raw_dir}")

        log_files = sorted(raw_dir.glob("*.log"))
        xyz_files = sorted(raw_dir.glob("*.xyz"))
        if len(log_files) != 1:
            raise RuntimeError(
                f"Expected exactly one .log file; found {len(log_files)}."
            )
        if len(xyz_files) != 1:
            raise RuntimeError(
                f"Expected exactly one supplied .xyz file; found {len(xyz_files)}."
            )

        log_path = log_files[0]
        supplied_xyz_path = xyz_files[0]
        source_hash = sha256_file(log_path)
        report["source_log"] = {
            "filename": log_path.name,
            "path": str(log_path),
            "size_bytes": log_path.stat().st_size,
            "sha256": source_hash,
        }

        report["checks"]["source_hash_matches"] = (
            source_hash.lower() == EXPECTED_SOURCE_SHA256
        )
        if not report["checks"]["source_hash_matches"]:
            raise ValueError(
                "Source-log SHA256 does not match the previously verified file."
            )

        ensure_expected_outputs_absent(
            list(output_paths.values()),
            overwrite=args.overwrite,
        )

        lines = log_path.read_text(errors="replace").splitlines()

        # cclib primary parse.
        parser = cclib.io.ccopen(str(log_path))
        if parser is None:
            raise RuntimeError("cclib could not identify the Gaussian parser.")
        data = parser.parse()

        cclib_coords = np.asarray(data.atomcoords, dtype=float)
        cclib_atomnos = np.asarray(data.atomnos, dtype=int)
        cclib_energies_ev = np.asarray(data.scfenergies, dtype=float)
        cclib_grads = np.asarray(data.grads, dtype=float)
        cclib_geovalues = np.asarray(data.geovalues, dtype=float)
        cclib_geotargets = np.asarray(data.geotargets, dtype=float)
        cclib_optstatus = np.asarray(data.optstatus, dtype=int)
        cclib_inputcoords_present = hasattr(data, "inputcoords")

        # Independent raw-text parse.
        standard_all, standard_atomnos = parse_standard_orientations(lines)
        raw_energies_hartree = parse_scf_energies_hartree(lines)
        raw_forces_input_au, force_atomnos = parse_raw_force_blocks(lines)
        old_x_bohr, new_x_bohr, berny_derivatives = parse_cartesian_berny_tables(
            lines
        )
        conv_values, conv_targets, conv_flags = parse_convergence_tables(lines)
        step_numbers, maximum_steps = parse_step_numbers(lines)

        counts = {
            "cclib_coordinate_frames": int(cclib_coords.shape[0]),
            "independent_standard_orientations": int(standard_all.shape[0]),
            "raw_scf_energies": int(raw_energies_hartree.shape[0]),
            "raw_force_blocks": int(raw_forces_input_au.shape[0]),
            "cartesian_berny_tables": int(old_x_bohr.shape[0]),
            "convergence_tables": int(conv_values.shape[0]),
            "step_number_records": int(step_numbers.shape[0]),
            "atoms_per_frame": int(cclib_atomnos.shape[0]),
        }
        report["counts"] = counts

        mandatory_count_arrays = {
            "cclib coordinates": cclib_coords.shape[0],
            "cclib energies": cclib_energies_ev.shape[0],
            "cclib gradients": cclib_grads.shape[0],
            "cclib geovalues": cclib_geovalues.shape[0],
            "cclib optstatus": cclib_optstatus.shape[0],
            "raw energies": raw_energies_hartree.shape[0],
            "raw force blocks": raw_forces_input_au.shape[0],
            "Berny Old X": old_x_bohr.shape[0],
            "Berny New X": new_x_bohr.shape[0],
            "convergence tables": conv_values.shape[0],
            "step numbers": step_numbers.shape[0],
        }
        count_failures = {
            name: int(count)
            for name, count in mandatory_count_arrays.items()
            if int(count) != EXPECTED_STEPS
        }
        report["checks"]["all_core_counts_equal_244"] = not count_failures
        if count_failures:
            raise ValueError(f"Unexpected core array counts: {count_failures}")

        report["checks"]["standard_orientation_count_is_245"] = (
            standard_all.shape[0] == EXPECTED_STEPS + 1
        )
        if not report["checks"]["standard_orientation_count_is_245"]:
            raise ValueError(
                "Expected 245 Standard orientations: 244 evaluated geometries "
                "plus one unmatched trailing orientation."
            )

        standard_evaluated = standard_all[:EXPECTED_STEPS]
        standard_extra = standard_all[EXPECTED_STEPS:]

        report["checks"]["atom_number_sources_match"] = bool(
            np.array_equal(cclib_atomnos, standard_atomnos)
            and np.array_equal(cclib_atomnos, force_atomnos)
        )
        if not report["checks"]["atom_number_sources_match"]:
            raise ValueError("Atomic-number order differs between parsed sources.")

        symbols = atomic_symbols(cclib_atomnos)
        full_counts = count_symbols(symbols)
        tacrine_counts = count_symbols(symbols[:30])
        water_counts = count_symbols(symbols[30:])
        water_triplets_valid = all(
            symbols[30 + 3 * water_index : 33 + 3 * water_index] == ["O", "H", "H"]
            for water_index in range(16)
        )

        report["checks"]["atom_count_is_78"] = len(symbols) == EXPECTED_ATOMS
        report["checks"]["full_formula_matches"] = full_counts == EXPECTED_FORMULA_COUNTS
        report["checks"]["tacrine_block_matches"] = (
            tacrine_counts == EXPECTED_TACRINE_COUNTS
        )
        report["checks"]["water_block_matches"] = water_counts == EXPECTED_WATER_COUNTS
        report["checks"]["water_triplet_order_is_OHH"] = water_triplets_valid

        if not all(
            report["checks"][key]
            for key in (
                "atom_count_is_78",
                "full_formula_matches",
                "tacrine_block_matches",
                "water_block_matches",
                "water_triplet_order_is_OHH",
            )
        ):
            raise ValueError(
                "Composition or atom-order validation failed: "
                f"full={full_counts}, tacrine={tacrine_counts}, water={water_counts}."
            )

        coordinate_max_difference = float(
            np.max(np.abs(cclib_coords - standard_evaluated))
        )
        report["checks"]["cclib_coordinates_match_raw_standard"] = (
            coordinate_max_difference <= COORDINATE_MATCH_TOL_A
        )
        if not report["checks"]["cclib_coordinates_match_raw_standard"]:
            raise ValueError(
                f"cclib/raw Standard coordinates differ by {coordinate_max_difference:.3e} Å."
            )

        # Validate the parsed SCF values using cclib's own documented unit
        # conversion.  Do not compare enormous absolute electronic energies
        # after converting them with two different physical-constant tables:
        # cclib and ASE use slightly different Hartree-to-eV constants, and the
        # resulting harmless offset is amplified by an approximately -1836 Eh
        # total energy.  The raw Gaussian Hartree values remain authoritative;
        # ASE's Hartree constant is used for all exported eV values so that the
        # extxyz is internally consistent with ASE calculators.
        raw_energies_cclib_ev = np.asarray(
            cclib_convertor(raw_energies_hartree, "hartree", "eV"),
            dtype=float,
        )
        authoritative_energies_ev = raw_energies_hartree * Hartree

        energy_cclib_conversion_max_difference_ev = float(
            np.max(np.abs(cclib_energies_ev - raw_energies_cclib_ev))
        )
        energy_ase_vs_cclib_absolute_offset_max_ev = float(
            np.max(np.abs(authoritative_energies_ev - cclib_energies_ev))
        )
        cclib_relative_ev = cclib_energies_ev - cclib_energies_ev[0]
        ase_relative_ev = authoritative_energies_ev - authoritative_energies_ev[0]
        energy_ase_vs_cclib_relative_max_difference_ev = float(
            np.max(np.abs(ase_relative_ev - cclib_relative_ev))
        )

        report["checks"]["cclib_energies_match_raw_scf"] = (
            energy_cclib_conversion_max_difference_ev
            <= CCLIB_ENERGY_MATCH_TOL_EV
        )
        if not report["checks"]["cclib_energies_match_raw_scf"]:
            raise ValueError(
                "cclib SCF energies do not match independently parsed Hartree "
                "energies when the same cclib conversion is used: "
                f"max difference={energy_cclib_conversion_max_difference_ev:.3e} eV."
            )

        geovalue_max_difference = float(
            np.max(np.abs(cclib_geovalues - conv_values))
        )
        geotarget_max_difference = float(
            np.max(np.abs(cclib_geotargets - conv_targets[0]))
        )
        report["checks"]["cclib_convergence_values_match_raw"] = (
            geovalue_max_difference <= 1.0e-12
            and geotarget_max_difference <= 1.0e-12
        )
        if not report["checks"]["cclib_convergence_values_match_raw"]:
            raise ValueError("cclib convergence arrays do not match the raw tables.")

        report["checks"]["gaussian_steps_are_sequential_1_to_244"] = bool(
            np.array_equal(step_numbers, np.arange(1, EXPECTED_STEPS + 1))
        )
        report["checks"]["reported_maximum_steps_are_244"] = bool(
            np.all(maximum_steps == EXPECTED_STEPS)
        )
        if not (
            report["checks"]["gaussian_steps_are_sequential_1_to_244"]
            and report["checks"]["reported_maximum_steps_are_244"]
        ):
            raise ValueError("Gaussian step numbering is not the expected 1–244 sequence.")

        # Reconstruct the original-axis coordinates at every evaluated SCF point.
        input_coords_bohr = np.empty_like(old_x_bohr)
        input_coords_bohr[0] = old_x_bohr[0]
        input_coords_bohr[1:] = new_x_bohr[:-1]
        input_coords_a = input_coords_bohr * Bohr

        corrected_forces_au = np.empty_like(raw_forces_input_au)
        orientation_rows: list[dict[str, Any]] = []

        alignment_rmsds: list[float] = []
        rotation_matrix_differences: list[float] = []
        force_rotation_differences: list[float] = []
        rotation_determinants: list[float] = []
        current_old_rmsds: list[float] = []

        for frame_index in range(EXPECTED_STEPS):
            source_coords = input_coords_a[frame_index]
            target_coords = standard_evaluated[frame_index]

            rotation_matrix, kabsch_rmsd = kabsch_row_rotation(
                source_coords,
                target_coords,
            )
            scipy_rotation, scipy_rmsd = scipy_rotation_check(
                source_coords,
                target_coords,
            )

            scipy_row_matrix = scipy_rotation.as_matrix().T
            matrix_difference = float(
                np.max(np.abs(rotation_matrix - scipy_row_matrix))
            )

            force_kabsch = raw_forces_input_au[frame_index] @ rotation_matrix
            force_scipy = scipy_rotation.apply(raw_forces_input_au[frame_index])
            force_method_difference = float(
                np.max(np.abs(force_kabsch - force_scipy))
            )
            corrected_forces_au[frame_index] = force_kabsch

            _, old_rmsd = kabsch_row_rotation(
                old_x_bohr[frame_index] * Bohr,
                target_coords,
            )

            alignment_rmsds.append(kabsch_rmsd)
            rotation_matrix_differences.append(matrix_difference)
            force_rotation_differences.append(force_method_difference)
            rotation_determinants.append(float(np.linalg.det(rotation_matrix)))
            current_old_rmsds.append(old_rmsd)

            orientation_rows.append(
                {
                    "frame_index_0based": frame_index,
                    "gaussian_step_1based": frame_index + 1,
                    "input_coordinate_source": (
                        "current_Old_X"
                        if frame_index == 0
                        else "previous_step_New_X"
                    ),
                    "input_to_standard_alignment_rmsd_A": kabsch_rmsd,
                    "scipy_alignment_rmsd_A": scipy_rmsd,
                    "kabsch_vs_scipy_matrix_max_abs_difference": matrix_difference,
                    "kabsch_vs_scipy_rotated_force_max_abs_difference_au": (
                        force_method_difference
                    ),
                    "rotation_determinant": float(np.linalg.det(rotation_matrix)),
                    "current_Old_X_to_standard_rmsd_A": old_rmsd,
                    "current_Old_X_is_same_evaluated_geometry": old_rmsd
                    <= ALIGNMENT_RMSD_TOL_A,
                }
            )

        max_alignment_rmsd = float(max(alignment_rmsds))
        max_rotation_method_difference = float(max(rotation_matrix_differences))
        max_force_rotation_method_difference = float(max(force_rotation_differences))
        determinant_deviation = float(
            np.max(np.abs(np.asarray(rotation_determinants) - 1.0))
        )

        report["checks"]["all_input_to_standard_alignments_pass"] = (
            max_alignment_rmsd <= ALIGNMENT_RMSD_TOL_A
        )
        report["checks"]["kabsch_and_scipy_rotations_agree"] = (
            max_rotation_method_difference <= ROTATION_METHOD_TOL
            and max_force_rotation_method_difference <= ROTATION_METHOD_TOL
        )
        report["checks"]["all_rotations_are_proper"] = determinant_deviation <= 1.0e-10

        if not all(
            report["checks"][key]
            for key in (
                "all_input_to_standard_alignments_pass",
                "kabsch_and_scipy_rotations_agree",
                "all_rotations_are_proper",
            )
        ):
            raise ValueError(
                "Coordinate-to-force orientation reconstruction failed QC."
            )

        # Raw Gaussian forces must reproduce its convergence force criteria.
        raw_max_force = np.max(np.abs(raw_forces_input_au), axis=(1, 2))
        raw_rms_force = np.sqrt(np.mean(raw_forces_input_au**2, axis=(1, 2)))
        max_force_convergence_difference = float(
            np.max(np.abs(raw_max_force - conv_values[:, 0]))
        )
        rms_force_convergence_difference = float(
            np.max(np.abs(raw_rms_force - conv_values[:, 1]))
        )
        report["checks"]["raw_forces_reproduce_gaussian_convergence_values"] = (
            max_force_convergence_difference <= FORCE_CONVERGENCE_TOL_AU
            and rms_force_convergence_difference <= FORCE_CONVERGENCE_TOL_AU
        )
        if not report["checks"]["raw_forces_reproduce_gaussian_convergence_values"]:
            raise ValueError(
                "Parsed raw forces do not reproduce Gaussian's maximum/RMS "
                "force convergence values."
            )

        # Diagnose the cclib gradient orientation for this exact file.
        cclib_vs_raw_force_max = float(
            np.nanmax(np.abs(cclib_grads - raw_forces_input_au))
        )
        cclib_vs_corrected_force_max = float(
            np.nanmax(np.abs(cclib_grads - corrected_forces_au))
        )
        report["checks"]["cclib_grads_are_finite"] = bool(
            np.isfinite(cclib_grads).all()
        )
        if not report["checks"]["cclib_grads_are_finite"]:
            raise ValueError("cclib gradients contain non-finite values.")

        if cclib_inputcoords_present:
            cclib_orientation_status = (
                "cclib_inputcoords_present_and_grads_compared_to_corrected"
            )
            report["checks"]["cclib_grads_match_corrected_forces"] = (
                cclib_vs_corrected_force_max <= 1.0e-10
            )
            if not report["checks"]["cclib_grads_match_corrected_forces"]:
                raise ValueError(
                    "cclib inputcoords are present, but cclib gradients do not "
                    "match the independently corrected forces."
                )
        else:
            cclib_orientation_status = (
                "cclib_inputcoords_absent; parsed_grads_remain_in_raw_input_axes"
            )
            report["checks"]["cclib_grads_match_raw_force_blocks"] = (
                cclib_vs_raw_force_max <= FORCE_TABLE_MATCH_TOL_AU
            )
            if not report["checks"]["cclib_grads_match_raw_force_blocks"]:
                raise ValueError(
                    "cclib inputcoords are absent, but cclib gradients do not "
                    "match the independently parsed raw force blocks."
                )

        report["orientation"] = {
            "cclib_inputcoords_present": cclib_inputcoords_present,
            "cclib_gradient_orientation_status": cclib_orientation_status,
            "maximum_input_to_standard_alignment_rmsd_A": max_alignment_rmsd,
            "maximum_kabsch_vs_scipy_matrix_difference": (
                max_rotation_method_difference
            ),
            "maximum_kabsch_vs_scipy_force_difference_au": (
                max_force_rotation_method_difference
            ),
            "maximum_rotation_determinant_deviation_from_1": determinant_deviation,
            "cclib_vs_raw_force_max_abs_difference_au": cclib_vs_raw_force_max,
            "cclib_vs_corrected_force_max_abs_difference_au": (
                cclib_vs_corrected_force_max
            ),
            "steps_where_current_Old_X_was_not_the_evaluated_geometry": int(
                np.sum(np.asarray(current_old_rmsds) > ALIGNMENT_RMSD_TOL_A)
            ),
            "force_conversion_factor_eV_per_A_per_Hartree_per_Bohr": (
                Hartree / Bohr
            ),
        }

        force_factor = Hartree / Bohr
        corrected_forces_ev_a = corrected_forces_au * force_factor

        all_finite = bool(
            np.isfinite(cclib_coords).all()
            and np.isfinite(authoritative_energies_ev).all()
            and np.isfinite(corrected_forces_au).all()
            and np.isfinite(conv_values).all()
        )
        report["checks"]["all_authoritative_arrays_are_finite"] = all_finite
        if not all_finite:
            raise ValueError("One or more authoritative arrays contain non-finite values.")

        lowest_index = int(np.argmin(authoritative_energies_ev))
        relative_energy_ev = (
            authoritative_energies_ev - authoritative_energies_ev[lowest_index]
        )

        report["energy_convergence"] = {
            "first_energy_hartree": float(raw_energies_hartree[0]),
            "last_energy_hartree": float(raw_energies_hartree[-1]),
            "lowest_sampled_energy_hartree": float(
                raw_energies_hartree[lowest_index]
            ),
            "lowest_sampled_energy_gaussian_step": lowest_index + 1,
            "optimization_converged_cclib_optdone": bool(data.optdone),
            "cclib_metadata_success": bool(data.metadata.get("success", False)),
            "all_four_criteria_met_at_any_step": bool(np.any(np.all(conv_flags, axis=1))),
            "all_four_criteria_met_at_final_step": bool(np.all(conv_flags[-1])),
            "maximum_raw_force_convergence_difference_au": (
                max_force_convergence_difference
            ),
            "rms_raw_force_convergence_difference_au": (
                rms_force_convergence_difference
            ),
            "energy_cclib_conversion_max_difference_eV": (
                energy_cclib_conversion_max_difference_ev
            ),
            "energy_ASE_vs_cclib_absolute_offset_max_eV": (
                energy_ase_vs_cclib_absolute_offset_max_ev
            ),
            "energy_ASE_vs_cclib_relative_max_difference_eV": (
                energy_ase_vs_cclib_relative_max_difference_ev
            ),
            "geovalue_cclib_vs_raw_max_difference": geovalue_max_difference,
            "geotarget_cclib_vs_raw_max_difference": geotarget_max_difference,
        }

        # Compare the supplied XYZ against all evaluated and the unmatched extra geometry.
        supplied_atoms = read(str(supplied_xyz_path))
        supplied_symbols = supplied_atoms.get_chemical_symbols()
        supplied_coords = np.asarray(supplied_atoms.positions, dtype=float)

        report["checks"]["supplied_xyz_has_78_atoms"] = len(supplied_atoms) == EXPECTED_ATOMS
        report["checks"]["supplied_xyz_atom_order_matches"] = supplied_symbols == symbols
        if not (
            report["checks"]["supplied_xyz_has_78_atoms"]
            and report["checks"]["supplied_xyz_atom_order_matches"]
        ):
            raise ValueError("The supplied XYZ atom count/order does not match the log.")

        closest_index, closest_rmsd = closest_frame_rmsd(
            supplied_coords,
            standard_evaluated,
        )
        _, extra_rmsd = kabsch_row_rotation(
            supplied_coords,
            standard_extra[0],
        )
        supplied_xyz_result = {
            "filename": supplied_xyz_path.name,
            "sha256": sha256_file(supplied_xyz_path),
            "closest_evaluated_frame_index_0based": closest_index,
            "closest_evaluated_gaussian_step_1based": closest_index + 1,
            "closest_evaluated_geometry_rmsd_A": closest_rmsd,
            "unmatched_trailing_standard_orientation_rmsd_A": extra_rmsd,
            "interpretation": (
                "The supplied XYZ most closely corresponds to the stated "
                "evaluated Gaussian step after rigid alignment. It must not "
                "be labelled as a converged optimised geometry."
            ),
        }
        report["supplied_xyz"] = supplied_xyz_result

        # Build structure and atomic tables.
        structure_rows: list[dict[str, Any]] = []
        atomic_rows: list[dict[str, Any]] = []
        convergence_rows: list[dict[str, Any]] = []
        ase_frames: list[Atoms] = []

        structure_hashes: list[str] = []
        for frame_index in range(EXPECTED_STEPS):
            gaussian_step = frame_index + 1
            structure_hash = sha256_structure(
                cclib_atomnos,
                cclib_coords[frame_index],
            )
            structure_hashes.append(structure_hash)

            corrected_force_rms_au = float(
                np.sqrt(np.mean(corrected_forces_au[frame_index] ** 2))
            )
            corrected_force_max_component_au = float(
                np.max(np.abs(corrected_forces_au[frame_index]))
            )
            all_converged = bool(np.all(conv_flags[frame_index]))

            structure_rows.append(
                {
                    "frame_index_0based": frame_index,
                    "gaussian_step_1based": gaussian_step,
                    "structure_sha256": structure_hash,
                    "scf_energy_hartree": raw_energies_hartree[frame_index],
                    "scf_energy_eV": authoritative_energies_ev[frame_index],
                    "relative_energy_eV_to_lowest_B3LYP_sample": (
                        relative_energy_ev[frame_index]
                    ),
                    "raw_max_force_hartree_per_bohr": raw_max_force[frame_index],
                    "raw_rms_force_hartree_per_bohr": raw_rms_force[frame_index],
                    "corrected_max_force_component_hartree_per_bohr": (
                        corrected_force_max_component_au
                    ),
                    "corrected_rms_force_hartree_per_bohr": corrected_force_rms_au,
                    "maximum_displacement_gaussian_au": conv_values[frame_index, 2],
                    "rms_displacement_gaussian_au": conv_values[frame_index, 3],
                    "maximum_force_converged": conv_flags[frame_index, 0],
                    "rms_force_converged": conv_flags[frame_index, 1],
                    "maximum_displacement_converged": conv_flags[frame_index, 2],
                    "rms_displacement_converged": conv_flags[frame_index, 3],
                    "all_four_criteria_converged": all_converged,
                    "cclib_optstatus_raw": cclib_optstatus[frame_index],
                    "input_to_standard_alignment_rmsd_A": alignment_rmsds[
                        frame_index
                    ],
                }
            )

            convergence_rows.append(
                {
                    "frame_index_0based": frame_index,
                    "gaussian_step_1based": gaussian_step,
                    "maximum_force_value_au": conv_values[frame_index, 0],
                    "maximum_force_target_au": conv_targets[frame_index, 0],
                    "maximum_force_converged": conv_flags[frame_index, 0],
                    "rms_force_value_au": conv_values[frame_index, 1],
                    "rms_force_target_au": conv_targets[frame_index, 1],
                    "rms_force_converged": conv_flags[frame_index, 1],
                    "maximum_displacement_value_au": conv_values[frame_index, 2],
                    "maximum_displacement_target_au": conv_targets[frame_index, 2],
                    "maximum_displacement_converged": conv_flags[frame_index, 2],
                    "rms_displacement_value_au": conv_values[frame_index, 3],
                    "rms_displacement_target_au": conv_targets[frame_index, 3],
                    "rms_displacement_converged": conv_flags[frame_index, 3],
                    "all_four_criteria_converged": bool(
                        np.all(conv_flags[frame_index])
                    ),
                }
            )

            for atom_index in range(EXPECTED_ATOMS):
                atom_index_1based = atom_index + 1
                if atom_index < 30:
                    molecular_group = "tacrine"
                    water_index = None
                    atom_role = "solute"
                else:
                    molecular_group = "water"
                    water_index = ((atom_index - 30) // 3) + 1
                    atom_role = (
                        "water_O"
                        if (atom_index - 30) % 3 == 0
                        else "water_H"
                    )

                atomic_rows.append(
                    {
                        "frame_index_0based": frame_index,
                        "gaussian_step_1based": gaussian_step,
                        "structure_sha256": structure_hash,
                        "atom_index_1based": atom_index_1based,
                        "element": symbols[atom_index],
                        "molecular_group": molecular_group,
                        "water_index": water_index,
                        "atom_role": atom_role,
                        "x_standard_A": cclib_coords[frame_index, atom_index, 0],
                        "y_standard_A": cclib_coords[frame_index, atom_index, 1],
                        "z_standard_A": cclib_coords[frame_index, atom_index, 2],
                        "x_reconstructed_input_A": input_coords_a[
                            frame_index, atom_index, 0
                        ],
                        "y_reconstructed_input_A": input_coords_a[
                            frame_index, atom_index, 1
                        ],
                        "z_reconstructed_input_A": input_coords_a[
                            frame_index, atom_index, 2
                        ],
                        "Fx_raw_input_hartree_per_bohr": raw_forces_input_au[
                            frame_index, atom_index, 0
                        ],
                        "Fy_raw_input_hartree_per_bohr": raw_forces_input_au[
                            frame_index, atom_index, 1
                        ],
                        "Fz_raw_input_hartree_per_bohr": raw_forces_input_au[
                            frame_index, atom_index, 2
                        ],
                        "Fx_corrected_standard_hartree_per_bohr": (
                            corrected_forces_au[frame_index, atom_index, 0]
                        ),
                        "Fy_corrected_standard_hartree_per_bohr": (
                            corrected_forces_au[frame_index, atom_index, 1]
                        ),
                        "Fz_corrected_standard_hartree_per_bohr": (
                            corrected_forces_au[frame_index, atom_index, 2]
                        ),
                        "Fx_corrected_standard_eV_per_A": (
                            corrected_forces_ev_a[frame_index, atom_index, 0]
                        ),
                        "Fy_corrected_standard_eV_per_A": (
                            corrected_forces_ev_a[frame_index, atom_index, 1]
                        ),
                        "Fz_corrected_standard_eV_per_A": (
                            corrected_forces_ev_a[frame_index, atom_index, 2]
                        ),
                        "corrected_force_magnitude_eV_per_A": float(
                            np.linalg.norm(
                                corrected_forces_ev_a[frame_index, atom_index]
                            )
                        ),
                    }
                )

            atoms = Atoms(
                numbers=cclib_atomnos,
                positions=cclib_coords[frame_index],
                pbc=False,
            )
            atoms.info.update(
                {
                    "frame_index_0based": frame_index,
                    "gaussian_step_1based": gaussian_step,
                    "charge": int(data.charge),
                    "multiplicity": int(data.mult),
                    "method": "B3LYP",
                    "basis_set": "6-31G(d)",
                    "source_log_sha256": source_hash,
                    "structure_sha256": structure_hash,
                    "optimization_converged": False,
                    "all_four_criteria_converged": all_converged,
                    "reference_geometry_step_1based": lowest_index + 1,
                }
            )
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=float(authoritative_energies_ev[frame_index]),
                forces=corrected_forces_ev_a[frame_index],
            )
            ase_frames.append(atoms)

        report["checks"]["all_structure_hashes_unique"] = (
            len(set(structure_hashes)) == EXPECTED_STEPS
        )
        if not report["checks"]["all_structure_hashes_unique"]:
            raise ValueError("One or more evaluated Gaussian geometries are duplicated.")

        # Write authoritative outputs.
        pd.DataFrame(structure_rows).to_csv(
            output_paths["structures"], index=False
        )
        pd.DataFrame(atomic_rows).to_csv(
            output_paths["atomic_forces"], index=False
        )
        pd.DataFrame(convergence_rows).to_csv(
            output_paths["convergence"], index=False
        )
        pd.DataFrame(orientation_rows).to_csv(
            output_paths["orientation_qc"], index=False
        )
        output_paths["xyz_comparison"].write_text(
            json.dumps(json_safe(supplied_xyz_result), indent=2),
            encoding="utf-8",
        )

        write(
            str(output_paths["extxyz"]),
            ase_frames,
            format="extxyz",
        )

        # Read the extxyz back through ASE and compare all key arrays.
        readback_frames = read(
            str(output_paths["extxyz"]),
            index=":",
            format="extxyz",
        )
        if not isinstance(readback_frames, list):
            readback_frames = [readback_frames]

        report["checks"]["extxyz_readback_frame_count_is_244"] = (
            len(readback_frames) == EXPECTED_STEPS
        )
        readback_position_difference = 0.0
        readback_energy_difference = 0.0
        readback_force_difference = 0.0

        if report["checks"]["extxyz_readback_frame_count_is_244"]:
            for frame_index, atoms in enumerate(readback_frames):
                readback_position_difference = max(
                    readback_position_difference,
                    float(
                        np.max(
                            np.abs(
                                atoms.positions - cclib_coords[frame_index]
                            )
                        )
                    ),
                )
                readback_energy_difference = max(
                    readback_energy_difference,
                    abs(
                        float(atoms.get_potential_energy())
                        - float(authoritative_energies_ev[frame_index])
                    ),
                )
                readback_force_difference = max(
                    readback_force_difference,
                    float(
                        np.max(
                            np.abs(
                                atoms.get_forces()
                                - corrected_forces_ev_a[frame_index]
                            )
                        )
                    ),
                )

        report["checks"]["extxyz_positions_read_back_exactly"] = (
            readback_position_difference <= EXTXYZ_POSITION_READBACK_TOL_A
        )
        report["checks"]["extxyz_energies_read_back_exactly"] = (
            readback_energy_difference <= EXTXYZ_ENERGY_READBACK_TOL_EV
        )
        report["checks"]["extxyz_forces_read_back_exactly"] = (
            readback_force_difference <= EXTXYZ_FORCE_READBACK_TOL_EV_PER_A
        )

        if not all(
            report["checks"][key]
            for key in (
                "extxyz_readback_frame_count_is_244",
                "extxyz_positions_read_back_exactly",
                "extxyz_energies_read_back_exactly",
                "extxyz_forces_read_back_exactly",
            )
        ):
            raise ValueError(
                "ASE extxyz read-back validation failed: "
                f"positions={readback_position_difference:.3e}, "
                f"energies={readback_energy_difference:.3e}, "
                f"forces={readback_force_difference:.3e}."
            )

        report["extxyz_readback"] = {
            "frame_count": len(readback_frames),
            "maximum_position_difference_A": readback_position_difference,
            "position_tolerance_A": EXTXYZ_POSITION_READBACK_TOL_A,
            "maximum_energy_difference_eV": readback_energy_difference,
            "energy_tolerance_eV": EXTXYZ_ENERGY_READBACK_TOL_EV,
            "maximum_force_difference_eV_per_A": readback_force_difference,
            "force_tolerance_eV_per_A": EXTXYZ_FORCE_READBACK_TOL_EV_PER_A,
            "interpretation": (
                "The extxyz round-trip differences are limited to the decimal "
                "precision of ASE's text serialisation. Upstream authoritative "
                "arrays and orientation QC were evaluated before serialisation."
            ),
        }

        report["methodology"] = {
            "coordinate_source": (
                "cclib atomcoords; Gaussian Standard orientation; independently "
                "matched to raw Standard-orientation tables"
            ),
            "energy_source": (
                "raw SCF Done energies in Hartree converted to eV with ASE; "
                "cclib scfenergies independently validated using cclib's own "
                "Hartree-to-eV conversion"
            ),
            "force_source": (
                "independently parsed Gaussian force blocks in original axes, "
                "rotated to Standard orientation using reconstructed Cartesian "
                "Berny coordinates"
            ),
            "input_coordinate_reconstruction": (
                "Step 1 uses the first Berny Old X table; step k>1 uses New X "
                "from step k-1, which is the geometry evaluated at the next SCF point"
            ),
            "rotation_methods": (
                "independent Kabsch SVD and scipy Rotation.align_vectors"
            ),
            "energy_unit": "eV",
            "coordinate_unit": "angstrom",
            "force_unit": "eV/angstrom",
            "force_conversion_factor": Hartree / Bohr,
            "structure_hash_method": (
                "SHA256 of little-endian int32 atomic numbers followed by "
                "little-endian float64 Standard-orientation coordinates"
            ),
        }

        report["status"] = "PASS"
        output_paths["metadata"].write_text(
            json.dumps(json_safe(report), indent=2),
            encoding="utf-8",
        )
        write_text_report(output_paths["qc_report"], report)
        write_text_report(output_paths["run_log"], report)

        manifest_targets = [
            output_paths["extxyz"],
            output_paths["structures"],
            output_paths["atomic_forces"],
            output_paths["convergence"],
            output_paths["orientation_qc"],
            output_paths["metadata"],
            output_paths["xyz_comparison"],
            output_paths["qc_report"],
            output_paths["run_log"],
        ]
        manifest_rows = [
            {
                "filename": path.name,
                "relative_path": str(path.relative_to(stage)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in manifest_targets
        ]
        pd.DataFrame(manifest_rows).to_csv(
            output_paths["manifest"], index=False
        )

        print(output_paths["qc_report"].read_text(encoding="utf-8"))
        print("Authoritative outputs written:")
        for key, path in output_paths.items():
            print(f"  {key}: {path}")
        return 0

    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()

        # Always retain a failure report in the log directory.
        failure_log = log_dir / "02_extract_gaussian_path_corrected_FAILED.log"
        write_text_report(failure_log, report)
        print(failure_log.read_text(encoding="utf-8"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

