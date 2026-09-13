#!/usr/bin/env python
"""
CHE701P full Gaussian-path MLFF input package
=============================================

Creates 244 geometry-only XYZ files from the authoritative corrected Gaussian
path for fixed-geometry MACE, AIMNet2 and UMA inference.

Authoritative inputs
--------------------
02_reference_extraction/02_outputs/gaussian_path_corrected.extxyz
02_reference_extraction/02_outputs/gaussian_path_structures.csv
02_reference_extraction/02_outputs/gaussian_path_extraction_qc.txt

Outputs
-------
04_mlff_path_predictions/00_full_path_inputs/
    input_geometries/
        gaussian_step_001.xyz
        ...
        gaussian_step_244.xyz
    full_path_input_index.csv
    full_path_inputs_SHA256.csv
    full_path_input_report.txt
    full_path_input_metadata.json

The individual XYZ files contain coordinates and element ordering only.
B3LYP reference energies/forces remain in the authoritative corrected extxyz
and are linked through Gaussian step and structure SHA256.

Usage
-----
Activate the Gaussian analysis environment:

    conda activate che701p-gaussian-analysis

Then run:

    python 04_prepare_full_gaussian_path_inputs.py "%STAGE%"

Do not use --overwrite unless an earlier derived package has been archived.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read, write


DEFAULT_STAGE = None

EXPECTED_FRAMES = 244
EXPECTED_ATOMS = 78
EXPECTED_COMPOSITION = Counter({"C": 13, "H": 47, "N": 2, "O": 16})
POSITION_READBACK_TOL_A = 5.1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 244 geometry-only XYZ inputs from the corrected Gaussian path."
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
        help="Replace an existing derived full-path input package.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.stage is None:
        raise SystemExit(
            "A private 12_gaussian_path_benchmark stage path is required. "
            "Pass it explicitly as the first argument; raw Gaussian/project files are not redistributed."
        )
    stage = Path(args.stage).expanduser().resolve()

    source_dir = stage / "02_reference_extraction" / "02_outputs"
    source_extxyz = source_dir / "gaussian_path_corrected.extxyz"
    source_structures_csv = source_dir / "gaussian_path_structures.csv"
    source_qc = source_dir / "gaussian_path_extraction_qc.txt"

    output_dir = stage / "04_mlff_path_predictions" / "00_full_path_inputs"
    xyz_dir = output_dir / "input_geometries"
    index_csv = output_dir / "full_path_input_index.csv"
    manifest_csv = output_dir / "full_path_inputs_SHA256.csv"
    report_txt = output_dir / "full_path_input_report.txt"
    metadata_json = output_dir / "full_path_input_metadata.json"

    for required in (source_extxyz, source_structures_csv, source_qc):
        if not required.is_file():
            raise FileNotFoundError(f"Required authoritative input is missing: {required}")

    qc_text = source_qc.read_text(encoding="utf-8")
    if "Status: PASS" not in qc_text:
        raise RuntimeError("The corrected Gaussian extraction QC does not report PASS.")

    if output_dir.exists():
        existing_items = list(output_dir.iterdir())
        if existing_items and not args.overwrite:
            raise FileExistsError(
                "The full-path input package already contains files. Archive it or "
                "rerun deliberately with --overwrite:\n"
                + "\n".join(str(path) for path in existing_items)
            )
        if args.overwrite:
            shutil.rmtree(output_dir)

    xyz_dir.mkdir(parents=True, exist_ok=True)

    frames = read(str(source_extxyz), index=":", format="extxyz")
    if not isinstance(frames, list):
        frames = [frames]

    structure_rows = read_csv_rows(source_structures_csv)

    if len(frames) != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} extxyz frames; found {len(frames)}.")
    if len(structure_rows) != EXPECTED_FRAMES:
        raise ValueError(
            f"Expected {EXPECTED_FRAMES} structure-table rows; found {len(structure_rows)}."
        )

    required_columns = {
        "frame_index_0based",
        "gaussian_step_1based",
        "structure_sha256",
        "scf_energy_hartree",
        "scf_energy_eV",
        "relative_energy_eV_to_lowest_B3LYP_sample",
        "raw_max_force_hartree_per_bohr",
        "raw_rms_force_hartree_per_bohr",
    }
    missing = required_columns.difference(structure_rows[0])
    if missing:
        raise ValueError(f"Structure CSV is missing required columns: {sorted(missing)}")

    output_index_rows: list[dict[str, Any]] = []
    xyz_paths: list[Path] = []
    maximum_position_readback_difference = 0.0

    for expected_index, (frame, row) in enumerate(
        zip(frames, structure_rows, strict=True)
    ):
        frame_index = int(row["frame_index_0based"])
        gaussian_step = int(row["gaussian_step_1based"])

        if frame_index != expected_index:
            raise ValueError(
                f"Unexpected frame index at row {expected_index}: {frame_index}."
            )
        if gaussian_step != expected_index + 1:
            raise ValueError(
                f"Unexpected Gaussian step at row {expected_index}: {gaussian_step}."
            )
        if int(frame.info.get("gaussian_step_1based", -1)) != gaussian_step:
            raise ValueError(
                f"Extxyz/CSV Gaussian-step mismatch at frame {expected_index}."
            )
        if str(frame.info.get("structure_sha256", "")) != row["structure_sha256"]:
            raise ValueError(
                f"Extxyz/CSV structure-hash mismatch at Gaussian step {gaussian_step}."
            )
        if len(frame) != EXPECTED_ATOMS:
            raise ValueError(
                f"Gaussian step {gaussian_step}: expected {EXPECTED_ATOMS} atoms; "
                f"found {len(frame)}."
            )

        symbols = frame.get_chemical_symbols()
        if Counter(symbols) != EXPECTED_COMPOSITION:
            raise ValueError(
                f"Gaussian step {gaussian_step}: unexpected composition {Counter(symbols)}."
            )
        if np.any(frame.get_pbc()):
            raise ValueError(f"Gaussian step {gaussian_step}: unexpected PBC.")

        geometry_only = frame.copy()
        geometry_only.calc = None
        geometry_only.info = {}

        xyz_path = xyz_dir / f"gaussian_step_{gaussian_step:03d}.xyz"
        write(str(xyz_path), geometry_only, format="xyz")
        xyz_paths.append(xyz_path)

        reread = read(str(xyz_path), index=0)
        if reread.get_chemical_symbols() != symbols:
            raise ValueError(
                f"Gaussian step {gaussian_step}: element-order read-back failed."
            )

        position_difference = float(
            np.max(np.abs(reread.positions - geometry_only.positions))
        )
        maximum_position_readback_difference = max(
            maximum_position_readback_difference,
            position_difference,
        )
        if position_difference > POSITION_READBACK_TOL_A:
            raise ValueError(
                f"Gaussian step {gaussian_step}: position read-back difference "
                f"{position_difference:.3e} Å exceeds tolerance."
            )

        output_index_rows.append(
            {
                "frame_index_0based": frame_index,
                "gaussian_step_1based": gaussian_step,
                "input_filename": xyz_path.name,
                "input_file_sha256": sha256_file(xyz_path),
                "source_structure_sha256": row["structure_sha256"],
                "atom_count": len(frame),
                "composition": json.dumps(dict(sorted(Counter(symbols).items()))),
                "scf_energy_hartree_reference": row["scf_energy_hartree"],
                "scf_energy_eV_reference": row["scf_energy_eV"],
                "relative_energy_eV_to_lowest_B3LYP_sample": row[
                    "relative_energy_eV_to_lowest_B3LYP_sample"
                ],
                "raw_max_force_hartree_per_bohr_reference": row[
                    "raw_max_force_hartree_per_bohr"
                ],
                "raw_rms_force_hartree_per_bohr_reference": row[
                    "raw_rms_force_hartree_per_bohr"
                ],
                "coordinate_readback_max_abs_difference_A": position_difference,
            }
        )

    if len(xyz_paths) != EXPECTED_FRAMES:
        raise RuntimeError(
            f"Expected {EXPECTED_FRAMES} generated XYZ files; found {len(xyz_paths)}."
        )
    if len({path.name for path in xyz_paths}) != EXPECTED_FRAMES:
        raise RuntimeError("Generated XYZ filenames are not unique.")

    index_fields = [
        "frame_index_0based",
        "gaussian_step_1based",
        "input_filename",
        "input_file_sha256",
        "source_structure_sha256",
        "atom_count",
        "composition",
        "scf_energy_hartree_reference",
        "scf_energy_eV_reference",
        "relative_energy_eV_to_lowest_B3LYP_sample",
        "raw_max_force_hartree_per_bohr_reference",
        "raw_rms_force_hartree_per_bohr_reference",
        "coordinate_readback_max_abs_difference_A",
    ]
    write_csv(index_csv, index_fields, output_index_rows)

    metadata = {
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Geometry-only input package for fixed-geometry MACE, AIMNet2 and "
            "UMA inference on all 244 corrected Gaussian-path structures."
        ),
        "source_extxyz": str(source_extxyz),
        "source_extxyz_sha256": sha256_file(source_extxyz),
        "source_structures_csv": str(source_structures_csv),
        "source_structures_csv_sha256": sha256_file(source_structures_csv),
        "source_qc_report": str(source_qc),
        "source_qc_report_sha256": sha256_file(source_qc),
        "generated_xyz_count": len(xyz_paths),
        "atoms_per_xyz": EXPECTED_ATOMS,
        "gaussian_steps": [1, EXPECTED_FRAMES],
        "maximum_coordinate_readback_difference_A": (
            maximum_position_readback_difference
        ),
        "coordinate_readback_tolerance_A": POSITION_READBACK_TOL_A,
        "scientific_limits": [
            "The 244 geometries are sequentially correlated points from one unconverged Gaussian path.",
            "The generated XYZ files contain geometry only.",
            "B3LYP reference energies and forces remain in the corrected authoritative extxyz and linked CSV.",
            "All subsequent model calculations must be fixed-geometry single-point inference.",
        ],
    }
    metadata_json.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "CHE701P full Gaussian-path MLFF input package",
        "=" * 47,
        "Status: PASS",
        f"Generated XYZ files: {len(xyz_paths)}",
        f"Expected XYZ files: {EXPECTED_FRAMES}",
        f"Atoms per structure: {EXPECTED_ATOMS}",
        "Gaussian steps: 1 to 244",
        (
            "Maximum coordinate read-back difference (A): "
            f"{maximum_position_readback_difference:.3e}"
        ),
        f"Read-back tolerance (A): {POSITION_READBACK_TOL_A:.3e}",
        "",
        "Use",
        "---",
        "Use input_geometries for fixed-geometry MLFF energy and force inference.",
        "Do not optimise, relax or propagate these structures.",
        "Match predictions to B3LYP references using gaussian_step_1based and",
        "source_structure_sha256 in full_path_input_index.csv.",
        "",
        "Scientific scope",
        "----------------",
        "These are 244 sequentially correlated geometries from one unconverged",
        "B3LYP/6-31G(d) checkpoint-continuation path. They are not 244 independent",
        "molecular configurations and do not validate the original 100 MD clusters.",
        "",
    ]
    report_txt.write_text("\n".join(report_lines), encoding="utf-8")

    manifest_targets = [index_csv, report_txt, metadata_json, *xyz_paths]
    manifest_rows = [
        {
            "filename": path.name,
            "relative_path": str(path.relative_to(output_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in manifest_targets
    ]
    write_csv(
        manifest_csv,
        ["filename", "relative_path", "bytes", "sha256"],
        manifest_rows,
    )

    print(report_txt.read_text(encoding="utf-8"))
    print(f"Input directory: {xyz_dir}")
    print(f"Index: {index_csv}")
    print(f"Manifest: {manifest_csv}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

