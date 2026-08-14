#!/usr/bin/env python
"""
CHE701P MACE-OFF23 full Gaussian-path fixed-geometry inference
===============================================================

Runs MACE-OFF23-medium single-point energy and force inference on all 244
quality-controlled Gaussian-path geometries.

Scientific scope
----------------
MACE-OFF23 receives no explicit molecular charge input. The intended cluster
charge is +1, so this workflow is an exploratory applicability baseline rather
than a charge-aware calculation.

No geometry optimisation or molecular dynamics is performed.

Usage
-----
conda activate che701p-mace

python run_mace_gaussian_full_244.py ^
  --input-dir "<...>\\00_full_path_inputs\\input_geometries" ^
  --input-index "<...>\\00_full_path_inputs\\full_path_input_index.csv" ^
  --reference-extxyz "<...>\\02_reference_extraction\\02_outputs\\gaussian_path_corrected.extxyz" ^
  --outdir "<...>\\01_mace\\02_full_path"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import ase
import numpy as np
import torch
from ase.io import read
from mace.calculators import mace_off


EXPECTED_STRUCTURES = 244
EXPECTED_ATOMS = 78
EXPECTED_FORCE_ROWS = EXPECTED_STRUCTURES * EXPECTED_ATOMS
EXPECTED_COMPOSITION = Counter({"C": 13, "H": 47, "N": 2, "O": 16})
POSITION_MATCH_TOL_A = 5.1e-9

STRUCTURE_FIELDS = [
    "frame_index_0based",
    "gaussian_step_1based",
    "filename",
    "input_file_sha256",
    "source_structure_sha256",
    "atom_count",
    "composition",
    "model_family",
    "model_size",
    "device",
    "explicit_charge_input",
    "intended_system_charge_e",
    "total_energy_eV",
    "maximum_force_eV_per_A",
    "mean_force_eV_per_A",
    "rms_force_eV_per_A",
    "net_force_x_eV_per_A",
    "net_force_y_eV_per_A",
    "net_force_z_eV_per_A",
    "calculation_seconds",
    "status",
    "error",
]

FORCE_FIELDS = [
    "frame_index_0based",
    "gaussian_step_1based",
    "filename",
    "input_file_sha256",
    "source_structure_sha256",
    "atom_index_1based",
    "element",
    "molecular_group",
    "water_index",
    "atom_role",
    "force_x_eV_per_A",
    "force_y_eV_per_A",
    "force_z_eV_per_A",
    "force_magnitude_eV_per_A",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MACE-OFF23-medium on all 244 corrected Gaussian-path geometries."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--input-index", required=True, type=Path)
    parser.add_argument("--reference-extxyz", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def molecular_labels(atom_index_zero_based: int) -> tuple[str, int | None, str]:
    if atom_index_zero_based < 30:
        return "tacrine", None, "solute"
    water_index = ((atom_index_zero_based - 30) // 3) + 1
    role = "water_O" if (atom_index_zero_based - 30) % 3 == 0 else "water_H"
    return "water", water_index, role


def prepare_outputs(outdir: Path, overwrite: bool) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)

    outputs = {
        "structure_csv": outdir / "mace_gaussian_full_structure_results.csv",
        "forces_csv": outdir / "mace_gaussian_full_atomic_forces.csv",
        "failures_csv": outdir / "mace_gaussian_full_failures.csv",
        "metadata_json": outdir / "mace_gaussian_full_metadata.json",
        "summary_txt": outdir / "mace_gaussian_full_summary.txt",
        "manifest_csv": outdir / "mace_gaussian_full_outputs_SHA256.csv",
        "error_log": outdir / "logs" / "mace_gaussian_full_error.txt",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Derived outputs already exist. Archive them or rerun deliberately "
            "with --overwrite:\n" + "\n".join(str(path) for path in existing)
        )
    return outputs


def load_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input index has no header.")
        required = {
            "frame_index_0based",
            "gaussian_step_1based",
            "input_filename",
            "input_file_sha256",
            "source_structure_sha256",
            "atom_count",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Input index is missing columns: {sorted(missing)}")
        rows = list(reader)

    if len(rows) != EXPECTED_STRUCTURES:
        raise ValueError(
            f"Expected {EXPECTED_STRUCTURES} index rows; found {len(rows)}."
        )

    rows.sort(key=lambda row: int(row["gaussian_step_1based"]))
    expected_steps = list(range(1, EXPECTED_STRUCTURES + 1))
    actual_steps = [int(row["gaussian_step_1based"]) for row in rows]
    if actual_steps != expected_steps:
        raise ValueError("Input index does not contain sequential Gaussian steps 1-244.")
    return rows


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    input_index = args.input_index.expanduser().resolve()
    reference_extxyz = args.reference_extxyz.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()

    for path in (input_dir, input_index, reference_extxyz):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    outputs = prepare_outputs(outdir, args.overwrite)
    index_rows = load_index(input_index)

    xyz_files = sorted(input_dir.glob("*.xyz"))
    if len(xyz_files) != EXPECTED_STRUCTURES:
        raise ValueError(
            f"Expected {EXPECTED_STRUCTURES} XYZ files; found {len(xyz_files)}."
        )

    reference_frames = read(str(reference_extxyz), index=":", format="extxyz")
    if not isinstance(reference_frames, list):
        reference_frames = [reference_frames]
    if len(reference_frames) != EXPECTED_STRUCTURES:
        raise ValueError(
            f"Expected {EXPECTED_STRUCTURES} reference frames; "
            f"found {len(reference_frames)}."
        )

    print("CHE701P MACE-OFF23 FULL GAUSSIAN-PATH RUN")
    print("=" * 47)
    print(f"Structures: {EXPECTED_STRUCTURES}")
    print(f"Model: MACE-OFF23-{args.model}")
    print(f"Device: {args.device}")
    print("Explicit molecular charge input: none")
    print("Intended finite-cluster charge: +1")
    print("Fixed-geometry energies and forces only")
    print()

    load_start = time.perf_counter()
    calculator = mace_off(model=args.model, device=args.device)
    model_load_seconds = time.perf_counter() - load_start

    structure_rows: list[dict[str, Any]] = []
    force_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    for sequence_index, index_row in enumerate(index_rows, start=1):
        frame_index = int(index_row["frame_index_0based"])
        gaussian_step = int(index_row["gaussian_step_1based"])
        filename = index_row["input_filename"]
        input_file = input_dir / filename
        calculation_start = time.perf_counter()

        result: dict[str, Any] = {
            "frame_index_0based": frame_index,
            "gaussian_step_1based": gaussian_step,
            "filename": filename,
            "input_file_sha256": "",
            "source_structure_sha256": index_row["source_structure_sha256"],
            "atom_count": "",
            "composition": "",
            "model_family": "MACE-OFF23",
            "model_size": args.model,
            "device": args.device,
            "explicit_charge_input": "none",
            "intended_system_charge_e": 1,
            "total_energy_eV": "",
            "maximum_force_eV_per_A": "",
            "mean_force_eV_per_A": "",
            "rms_force_eV_per_A": "",
            "net_force_x_eV_per_A": "",
            "net_force_y_eV_per_A": "",
            "net_force_z_eV_per_A": "",
            "calculation_seconds": "",
            "status": "FAIL",
            "error": "",
        }

        try:
            if not input_file.is_file():
                raise FileNotFoundError(f"Missing indexed XYZ: {input_file}")

            actual_input_hash = sha256_file(input_file)
            result["input_file_sha256"] = actual_input_hash
            if actual_input_hash.lower() != index_row["input_file_sha256"].lower():
                raise ValueError("Input XYZ SHA256 does not match the input index.")

            atoms = read(str(input_file), index=0)
            reference = reference_frames[frame_index]
            symbols = atoms.get_chemical_symbols()

            if len(atoms) != EXPECTED_ATOMS:
                raise ValueError(f"Expected {EXPECTED_ATOMS} atoms; found {len(atoms)}.")
            if Counter(symbols) != EXPECTED_COMPOSITION:
                raise ValueError(f"Unexpected composition: {dict(Counter(symbols))}")
            if symbols != reference.get_chemical_symbols():
                raise ValueError("Element order differs from authoritative reference.")
            if str(reference.info.get("structure_sha256", "")) != index_row[
                "source_structure_sha256"
            ]:
                raise ValueError("Authoritative structure hash mismatch.")
            if np.any(atoms.get_pbc()):
                raise ValueError("Unexpected periodic boundary conditions.")

            position_difference = float(
                np.max(np.abs(atoms.positions - reference.positions))
            )
            if position_difference > POSITION_MATCH_TOL_A:
                raise ValueError(
                    "Geometry differs from authoritative reference: "
                    f"{position_difference:.3e} A."
                )

            atoms.calc = calculator
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)

            if not math.isfinite(energy):
                raise ValueError("Non-finite MACE energy.")
            if forces.shape != (EXPECTED_ATOMS, 3):
                raise ValueError(f"Unexpected force-array shape: {forces.shape}")
            if not np.all(np.isfinite(forces)):
                raise ValueError("Non-finite MACE force component.")

            force_magnitudes = np.linalg.norm(forces, axis=1)
            net_force = forces.sum(axis=0)
            elapsed = time.perf_counter() - calculation_start

            result.update(
                {
                    "atom_count": len(atoms),
                    "composition": json.dumps(
                        dict(sorted(Counter(symbols).items()))
                    ),
                    "total_energy_eV": energy,
                    "maximum_force_eV_per_A": float(force_magnitudes.max()),
                    "mean_force_eV_per_A": float(force_magnitudes.mean()),
                    "rms_force_eV_per_A": float(
                        np.sqrt(np.mean(force_magnitudes**2))
                    ),
                    "net_force_x_eV_per_A": float(net_force[0]),
                    "net_force_y_eV_per_A": float(net_force[1]),
                    "net_force_z_eV_per_A": float(net_force[2]),
                    "calculation_seconds": elapsed,
                    "status": "PASS",
                }
            )

            for atom_index, (symbol, vector, magnitude) in enumerate(
                zip(symbols, forces, force_magnitudes, strict=True)
            ):
                group, water_index, atom_role = molecular_labels(atom_index)
                force_rows.append(
                    {
                        "frame_index_0based": frame_index,
                        "gaussian_step_1based": gaussian_step,
                        "filename": filename,
                        "input_file_sha256": actual_input_hash,
                        "source_structure_sha256": index_row[
                            "source_structure_sha256"
                        ],
                        "atom_index_1based": atom_index + 1,
                        "element": symbol,
                        "molecular_group": group,
                        "water_index": water_index,
                        "atom_role": atom_role,
                        "force_x_eV_per_A": float(vector[0]),
                        "force_y_eV_per_A": float(vector[1]),
                        "force_z_eV_per_A": float(vector[2]),
                        "force_magnitude_eV_per_A": float(magnitude),
                    }
                )

        except Exception as exc:
            result["calculation_seconds"] = (
                time.perf_counter() - calculation_start
            )
            result["error"] = f"{type(exc).__name__}: {exc}"
            failure_rows.append(
                {
                    "frame_index_0based": frame_index,
                    "gaussian_step_1based": gaussian_step,
                    "filename": filename,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

        structure_rows.append(result)

        if sequence_index == 1 or sequence_index % 10 == 0 or sequence_index == 244:
            print(
                f"[{sequence_index:03d}/{EXPECTED_STRUCTURES}] "
                f"step {gaussian_step:03d}: {result['status']}"
            )

    total_calculation_seconds = time.perf_counter() - total_start
    pass_count = sum(row["status"] == "PASS" for row in structure_rows)

    overall_pass = (
        len(structure_rows) == EXPECTED_STRUCTURES
        and pass_count == EXPECTED_STRUCTURES
        and not failure_rows
        and len(force_rows) == EXPECTED_FORCE_ROWS
    )

    write_csv(outputs["structure_csv"], STRUCTURE_FIELDS, structure_rows)
    write_csv(outputs["forces_csv"], FORCE_FIELDS, force_rows)
    write_csv(
        outputs["failures_csv"],
        [
            "frame_index_0based",
            "gaussian_step_1based",
            "filename",
            "error_type",
            "error_message",
        ],
        failure_rows,
    )

    metadata = {
        "status": "PASS" if overall_pass else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "MACE-OFF23",
        "model_size": args.model,
        "device": args.device,
        "explicit_molecular_charge_input": None,
        "intended_finite_cluster_charge_e": 1,
        "expected_structures": EXPECTED_STRUCTURES,
        "successful_structures": pass_count,
        "failed_structures": len(failure_rows),
        "expected_atomic_force_rows": EXPECTED_FORCE_ROWS,
        "atomic_force_rows": len(force_rows),
        "model_load_seconds": model_load_seconds,
        "total_calculation_seconds": total_calculation_seconds,
        "mean_seconds_per_structure": (
            total_calculation_seconds / pass_count if pass_count else None
        ),
        "input_dir": str(input_dir),
        "input_index": str(input_index),
        "input_index_sha256": sha256_file(input_index),
        "reference_extxyz": str(reference_extxyz),
        "reference_extxyz_sha256": sha256_file(reference_extxyz),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "ase_version": ase.__version__,
        "torch_version": torch.__version__,
        "mace_torch_version": safe_version("mace-torch"),
        "fixed_geometry_only": True,
        "coordinates_optimised_or_propagated": False,
        "scientific_warning": (
            "MACE-OFF23 received no explicit total charge. The +1 tacrinium-"
            "(H2O)16 calculations are an exploratory applicability baseline."
        ),
    }
    outputs["metadata_json"].write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    summary = "\n".join(
        [
            "CHE701P MACE-OFF23 FULL GAUSSIAN-PATH RUN",
            "=" * 47,
            f"Status: {'PASS' if overall_pass else 'FAIL'}",
            f"Model: MACE-OFF23-{args.model}",
            f"Device: {args.device}",
            "Explicit molecular charge input: none",
            "Intended finite-cluster charge: +1",
            f"Expected structures: {EXPECTED_STRUCTURES}",
            f"Successful structures: {pass_count}",
            f"Failed structures: {len(failure_rows)}",
            f"Atomic force rows: {len(force_rows)}",
            f"Expected atomic force rows: {EXPECTED_FORCE_ROWS}",
            f"Model loading time (s): {model_load_seconds:.6f}",
            f"Total calculation time (s): {total_calculation_seconds:.6f}",
            "",
            "Scientific status",
            "-----------------",
            metadata["scientific_warning"],
            "Fixed-geometry single-point inference only.",
            "No geometry optimisation or molecular dynamics was performed.",
            "",
        ]
    )
    outputs["summary_txt"].write_text(summary, encoding="utf-8")

    manifest_targets = [
        outputs["structure_csv"],
        outputs["forces_csv"],
        outputs["failures_csv"],
        outputs["metadata_json"],
        outputs["summary_txt"],
    ]
    manifest_rows = [
        {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in manifest_targets
    ]
    write_csv(
        outputs["manifest_csv"],
        ["filename", "bytes", "sha256"],
        manifest_rows,
    )

    print()
    print(summary)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        error_text = traceback.format_exc()
        print(error_text, file=sys.stderr)
        try:
            parsed = parse_args()
            outdir = parsed.outdir.expanduser().resolve()
            (outdir / "logs").mkdir(parents=True, exist_ok=True)
            (outdir / "logs" / "mace_gaussian_full_error.txt").write_text(
                error_text, encoding="utf-8"
            )
        except Exception:
            pass
        raise SystemExit(1)
