#!/usr/bin/env python
"""
CHE701P UMA/OMol full Gaussian-path fixed-geometry inference
=============================================================

Runs UMA-S/OMol single-point energy and force inference on all 244 corrected
Gaussian-path geometries.

Audited settings
----------------
Model: uma-s-1p2
Task: omol
Device: cpu
Seed: 701
Total charge: +1
Spin multiplicity: 1

No geometry optimisation or molecular dynamics is performed.

Expected outputs
----------------
Structure rows: 244
Atomic-force rows: 244 * 78 = 19032

Usage
-----
conda activate che701p-uma

python run_uma_gaussian_full_244.py ^
  --input-dir "<...>\\00_full_path_inputs\\input_geometries" ^
  --input-index "<...>\\00_full_path_inputs\\full_path_input_index.csv" ^
  --reference-extxyz "<...>\\02_reference_extraction\\02_outputs\\gaussian_path_corrected.extxyz" ^
  --outdir "<...>\\03_uma\\02_full_path"
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
from fairchem.core import FAIRChemCalculator, pretrained_mlip


EXPECTED_STRUCTURES = 244
EXPECTED_ATOMS = 78
EXPECTED_FORCE_ROWS = EXPECTED_STRUCTURES * EXPECTED_ATOMS
EXPECTED_COMPOSITION = Counter({"C": 13, "H": 47, "N": 2, "O": 16})
POSITION_MATCH_TOL_A = 5.1e-9

DEFAULT_MODEL = "uma-s-1p2"
DEFAULT_TASK = "omol"
DEFAULT_DEVICE = "cpu"
DEFAULT_SEED = 701
DEFAULT_CHARGE = 1
DEFAULT_SPIN = 1

STRUCTURE_FIELDS = [
    "frame_index_0based",
    "gaussian_step_1based",
    "filename",
    "input_file_sha256",
    "source_structure_sha256",
    "atom_count",
    "composition",
    "model_name",
    "task_name",
    "device",
    "seed",
    "charge_e",
    "spin_multiplicity",
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
    "model_name",
    "task_name",
    "charge_e",
    "spin_multiplicity",
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
        description=(
            "Run UMA-S/OMol fixed-geometry inference on all 244 corrected "
            "Gaussian-path structures."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--input-index", required=True, type=Path)
    parser.add_argument("--reference-extxyz", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--charge", type=int, default=DEFAULT_CHARGE)
    parser.add_argument("--spin", type=int, default=DEFAULT_SPIN)
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
        "structure_csv": outdir / "uma_gaussian_full_structure_results.csv",
        "forces_csv": outdir / "uma_gaussian_full_atomic_forces.csv",
        "failures_csv": outdir / "uma_gaussian_full_failures.csv",
        "metadata_json": outdir / "uma_gaussian_full_metadata.json",
        "summary_txt": outdir / "uma_gaussian_full_summary.txt",
        "manifest_csv": outdir / "uma_gaussian_full_outputs_SHA256.csv",
        "error_log": outdir / "logs" / "uma_gaussian_full_error.txt",
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
    steps = [int(row["gaussian_step_1based"]) for row in rows]
    if steps != list(range(1, EXPECTED_STRUCTURES + 1)):
        raise ValueError("Input index is not the sequential Gaussian path 1-244.")
    return rows


def main() -> int:
    args = parse_args()

    if args.model != DEFAULT_MODEL:
        raise ValueError(
            f"This audited workflow requires model {DEFAULT_MODEL}; got {args.model}."
        )
    if args.task.lower() != DEFAULT_TASK:
        raise ValueError(
            f"This audited workflow requires task {DEFAULT_TASK}; got {args.task}."
        )
    if args.device.lower() != DEFAULT_DEVICE:
        raise ValueError(
            f"This audited workflow requires device {DEFAULT_DEVICE}; got {args.device}."
        )
    if args.seed != DEFAULT_SEED:
        raise ValueError(
            f"This audited workflow requires seed {DEFAULT_SEED}; got {args.seed}."
        )
    if args.charge != DEFAULT_CHARGE:
        raise ValueError(
            f"This audited cluster requires charge +{DEFAULT_CHARGE}; got {args.charge}."
        )
    if args.spin != DEFAULT_SPIN:
        raise ValueError(
            f"This audited cluster requires multiplicity {DEFAULT_SPIN}; got {args.spin}."
        )

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

    references = read(str(reference_extxyz), index=":", format="extxyz")
    if not isinstance(references, list):
        references = [references]
    if len(references) != EXPECTED_STRUCTURES:
        raise ValueError(
            f"Expected {EXPECTED_STRUCTURES} reference frames; found {len(references)}."
        )

    validated_inputs: list[dict[str, Any]] = []

    for row in index_rows:
        frame_index = int(row["frame_index_0based"])
        step = int(row["gaussian_step_1based"])
        input_file = input_dir / row["input_filename"]

        if not input_file.is_file():
            raise FileNotFoundError(f"Missing indexed XYZ: {input_file}")

        input_hash = sha256_file(input_file)
        if input_hash.lower() != row["input_file_sha256"].lower():
            raise ValueError(f"Step {step}: input XYZ SHA256 mismatch.")

        atoms = read(str(input_file), index=0)
        reference = references[frame_index]
        symbols = atoms.get_chemical_symbols()

        if len(atoms) != EXPECTED_ATOMS:
            raise ValueError(f"Step {step}: expected 78 atoms; found {len(atoms)}.")
        if Counter(symbols) != EXPECTED_COMPOSITION:
            raise ValueError(f"Step {step}: unexpected composition.")
        if symbols != reference.get_chemical_symbols():
            raise ValueError(f"Step {step}: atom order differs from reference.")
        if str(reference.info.get("structure_sha256", "")) != row[
            "source_structure_sha256"
        ]:
            raise ValueError(f"Step {step}: authoritative structure-hash mismatch.")
        if np.any(atoms.get_pbc()):
            raise ValueError(f"Step {step}: unexpected PBC.")

        difference = float(np.max(np.abs(atoms.positions - reference.positions)))
        if difference > POSITION_MATCH_TOL_A:
            raise ValueError(
                f"Step {step}: coordinate difference {difference:.3e} A."
            )

        validated_inputs.append(
            {
                "frame_index": frame_index,
                "step": step,
                "filename": row["input_filename"],
                "input_file_sha256": input_hash,
                "source_structure_sha256": row["source_structure_sha256"],
                "atoms": atoms,
                "symbols": symbols,
            }
        )

    print("CHE701P UMA/OMOL FULL GAUSSIAN-PATH RUN")
    print("=" * 44)
    print(f"Structures: {EXPECTED_STRUCTURES}")
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Charge supplied: {args.charge:+d}")
    print(f"Spin multiplicity supplied: {args.spin}")
    print("Fixed-geometry energies and forces only")
    print()

    load_start = time.perf_counter()
    predictor = pretrained_mlip.get_predict_unit(
        args.model,
        device=args.device,
        seed=args.seed,
    )
    calculator = FAIRChemCalculator(predictor, task_name=args.task)
    model_load_seconds = time.perf_counter() - load_start

    structure_rows: list[dict[str, Any]] = []
    force_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    total_start = time.perf_counter()

    for sequence_index, item in enumerate(validated_inputs, start=1):
        start = time.perf_counter()
        step = item["step"]

        result: dict[str, Any] = {
            "frame_index_0based": item["frame_index"],
            "gaussian_step_1based": step,
            "filename": item["filename"],
            "input_file_sha256": item["input_file_sha256"],
            "source_structure_sha256": item["source_structure_sha256"],
            "atom_count": EXPECTED_ATOMS,
            "composition": json.dumps(
                dict(sorted(Counter(item["symbols"]).items()))
            ),
            "model_name": args.model,
            "task_name": args.task,
            "device": args.device,
            "seed": args.seed,
            "charge_e": args.charge,
            "spin_multiplicity": args.spin,
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
            atoms = item["atoms"].copy()
            atoms.info.update(
                {
                    "charge": int(args.charge),
                    "spin": int(args.spin),
                }
            )
            atoms.calc = calculator

            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)

            if not math.isfinite(energy):
                raise ValueError("Non-finite UMA energy.")
            if forces.shape != (EXPECTED_ATOMS, 3):
                raise ValueError(f"Unexpected force-array shape: {forces.shape}")
            if not np.all(np.isfinite(forces)):
                raise ValueError("Non-finite UMA force component.")

            magnitudes = np.linalg.norm(forces, axis=1)
            net_force = forces.sum(axis=0)
            elapsed = time.perf_counter() - start

            result.update(
                {
                    "total_energy_eV": energy,
                    "maximum_force_eV_per_A": float(magnitudes.max()),
                    "mean_force_eV_per_A": float(magnitudes.mean()),
                    "rms_force_eV_per_A": float(
                        np.sqrt(np.mean(magnitudes**2))
                    ),
                    "net_force_x_eV_per_A": float(net_force[0]),
                    "net_force_y_eV_per_A": float(net_force[1]),
                    "net_force_z_eV_per_A": float(net_force[2]),
                    "calculation_seconds": elapsed,
                    "status": "PASS",
                }
            )

            for atom_index, (symbol, vector, magnitude) in enumerate(
                zip(item["symbols"], forces, magnitudes, strict=True)
            ):
                group, water_index, atom_role = molecular_labels(atom_index)
                force_rows.append(
                    {
                        "frame_index_0based": item["frame_index"],
                        "gaussian_step_1based": step,
                        "filename": item["filename"],
                        "input_file_sha256": item["input_file_sha256"],
                        "source_structure_sha256": item[
                            "source_structure_sha256"
                        ],
                        "model_name": args.model,
                        "task_name": args.task,
                        "charge_e": args.charge,
                        "spin_multiplicity": args.spin,
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
            result["calculation_seconds"] = time.perf_counter() - start
            result["error"] = f"{type(exc).__name__}: {exc}"
            failure_rows.append(
                {
                    "frame_index_0based": item["frame_index"],
                    "gaussian_step_1based": step,
                    "filename": item["filename"],
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

        structure_rows.append(result)

        if sequence_index == 1 or sequence_index % 10 == 0 or sequence_index == 244:
            print(
                f"[{sequence_index:03d}/{EXPECTED_STRUCTURES}] "
                f"step {step:03d}: {result['status']}"
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
        "model_name": args.model,
        "task_name": args.task,
        "device": args.device,
        "seed": args.seed,
        "charge_e": args.charge,
        "spin_multiplicity": args.spin,
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
        "fairchem_core_version": safe_version("fairchem-core"),
        "ase_version": ase.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "fixed_geometry_only": True,
        "coordinates_optimised_or_propagated": False,
        "scientific_warning": (
            "UMA/OMol explicitly received total charge +1 and singlet "
            "multiplicity 1. These are pretrained model predictions and are "
            "assessed only against the path-local B3LYP/6-31G(d) reference."
        ),
    }
    outputs["metadata_json"].write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = "\n".join(
        [
            "CHE701P UMA/OMOL FULL GAUSSIAN-PATH RUN",
            "=" * 44,
            f"Status: {'PASS' if overall_pass else 'FAIL'}",
            f"Model: {args.model}",
            f"Task: {args.task}",
            f"Device: {args.device}",
            f"Seed: {args.seed}",
            f"Charge supplied: {args.charge:+d}",
            f"Spin multiplicity supplied: {args.spin}",
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
            (outdir / "logs" / "uma_gaussian_full_error.txt").write_text(
                error_text,
                encoding="utf-8",
            )
        except Exception:
            pass
        raise SystemExit(1)
