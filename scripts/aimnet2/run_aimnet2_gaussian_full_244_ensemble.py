#!/usr/bin/env python
"""
CHE701P AIMNet2 four-member full Gaussian-path inference
=========================================================

Runs fixed-geometry AIMNet2 energy and force inference on all 244 corrected
Gaussian-path geometries using the verified four-member model set:

    aimnet2
    aimnet2-wb97m-d3_1
    aimnet2-wb97m-d3_2
    aimnet2-wb97m-d3_3

Total molecular charge: +1
Execution backend: eager CPU (TORCH_COMPILE_DISABLE=1)

No geometry optimisation or molecular dynamics is performed.

Expected outputs
----------------
Member structure rows: 244 * 4 = 976
Member atomic-force rows: 244 * 4 * 78 = 76128
Ensemble structure rows: 244
Ensemble atomic rows: 244 * 78 = 19032

Usage
-----
conda activate che701p-aimnet
set "TORCH_COMPILE_DISABLE=1"

python run_aimnet2_gaussian_full_244_ensemble.py ^
  --input-dir "<...>\\00_full_path_inputs\\input_geometries" ^
  --input-index "<...>\\00_full_path_inputs\\full_path_input_index.csv" ^
  --reference-extxyz "<...>\\02_reference_extraction\\02_outputs\\gaussian_path_corrected.extxyz" ^
  --outdir "<...>\\02_aimnet2\\02_full_path" ^
  --charge 1
"""

from __future__ import annotations

import os

# Must be set before importing torch/AIMNet2.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import ase
import numpy as np
import torch
from ase.io import read
from aimnet.calculators import AIMNet2ASE


EXPECTED_STRUCTURES = 244
EXPECTED_ATOMS = 78
EXPECTED_MEMBERS = 4
EXPECTED_MEMBER_STRUCTURE_ROWS = EXPECTED_STRUCTURES * EXPECTED_MEMBERS
EXPECTED_MEMBER_FORCE_ROWS = (
    EXPECTED_STRUCTURES * EXPECTED_MEMBERS * EXPECTED_ATOMS
)
EXPECTED_ENSEMBLE_STRUCTURE_ROWS = EXPECTED_STRUCTURES
EXPECTED_ENSEMBLE_ATOMIC_ROWS = EXPECTED_STRUCTURES * EXPECTED_ATOMS
EXPECTED_COMPOSITION = Counter({"C": 13, "H": 47, "N": 2, "O": 16})
POSITION_MATCH_TOL_A = 5.1e-9

DEFAULT_MODELS = [
    "aimnet2",
    "aimnet2-wb97m-d3_1",
    "aimnet2-wb97m-d3_2",
    "aimnet2-wb97m-d3_3",
]

MEMBER_STRUCTURE_FIELDS = [
    "frame_index_0based",
    "gaussian_step_1based",
    "filename",
    "input_file_sha256",
    "source_structure_sha256",
    "member_index_0based",
    "model_alias",
    "charge_e",
    "atom_count",
    "composition",
    "energy_eV",
    "maximum_force_eV_per_A",
    "mean_force_eV_per_A",
    "rms_force_eV_per_A",
    "net_force_x_eV_per_A",
    "net_force_y_eV_per_A",
    "net_force_z_eV_per_A",
    "model_load_seconds",
    "calculation_seconds",
    "status",
    "error",
]

MEMBER_FORCE_FIELDS = [
    "frame_index_0based",
    "gaussian_step_1based",
    "filename",
    "input_file_sha256",
    "source_structure_sha256",
    "member_index_0based",
    "model_alias",
    "charge_e",
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

ENSEMBLE_STRUCTURE_FIELDS = [
    "frame_index_0based",
    "gaussian_step_1based",
    "filename",
    "source_structure_sha256",
    "member_count",
    "charge_e",
    "energy_mean_eV",
    "energy_std_population_eV",
    "energy_range_eV",
    "maximum_member_force_mean_eV_per_A",
    "maximum_member_force_std_population_eV_per_A",
    "per_atom_vector_disagreement_mean_eV_per_A",
    "per_atom_vector_disagreement_rms_eV_per_A",
    "per_atom_vector_disagreement_max_eV_per_A",
]

ENSEMBLE_ATOMIC_FIELDS = [
    "frame_index_0based",
    "gaussian_step_1based",
    "filename",
    "source_structure_sha256",
    "member_count",
    "charge_e",
    "atom_index_1based",
    "element",
    "molecular_group",
    "water_index",
    "atom_role",
    "force_mean_x_eV_per_A",
    "force_mean_y_eV_per_A",
    "force_mean_z_eV_per_A",
    "force_mean_magnitude_eV_per_A",
    "force_component_std_x_eV_per_A",
    "force_component_std_y_eV_per_A",
    "force_component_std_z_eV_per_A",
    "force_vector_disagreement_rms_eV_per_A",
    "force_vector_disagreement_max_eV_per_A",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a four-member AIMNet2 ensemble on all 244 corrected "
            "Gaussian-path geometries."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--input-index", required=True, type=Path)
    parser.add_argument("--reference-extxyz", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--charge", type=int, default=1)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
    )
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
        "member_structure_csv": (
            outdir / "aimnet2_gaussian_full_member_structure_results.csv"
        ),
        "member_force_csv": (
            outdir / "aimnet2_gaussian_full_member_atomic_forces.csv"
        ),
        "ensemble_structure_csv": (
            outdir / "aimnet2_gaussian_full_ensemble_structure_summary.csv"
        ),
        "ensemble_atomic_csv": (
            outdir / "aimnet2_gaussian_full_ensemble_atomic_summary.csv"
        ),
        "failures_csv": outdir / "aimnet2_gaussian_full_failures.csv",
        "metadata_json": outdir / "aimnet2_gaussian_full_metadata.json",
        "summary_txt": outdir / "aimnet2_gaussian_full_summary.txt",
        "manifest_csv": outdir / "aimnet2_gaussian_full_outputs_SHA256.csv",
        "backend_txt": outdir / "AIMNet2_execution_backend.txt",
        "error_log": outdir / "logs" / "aimnet2_gaussian_full_error.txt",
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
    actual_steps = [int(row["gaussian_step_1based"]) for row in rows]
    if actual_steps != list(range(1, EXPECTED_STRUCTURES + 1)):
        raise ValueError("Input index is not the sequential Gaussian path 1-244.")
    return rows


def main() -> int:
    args = parse_args()

    if args.charge != 1:
        raise ValueError(f"This audited workflow requires charge +1; got {args.charge}.")
    if args.models != DEFAULT_MODELS:
        raise ValueError(
            "This audited workflow requires the verified four-model alias order: "
            + ", ".join(DEFAULT_MODELS)
        )
    if os.environ.get("TORCH_COMPILE_DISABLE") != "1":
        raise RuntimeError("TORCH_COMPILE_DISABLE must equal 1 for eager CPU execution.")

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

    # Validate every input once before loading any model.
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

    print("CHE701P AIMNET2 FOUR-MEMBER FULL GAUSSIAN-PATH RUN")
    print("=" * 56)
    print(f"Structures: {EXPECTED_STRUCTURES}")
    print(f"Members: {EXPECTED_MEMBERS}")
    print(f"Charge supplied: {args.charge:+d}")
    print("Execution backend: eager CPU")
    print("TORCH_COMPILE_DISABLE=1")
    print("Fixed-geometry energies and forces only")
    print()

    member_structure_rows: list[dict[str, Any]] = []
    member_force_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    model_load_times: dict[str, float] = {}

    energies_by_step: dict[int, list[float]] = defaultdict(list)
    max_forces_by_step: dict[int, list[float]] = defaultdict(list)
    forces_by_step: dict[int, list[np.ndarray]] = defaultdict(list)

    total_start = time.perf_counter()

    for member_index, model_alias in enumerate(args.models):
        load_start = time.perf_counter()
        calculator = AIMNet2ASE(model_alias, charge=args.charge)
        model_load_times[model_alias] = time.perf_counter() - load_start

        print(
            f"Member {member_index}: {model_alias} "
            f"(load {model_load_times[model_alias]:.3f} s)"
        )

        for sequence_index, item in enumerate(validated_inputs, start=1):
            start = time.perf_counter()
            step = item["step"]

            result: dict[str, Any] = {
                "frame_index_0based": item["frame_index"],
                "gaussian_step_1based": step,
                "filename": item["filename"],
                "input_file_sha256": item["input_file_sha256"],
                "source_structure_sha256": item["source_structure_sha256"],
                "member_index_0based": member_index,
                "model_alias": model_alias,
                "charge_e": args.charge,
                "atom_count": EXPECTED_ATOMS,
                "composition": json.dumps(
                    dict(sorted(Counter(item["symbols"]).items()))
                ),
                "energy_eV": "",
                "maximum_force_eV_per_A": "",
                "mean_force_eV_per_A": "",
                "rms_force_eV_per_A": "",
                "net_force_x_eV_per_A": "",
                "net_force_y_eV_per_A": "",
                "net_force_z_eV_per_A": "",
                "model_load_seconds": model_load_times[model_alias],
                "calculation_seconds": "",
                "status": "FAIL",
                "error": "",
            }

            try:
                atoms = item["atoms"].copy()
                atoms.calc = calculator

                energy = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces(), dtype=float)

                if not math.isfinite(energy):
                    raise ValueError("Non-finite AIMNet2 energy.")
                if forces.shape != (EXPECTED_ATOMS, 3):
                    raise ValueError(f"Unexpected force shape: {forces.shape}")
                if not np.all(np.isfinite(forces)):
                    raise ValueError("Non-finite AIMNet2 force component.")

                magnitudes = np.linalg.norm(forces, axis=1)
                net_force = forces.sum(axis=0)
                elapsed = time.perf_counter() - start

                result.update(
                    {
                        "energy_eV": energy,
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

                energies_by_step[step].append(energy)
                max_forces_by_step[step].append(float(magnitudes.max()))
                forces_by_step[step].append(forces.copy())

                for atom_index, (symbol, vector, magnitude) in enumerate(
                    zip(item["symbols"], forces, magnitudes, strict=True)
                ):
                    group, water_index, atom_role = molecular_labels(atom_index)
                    member_force_rows.append(
                        {
                            "frame_index_0based": item["frame_index"],
                            "gaussian_step_1based": step,
                            "filename": item["filename"],
                            "input_file_sha256": item["input_file_sha256"],
                            "source_structure_sha256": item[
                                "source_structure_sha256"
                            ],
                            "member_index_0based": member_index,
                            "model_alias": model_alias,
                            "charge_e": args.charge,
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
                        "member_index_0based": member_index,
                        "model_alias": model_alias,
                        "frame_index_0based": item["frame_index"],
                        "gaussian_step_1based": step,
                        "filename": item["filename"],
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

            member_structure_rows.append(result)

            if sequence_index == 1 or sequence_index % 25 == 0 or sequence_index == 244:
                print(
                    f"  [{sequence_index:03d}/{EXPECTED_STRUCTURES}] "
                    f"step {step:03d}: {result['status']}"
                )

    total_seconds = time.perf_counter() - total_start

    # Always write member/failure tables before attempting ensemble aggregation.
    write_csv(
        outputs["member_structure_csv"],
        MEMBER_STRUCTURE_FIELDS,
        member_structure_rows,
    )
    write_csv(
        outputs["member_force_csv"],
        MEMBER_FORCE_FIELDS,
        member_force_rows,
    )
    write_csv(
        outputs["failures_csv"],
        [
            "member_index_0based",
            "model_alias",
            "frame_index_0based",
            "gaussian_step_1based",
            "filename",
            "error_type",
            "error_message",
        ],
        failure_rows,
    )

    ensemble_structure_rows: list[dict[str, Any]] = []
    ensemble_atomic_rows: list[dict[str, Any]] = []

    input_by_step = {item["step"]: item for item in validated_inputs}

    for step in range(1, EXPECTED_STRUCTURES + 1):
        item = input_by_step[step]
        energies = np.asarray(energies_by_step[step], dtype=float)
        maximum_forces = np.asarray(max_forces_by_step[step], dtype=float)
        force_stack = np.asarray(forces_by_step[step], dtype=float)

        if energies.shape != (EXPECTED_MEMBERS,) or force_stack.shape != (
            EXPECTED_MEMBERS,
            EXPECTED_ATOMS,
            3,
        ):
            failure_rows.append(
                {
                    "member_index_0based": "",
                    "model_alias": "ensemble",
                    "frame_index_0based": item["frame_index"],
                    "gaussian_step_1based": step,
                    "filename": item["filename"],
                    "error_type": "IncompleteEnsemble",
                    "error_message": (
                        f"energy shape {energies.shape}; force shape {force_stack.shape}"
                    ),
                }
            )
            continue

        mean_forces = force_stack.mean(axis=0)
        component_std = force_stack.std(axis=0, ddof=0)
        deviations = force_stack - mean_forces[None, :, :]
        vector_deviation_norms = np.linalg.norm(deviations, axis=2)
        per_atom_rms_disagreement = np.sqrt(
            np.mean(vector_deviation_norms**2, axis=0)
        )
        per_atom_max_disagreement = np.max(vector_deviation_norms, axis=0)

        ensemble_structure_rows.append(
            {
                "frame_index_0based": item["frame_index"],
                "gaussian_step_1based": step,
                "filename": item["filename"],
                "source_structure_sha256": item["source_structure_sha256"],
                "member_count": EXPECTED_MEMBERS,
                "charge_e": args.charge,
                "energy_mean_eV": float(energies.mean()),
                "energy_std_population_eV": float(energies.std(ddof=0)),
                "energy_range_eV": float(energies.max() - energies.min()),
                "maximum_member_force_mean_eV_per_A": float(
                    maximum_forces.mean()
                ),
                "maximum_member_force_std_population_eV_per_A": float(
                    maximum_forces.std(ddof=0)
                ),
                "per_atom_vector_disagreement_mean_eV_per_A": float(
                    per_atom_rms_disagreement.mean()
                ),
                "per_atom_vector_disagreement_rms_eV_per_A": float(
                    np.sqrt(np.mean(per_atom_rms_disagreement**2))
                ),
                "per_atom_vector_disagreement_max_eV_per_A": float(
                    per_atom_rms_disagreement.max()
                ),
            }
        )

        mean_force_magnitudes = np.linalg.norm(mean_forces, axis=1)
        for atom_index in range(EXPECTED_ATOMS):
            group, water_index, atom_role = molecular_labels(atom_index)
            ensemble_atomic_rows.append(
                {
                    "frame_index_0based": item["frame_index"],
                    "gaussian_step_1based": step,
                    "filename": item["filename"],
                    "source_structure_sha256": item["source_structure_sha256"],
                    "member_count": EXPECTED_MEMBERS,
                    "charge_e": args.charge,
                    "atom_index_1based": atom_index + 1,
                    "element": item["symbols"][atom_index],
                    "molecular_group": group,
                    "water_index": water_index,
                    "atom_role": atom_role,
                    "force_mean_x_eV_per_A": float(mean_forces[atom_index, 0]),
                    "force_mean_y_eV_per_A": float(mean_forces[atom_index, 1]),
                    "force_mean_z_eV_per_A": float(mean_forces[atom_index, 2]),
                    "force_mean_magnitude_eV_per_A": float(
                        mean_force_magnitudes[atom_index]
                    ),
                    "force_component_std_x_eV_per_A": float(
                        component_std[atom_index, 0]
                    ),
                    "force_component_std_y_eV_per_A": float(
                        component_std[atom_index, 1]
                    ),
                    "force_component_std_z_eV_per_A": float(
                        component_std[atom_index, 2]
                    ),
                    "force_vector_disagreement_rms_eV_per_A": float(
                        per_atom_rms_disagreement[atom_index]
                    ),
                    "force_vector_disagreement_max_eV_per_A": float(
                        per_atom_max_disagreement[atom_index]
                    ),
                }
            )

    # Rewrite failure table if ensemble-level failures were added.
    write_csv(
        outputs["failures_csv"],
        [
            "member_index_0based",
            "model_alias",
            "frame_index_0based",
            "gaussian_step_1based",
            "filename",
            "error_type",
            "error_message",
        ],
        failure_rows,
    )
    write_csv(
        outputs["ensemble_structure_csv"],
        ENSEMBLE_STRUCTURE_FIELDS,
        ensemble_structure_rows,
    )
    write_csv(
        outputs["ensemble_atomic_csv"],
        ENSEMBLE_ATOMIC_FIELDS,
        ensemble_atomic_rows,
    )

    member_pass_count = sum(
        row["status"] == "PASS" for row in member_structure_rows
    )
    overall_pass = (
        len(member_structure_rows) == EXPECTED_MEMBER_STRUCTURE_ROWS
        and member_pass_count == EXPECTED_MEMBER_STRUCTURE_ROWS
        and len(member_force_rows) == EXPECTED_MEMBER_FORCE_ROWS
        and len(ensemble_structure_rows) == EXPECTED_ENSEMBLE_STRUCTURE_ROWS
        and len(ensemble_atomic_rows) == EXPECTED_ENSEMBLE_ATOMIC_ROWS
        and not failure_rows
    )

    outputs["backend_txt"].write_text(
        "\n".join(
            [
                "AIMNet2 execution backend: eager CPU",
                "TORCH_COMPILE_DISABLE=1",
                (
                    "Reason: TorchInductor required cl.exe, which was unavailable; "
                    "eager CPU inference passed without changing model weights, "
                    "structures or charge."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    metadata = {
        "status": "PASS" if overall_pass else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "models": args.models,
        "member_count": EXPECTED_MEMBERS,
        "charge_e": args.charge,
        "explicit_charge_supplied": True,
        "execution_backend": "eager CPU",
        "TORCH_COMPILE_DISABLE": os.environ.get("TORCH_COMPILE_DISABLE"),
        "member_structure_rows": len(member_structure_rows),
        "expected_member_structure_rows": EXPECTED_MEMBER_STRUCTURE_ROWS,
        "member_pass_rows": member_pass_count,
        "member_force_rows": len(member_force_rows),
        "expected_member_force_rows": EXPECTED_MEMBER_FORCE_ROWS,
        "ensemble_structure_rows": len(ensemble_structure_rows),
        "expected_ensemble_structure_rows": EXPECTED_ENSEMBLE_STRUCTURE_ROWS,
        "ensemble_atomic_rows": len(ensemble_atomic_rows),
        "expected_ensemble_atomic_rows": EXPECTED_ENSEMBLE_ATOMIC_ROWS,
        "failure_rows": len(failure_rows),
        "model_load_seconds": model_load_times,
        "total_wall_time_seconds": total_seconds,
        "input_dir": str(input_dir),
        "input_index": str(input_index),
        "input_index_sha256": sha256_file(input_index),
        "reference_extxyz": str(reference_extxyz),
        "reference_extxyz_sha256": sha256_file(reference_extxyz),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "aimnet_version": safe_version("aimnet"),
        "ase_version": ase.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "fixed_geometry_only": True,
        "coordinates_optimised_or_propagated": False,
        "scientific_warning": (
            "AIMNet2 explicitly received total charge +1. Four-model spread is "
            "ensemble disagreement, not calibrated uncertainty. Accuracy must be "
            "assessed against the path-local B3LYP/6-31G(d) reference."
        ),
    }
    outputs["metadata_json"].write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = "\n".join(
        [
            "CHE701P AIMNET2 FOUR-MEMBER FULL GAUSSIAN-PATH RUN",
            "=" * 56,
            f"Status: {'PASS' if overall_pass else 'FAIL'}",
            f"Charge supplied: {args.charge:+d}",
            f"Models: {', '.join(args.models)}",
            "Execution backend: eager CPU",
            "TORCH_COMPILE_DISABLE=1",
            f"Member structure rows: {len(member_structure_rows)} "
            f"(expected {EXPECTED_MEMBER_STRUCTURE_ROWS})",
            f"Member PASS rows: {member_pass_count}",
            f"Member force rows: {len(member_force_rows)} "
            f"(expected {EXPECTED_MEMBER_FORCE_ROWS})",
            f"Ensemble structure rows: {len(ensemble_structure_rows)} "
            f"(expected {EXPECTED_ENSEMBLE_STRUCTURE_ROWS})",
            f"Ensemble atomic rows: {len(ensemble_atomic_rows)} "
            f"(expected {EXPECTED_ENSEMBLE_ATOMIC_ROWS})",
            f"Failure rows: {len(failure_rows)}",
            f"Total wall time (s): {total_seconds:.6f}",
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
        outputs["member_structure_csv"],
        outputs["member_force_csv"],
        outputs["ensemble_structure_csv"],
        outputs["ensemble_atomic_csv"],
        outputs["failures_csv"],
        outputs["metadata_json"],
        outputs["summary_txt"],
        outputs["backend_txt"],
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
            (outdir / "logs" / "aimnet2_gaussian_full_error.txt").write_text(
                error_text, encoding="utf-8"
            )
        except Exception:
            pass
        raise SystemExit(1)
