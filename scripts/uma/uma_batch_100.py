from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.io import read
from fairchem.core import FAIRChemCalculator, pretrained_mlip


EXPECTED_COMPOSITION = {"C": 13, "H": 47, "N": 2, "O": 16}

RAW_RESULT_FIELDS = [
    "structure_index",
    "filename",
    "frame_number",
    "input_sha256",
    "charge",
    "spin_multiplicity",
    "atom_count",
    "composition",
    "total_energy_eV",
    "maximum_force_eV_per_A",
    "mean_force_eV_per_A",
    "rms_force_eV_per_A",
    "calculation_seconds",
    "status",
    "error",
]

FORCE_FIELDS = [
    "structure_index",
    "filename",
    "frame_number",
    "atom_index_1based",
    "element",
    "force_x_eV_per_A",
    "force_y_eV_per_A",
    "force_z_eV_per_A",
    "force_magnitude_eV_per_A",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def composition(symbols: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(symbols).items()))


def frame_number(filename: str) -> int | None:
    match = re.search(r"frame(\d+)", filename)
    return int(match.group(1)) if match else None


def initialise_csv(path: Path, fields: list[str]) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()


def completed_filenames(
    raw_results_path: Path,
    forces_path: Path,
) -> set[str]:
    if not raw_results_path.exists() or not forces_path.exists():
        return set()

    results = pd.read_csv(raw_results_path)
    forces = pd.read_csv(forces_path)

    successful = set(
        results.loc[results["status"].eq("complete"), "filename"]
    )
    force_counts = forces.groupby("filename").size()
    complete_forces = set(force_counts[force_counts.eq(78)].index)
    return successful & complete_forces


def append_dict(path: Path, fields: list[str], row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerow(row)
        handle.flush()


def append_force_rows(path: Path, rows: list[dict]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FORCE_FIELDS)
        writer.writerows(rows)
        handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable UMA/OMol fixed-geometry batch over the "
            "100 tacrinium-(H2O)16 XYZ structures."
        )
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--model", default="uma-s-1p2")
    parser.add_argument("--task", default="omol")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--charge", type=int, default=1)
    parser.add_argument("--spin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from complete structures already in the output CSV files.",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise NotADirectoryError(args.input_dir)

    xyz_files = sorted(args.input_dir.glob("cluster_N16_*.xyz"))
    if len(xyz_files) != 100:
        raise ValueError(
            f"Expected exactly 100 N16 XYZ files; found {len(xyz_files)}."
        )

    args.outdir.mkdir(parents=True, exist_ok=True)

    raw_results_path = args.outdir / "uma_structure_results_raw.csv"
    forces_path = args.outdir / "uma_atomic_forces.csv"
    errors_path = args.outdir / "uma_errors.csv"

    existing_nonempty = any(args.outdir.iterdir())
    if existing_nonempty and not args.resume:
        raise FileExistsError(
            f"Output directory is not empty: {args.outdir}. "
            "Use --resume or a new output directory."
        )

    initialise_csv(raw_results_path, RAW_RESULT_FIELDS)
    initialise_csv(forces_path, FORCE_FIELDS)
    initialise_csv(errors_path, ["filename", "error"])

    already_complete = (
        completed_filenames(raw_results_path, forces_path)
        if args.resume
        else set()
    )

    model_load_start = time.perf_counter()
    predictor = pretrained_mlip.get_predict_unit(
        args.model,
        device=args.device,
        seed=args.seed,
    )
    calculator = FAIRChemCalculator(
        predictor,
        task_name=args.task,
    )
    model_load_seconds = time.perf_counter() - model_load_start

    batch_start = time.perf_counter()

    for structure_index, xyz_path in enumerate(xyz_files, start=1):
        if xyz_path.name in already_complete:
            print(
                f"[{structure_index:03d}/100] SKIP complete: {xyz_path.name}",
                flush=True,
            )
            continue

        print(
            f"[{structure_index:03d}/100] RUN: {xyz_path.name}",
            flush=True,
        )
        start = time.perf_counter()

        try:
            atoms = read(xyz_path)
            atoms.pbc = False
            symbols = atoms.get_chemical_symbols()
            observed = composition(symbols)

            if len(atoms) != 78:
                raise ValueError(f"Expected 78 atoms; found {len(atoms)}.")
            if observed != EXPECTED_COMPOSITION:
                raise ValueError(
                    f"Expected {EXPECTED_COMPOSITION}; found {observed}."
                )
            if not np.isfinite(atoms.positions).all():
                raise ValueError("Input contains non-finite coordinates.")

            atoms.info.update(
                {"charge": args.charge, "spin": args.spin}
            )
            atoms.calc = calculator

            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)

            if not np.isfinite(energy):
                raise ValueError("Non-finite UMA energy.")
            if forces.shape != (78, 3) or not np.isfinite(forces).all():
                raise ValueError(
                    f"Invalid UMA forces with shape {forces.shape}."
                )

            force_norms = np.linalg.norm(forces, axis=1)
            elapsed = time.perf_counter() - start
            frame = frame_number(xyz_path.name)

            result_row = {
                "structure_index": structure_index,
                "filename": xyz_path.name,
                "frame_number": frame,
                "input_sha256": sha256_file(xyz_path),
                "charge": args.charge,
                "spin_multiplicity": args.spin,
                "atom_count": len(atoms),
                "composition": json.dumps(observed, sort_keys=True),
                "total_energy_eV": energy,
                "maximum_force_eV_per_A": float(force_norms.max()),
                "mean_force_eV_per_A": float(force_norms.mean()),
                "rms_force_eV_per_A": float(
                    np.sqrt(np.mean(force_norms**2))
                ),
                "calculation_seconds": elapsed,
                "status": "complete",
                "error": "",
            }
            append_dict(
                raw_results_path,
                RAW_RESULT_FIELDS,
                result_row,
            )

            force_rows = []
            for atom_index, (symbol, vector, magnitude) in enumerate(
                zip(symbols, forces, force_norms),
                start=1,
            ):
                force_rows.append(
                    {
                        "structure_index": structure_index,
                        "filename": xyz_path.name,
                        "frame_number": frame,
                        "atom_index_1based": atom_index,
                        "element": symbol,
                        "force_x_eV_per_A": float(vector[0]),
                        "force_y_eV_per_A": float(vector[1]),
                        "force_z_eV_per_A": float(vector[2]),
                        "force_magnitude_eV_per_A": float(magnitude),
                    }
                )
            append_force_rows(forces_path, force_rows)

            print(
                f"    complete in {elapsed:.3f} s; "
                f"E={energy:.6f} eV; "
                f"Fmax={force_norms.max():.6f} eV/A",
                flush=True,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start
            error_text = f"{type(exc).__name__}: {exc}"
            result_row = {
                "structure_index": structure_index,
                "filename": xyz_path.name,
                "frame_number": frame_number(xyz_path.name),
                "input_sha256": sha256_file(xyz_path),
                "charge": args.charge,
                "spin_multiplicity": args.spin,
                "atom_count": "",
                "composition": "",
                "total_energy_eV": "",
                "maximum_force_eV_per_A": "",
                "mean_force_eV_per_A": "",
                "rms_force_eV_per_A": "",
                "calculation_seconds": elapsed,
                "status": "failed",
                "error": error_text,
            }
            append_dict(
                raw_results_path,
                RAW_RESULT_FIELDS,
                result_row,
            )
            append_dict(
                errors_path,
                ["filename", "error"],
                {"filename": xyz_path.name, "error": error_text},
            )
            print(f"    FAILED: {error_text}", flush=True)

    total_batch_seconds = time.perf_counter() - batch_start

    raw = pd.read_csv(raw_results_path)
    raw = raw.drop_duplicates(
        subset=["filename"],
        keep="last",
    ).sort_values("structure_index")

    complete = raw[raw["status"].eq("complete")].copy()
    failed = raw[~raw["status"].eq("complete")].copy()

    if len(complete) > 0:
        minimum_energy = complete["total_energy_eV"].min()
        complete["relative_energy_eV"] = (
            complete["total_energy_eV"] - minimum_energy
        )
        complete["relative_energy_kJ_mol"] = (
            complete["relative_energy_eV"] * 96.4853321233
        )
        complete["energy_rank"] = complete[
            "total_energy_eV"
        ].rank(method="min").astype(int)
    else:
        minimum_energy = None

    final_columns = [
        "structure_index",
        "filename",
        "frame_number",
        "input_sha256",
        "charge",
        "spin_multiplicity",
        "atom_count",
        "composition",
        "total_energy_eV",
        "relative_energy_eV",
        "relative_energy_kJ_mol",
        "energy_rank",
        "maximum_force_eV_per_A",
        "mean_force_eV_per_A",
        "rms_force_eV_per_A",
        "calculation_seconds",
        "status",
        "error",
    ]
    for column in final_columns:
        if column not in complete.columns:
            complete[column] = np.nan

    complete[final_columns].to_csv(
        args.outdir / "uma_structure_results.csv",
        index=False,
    )

    forces = pd.read_csv(forces_path)
    forces = forces.drop_duplicates(
        subset=["filename", "atom_index_1based"],
        keep="last",
    ).sort_values(
        ["structure_index", "atom_index_1based"]
    )
    forces.to_csv(forces_path, index=False)

    summary = {
        "status": (
            "complete"
            if len(complete) == 100 and len(failed) == 0
            else "incomplete"
        ),
        "model": args.model,
        "task": args.task,
        "device": args.device,
        "seed": args.seed,
        "charge": args.charge,
        "spin_multiplicity": args.spin,
        "structures_expected": 100,
        "structures_completed": int(len(complete)),
        "structures_failed": int(len(failed)),
        "atomic_force_rows": int(len(forces)),
        "minimum_total_energy_eV": (
            float(minimum_energy)
            if minimum_energy is not None
            else None
        ),
        "model_load_seconds": model_load_seconds,
        "batch_wall_time_this_invocation_s": total_batch_seconds,
        "mean_calculation_seconds": (
            float(complete["calculation_seconds"].mean())
            if len(complete)
            else None
        ),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "fairchem_core_version": version("fairchem-core"),
        "ase_version": version("ase"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "scientific_status": (
            "Pretrained UMA/OMol predictions. Agreement with other ML "
            "models is not independent ab-initio validation."
        ),
    }

    (args.outdir / "uma_batch_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "model_identifier": args.model,
        "task_name": args.task,
        "charge_metadata": args.charge,
        "spin_multiplicity_metadata": args.spin,
        "input_directory": str(args.input_dir.resolve()),
        "output_directory": str(args.outdir.resolve()),
        "command_line": " ".join(sys.argv),
    }
    (args.outdir / "uma_batch_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nSaved outputs to: {args.outdir.resolve()}")

    if summary["status"] != "complete":
        raise SystemExit(
            "Batch is incomplete. Inspect uma_errors.csv and rerun with --resume."
        )


if __name__ == "__main__":
    main()
