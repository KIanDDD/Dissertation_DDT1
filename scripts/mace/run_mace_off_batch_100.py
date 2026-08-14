from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import ase
import numpy as np
import torch
from ase.io import read
from mace.calculators import mace_off


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_DIR.parent
    / "06_cluster_exports"
    / "N16_xyz"
)

RESULTS_DIR = PROJECT_DIR / "02_results"
LOGS_DIR = PROJECT_DIR / "03_logs"

RESULTS_CSV = RESULTS_DIR / "mace_off_batch_100_results.csv"
FORCES_CSV = RESULTS_DIR / "mace_off_batch_100_forces.csv"
METADATA_JSON = RESULTS_DIR / "mace_off_batch_100_metadata.json"
SUMMARY_LOG = LOGS_DIR / "mace_off_batch_100_summary.txt"
ERROR_LOG = LOGS_DIR / "mace_off_batch_100_error.txt"

EXPECTED_STRUCTURES = 100
EXPECTED_ATOMS = 78
EXPECTED_COMPOSITION = Counter(
    {
        "C": 13,
        "H": 47,
        "N": 2,
        "O": 16,
    }
)

MODEL_SIZE = "medium"
DEVICE = "cpu"

EV_TO_KJ_PER_MOL = 96.4853321233


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def extract_frame_number(filename: str) -> int | None:
    match = re.search(r"frame(\d+)", filename, flags=re.IGNORECASE)

    if match is None:
        return None

    return int(match.group(1))


def composition_dict(symbols: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(symbols).items()))


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Input directory was not found:\n{INPUT_DIR}"
        )

    xyz_files = sorted(INPUT_DIR.glob("*.xyz"))

    if len(xyz_files) != EXPECTED_STRUCTURES:
        raise RuntimeError(
            f"Expected {EXPECTED_STRUCTURES} XYZ files, "
            f"but found {len(xyz_files)}."
        )

    print("CHE701P MACE-OFF23 BATCH BASELINE")
    print("=" * 39)
    print("Input directory:", INPUT_DIR)
    print("Structures:", len(xyz_files))
    print("Model:", f"MACE-OFF23-{MODEL_SIZE}")
    print("Device:", DEVICE)
    print()
    print("Loading model once for all 100 structures...")

    model_load_start = time.perf_counter()

    calculator = mace_off(
        model=MODEL_SIZE,
        device=DEVICE,
    )

    model_load_seconds = time.perf_counter() - model_load_start

    print(f"Model loaded in {model_load_seconds:.3f} seconds.")
    print()

    structure_rows: list[dict[str, object]] = []
    force_rows: list[dict[str, object]] = []

    batch_start = time.perf_counter()

    for structure_index, input_file in enumerate(
        xyz_files,
        start=1,
    ):
        calculation_start = time.perf_counter()

        row: dict[str, object] = {
            "structure_index": structure_index,
            "filename": input_file.name,
            "frame_number": extract_frame_number(input_file.name),
            "sha256": sha256_file(input_file),
            "read_ok": False,
            "calculation_ok": False,
            "atom_count": "",
            "composition": "",
            "total_energy_eV": "",
            "energy_per_atom_eV": "",
            "relative_energy_eV": "",
            "relative_energy_kJ_mol": "",
            "energy_rank": "",
            "maximum_force_eV_per_A": "",
            "mean_force_eV_per_A": "",
            "rms_force_eV_per_A": "",
            "calculation_seconds": "",
            "error": "",
        }

        try:
            atoms = read(input_file, index=0)

            row["read_ok"] = True

            if len(atoms) != EXPECTED_ATOMS:
                raise RuntimeError(
                    f"Expected {EXPECTED_ATOMS} atoms, "
                    f"found {len(atoms)}."
                )

            counts = Counter(atoms.get_chemical_symbols())

            if counts != EXPECTED_COMPOSITION:
                raise RuntimeError(
                    f"Unexpected composition: {dict(counts)}"
                )

            if np.any(atoms.get_pbc()):
                raise RuntimeError(
                    "Unexpected periodic boundary conditions."
                )

            atoms.calc = calculator

            total_energy_ev = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)

            if forces.shape != (EXPECTED_ATOMS, 3):
                raise RuntimeError(
                    f"Unexpected force-array shape: {forces.shape}"
                )

            if not np.isfinite(total_energy_ev):
                raise RuntimeError("Energy is not finite.")

            if not np.all(np.isfinite(forces)):
                raise RuntimeError(
                    "At least one force component is not finite."
                )

            force_magnitudes = np.linalg.norm(forces, axis=1)

            calculation_seconds = (
                time.perf_counter() - calculation_start
            )

            row.update(
                {
                    "calculation_ok": True,
                    "atom_count": len(atoms),
                    "composition": json.dumps(
                        composition_dict(
                            atoms.get_chemical_symbols()
                        ),
                        sort_keys=True,
                    ),
                    "total_energy_eV": total_energy_ev,
                    "energy_per_atom_eV": (
                        total_energy_ev / len(atoms)
                    ),
                    "maximum_force_eV_per_A": float(
                        np.max(force_magnitudes)
                    ),
                    "mean_force_eV_per_A": float(
                        np.mean(force_magnitudes)
                    ),
                    "rms_force_eV_per_A": float(
                        np.sqrt(
                            np.mean(
                                np.square(force_magnitudes)
                            )
                        )
                    ),
                    "calculation_seconds": calculation_seconds,
                }
            )

            for atom_index, (
                symbol,
                force_vector,
                force_magnitude,
            ) in enumerate(
                zip(
                    atoms.get_chemical_symbols(),
                    forces,
                    force_magnitudes,
                    strict=True,
                ),
                start=1,
            ):
                force_rows.append(
                    {
                        "structure_index": structure_index,
                        "filename": input_file.name,
                        "frame_number": extract_frame_number(
                            input_file.name
                        ),
                        "atom_index_1_based": atom_index,
                        "element": symbol,
                        "force_x_eV_per_A": float(
                            force_vector[0]
                        ),
                        "force_y_eV_per_A": float(
                            force_vector[1]
                        ),
                        "force_z_eV_per_A": float(
                            force_vector[2]
                        ),
                        "force_magnitude_eV_per_A": float(
                            force_magnitude
                        ),
                    }
                )

        except Exception as exc:
            row["calculation_seconds"] = (
                time.perf_counter() - calculation_start
            )
            row["error"] = f"{type(exc).__name__}: {exc}"

        structure_rows.append(row)

        if (
            structure_index == 1
            or structure_index % 10 == 0
            or structure_index == len(xyz_files)
        ):
            status = (
                "PASS"
                if row["calculation_ok"]
                else "FAIL"
            )

            print(
                f"[{structure_index:3d}/"
                f"{len(xyz_files)}] "
                f"{input_file.name}: {status}"
            )

    total_batch_seconds = time.perf_counter() - batch_start

    successful_rows = [
        row
        for row in structure_rows
        if row["calculation_ok"]
    ]

    failed_rows = [
        row
        for row in structure_rows
        if not row["calculation_ok"]
    ]

    if successful_rows:
        minimum_energy_ev = min(
            float(row["total_energy_eV"])
            for row in successful_rows
        )

        energy_order = sorted(
            successful_rows,
            key=lambda row: float(row["total_energy_eV"]),
        )

        rank_by_filename = {
            str(row["filename"]): rank
            for rank, row in enumerate(
                energy_order,
                start=1,
            )
        }

        for row in successful_rows:
            relative_energy_ev = (
                float(row["total_energy_eV"])
                - minimum_energy_ev
            )

            row["relative_energy_eV"] = relative_energy_ev
            row["relative_energy_kJ_mol"] = (
                relative_energy_ev * EV_TO_KJ_PER_MOL
            )
            row["energy_rank"] = rank_by_filename[
                str(row["filename"])
            ]

    structure_fieldnames = [
        "structure_index",
        "filename",
        "frame_number",
        "sha256",
        "read_ok",
        "calculation_ok",
        "atom_count",
        "composition",
        "total_energy_eV",
        "energy_per_atom_eV",
        "relative_energy_eV",
        "relative_energy_kJ_mol",
        "energy_rank",
        "maximum_force_eV_per_A",
        "mean_force_eV_per_A",
        "rms_force_eV_per_A",
        "calculation_seconds",
        "error",
    ]

    with RESULTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=structure_fieldnames,
        )
        writer.writeheader()
        writer.writerows(structure_rows)

    force_fieldnames = [
        "structure_index",
        "filename",
        "frame_number",
        "atom_index_1_based",
        "element",
        "force_x_eV_per_A",
        "force_y_eV_per_A",
        "force_z_eV_per_A",
        "force_magnitude_eV_per_A",
    ]

    with FORCES_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=force_fieldnames,
        )
        writer.writeheader()
        writer.writerows(force_rows)

    overall_pass = (
        len(successful_rows) == EXPECTED_STRUCTURES
        and len(failed_rows) == 0
    )

    metadata = {
        "status": (
            "EXPLORATORY_BATCH_BASELINE_PASS"
            if overall_pass
            else "EXPLORATORY_BATCH_BASELINE_FAIL"
        ),
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_directory": str(INPUT_DIR),
        "expected_structures": EXPECTED_STRUCTURES,
        "successful_structures": len(successful_rows),
        "failed_structures": len(failed_rows),
        "atoms_per_structure": EXPECTED_ATOMS,
        "composition": dict(EXPECTED_COMPOSITION),
        "intended_total_charge_e": 1,
        "model_family": "MACE-OFF23",
        "model_size": MODEL_SIZE,
        "device": DEVICE,
        "model_load_seconds": model_load_seconds,
        "total_batch_seconds": total_batch_seconds,
        "mean_seconds_per_successful_structure": (
            total_batch_seconds / len(successful_rows)
            if successful_rows
            else None
        ),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "ase_version": ase.__version__,
        "mace_torch_version": version("mace-torch"),
        "scientific_warning": (
            "MACE-OFF23 is documented for neutral organic "
            "molecular systems. These clusters represent "
            "protonated tacrine with an intended total charge "
            "of +1. The batch is therefore an exploratory "
            "technical baseline and not validated reference data."
        ),
    }

    METADATA_JSON.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    if successful_rows:
        energies = np.asarray(
            [
                float(row["total_energy_eV"])
                for row in successful_rows
            ],
            dtype=float,
        )

        max_forces = np.asarray(
            [
                float(row["maximum_force_eV_per_A"])
                for row in successful_rows
            ],
            dtype=float,
        )

        lowest_energy_row = min(
            successful_rows,
            key=lambda row: float(row["total_energy_eV"]),
        )

        highest_energy_row = max(
            successful_rows,
            key=lambda row: float(row["total_energy_eV"]),
        )

        summary_lines = [
            "CHE701P MACE-OFF23 BATCH BASELINE",
            "=" * 39,
            f"Input directory: {INPUT_DIR}",
            f"Model: MACE-OFF23-{MODEL_SIZE}",
            f"Device: {DEVICE}",
            f"Expected structures: {EXPECTED_STRUCTURES}",
            f"Successful structures: {len(successful_rows)}",
            f"Failed structures: {len(failed_rows)}",
            f"Model loading time: {model_load_seconds:.3f} s",
            f"Total batch time: {total_batch_seconds:.3f} s",
            (
                "Mean time per successful structure: "
                f"{total_batch_seconds / len(successful_rows):.3f} s"
            ),
            "",
            "ENERGY SUMMARY",
            f"Minimum total energy: {energies.min():.12f} eV",
            f"Maximum total energy: {energies.max():.12f} eV",
            f"Mean total energy: {energies.mean():.12f} eV",
            f"Energy standard deviation: {energies.std(ddof=1):.12f} eV",
            f"Energy range: {np.ptp(energies):.12f} eV",
            (
                "Lowest-energy structure: "
                f"{lowest_energy_row['filename']}"
            ),
            (
                "Highest-energy structure: "
                f"{highest_energy_row['filename']}"
            ),
            "",
            "FORCE SUMMARY",
            (
                "Minimum of per-structure maximum forces: "
                f"{max_forces.min():.12f} eV/A"
            ),
            (
                "Maximum of per-structure maximum forces: "
                f"{max_forces.max():.12f} eV/A"
            ),
            (
                "Mean of per-structure maximum forces: "
                f"{max_forces.mean():.12f} eV/A"
            ),
            "",
            f"Results CSV: {RESULTS_CSV}",
            f"Forces CSV: {FORCES_CSV}",
            f"Metadata JSON: {METADATA_JSON}",
            "",
            "SCIENTIFIC STATUS:",
            metadata["scientific_warning"],
            "",
            (
                "BATCH STATUS: PASS"
                if overall_pass
                else "BATCH STATUS: FAIL"
            ),
        ]

    else:
        summary_lines = [
            "CHE701P MACE-OFF23 BATCH BASELINE",
            "=" * 39,
            "No structures were evaluated successfully.",
            "BATCH STATUS: FAIL",
        ]

    summary = "\n".join(summary_lines)
    SUMMARY_LOG.write_text(summary + "\n", encoding="utf-8")

    print()
    print(summary)

    if failed_rows:
        print("\nFailed structures:")

        for row in failed_rows:
            print(
                f"  {row['filename']}: {row['error']}"
            )

    if overall_pass:
        if ERROR_LOG.exists():
            ERROR_LOG.unlink()

        return 0

    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())

    except Exception:
        error_text = traceback.format_exc()

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ERROR_LOG.write_text(
            error_text,
            encoding="utf-8",
        )

        print("\nBATCH STATUS: FAIL")
        print(error_text)
        print("Error log:", ERROR_LOG)

        sys.exit(1)