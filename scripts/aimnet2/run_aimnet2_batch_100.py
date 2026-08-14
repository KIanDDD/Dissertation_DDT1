from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import ase
import numpy as np
import torch
from ase.io import read
from aimnet.calculators import AIMNet2ASE


EXPECTED_COMPOSITION = {
    "C": 13,
    "H": 47,
    "N": 2,
    "O": 16,
}


def sha256_file(path: Path) -> str:
    """Return the SHA256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def validate_structure(path: Path):
    """Read and validate one N16 XYZ structure."""
    atoms = read(path)

    if len(atoms) != 78:
        raise ValueError(
            f"{path.name}: expected 78 atoms, found {len(atoms)}"
        )

    symbols = atoms.get_chemical_symbols()
    observed = {
        symbol: symbols.count(symbol)
        for symbol in sorted(set(symbols))
    }

    if observed != EXPECTED_COMPOSITION:
        raise ValueError(
            f"{path.name}: expected {EXPECTED_COMPOSITION}, "
            f"found {observed}"
        )

    if bool(np.asarray(atoms.pbc).any()):
        raise ValueError(
            f"{path.name}: unexpected periodic boundary conditions"
        )

    coordinates = np.asarray(atoms.positions, dtype=float)

    if not np.isfinite(coordinates).all():
        raise ValueError(
            f"{path.name}: non-finite coordinates detected"
        )

    return atoms, symbols


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed-geometry AIMNet2 energies and forces "
            "for a directory of N16 XYZ structures."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing the original N16 XYZ files",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory for batch results",
    )

    parser.add_argument(
        "--model",
        default="aimnet2",
        help="AIMNet2 model alias",
    )

    parser.add_argument(
        "--charge",
        type=int,
        default=1,
        help="Total charge assigned to every N16 cluster",
    )

    parser.add_argument(
        "--expected-count",
        type=int,
        default=100,
        help="Expected number of XYZ structures",
    )

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {args.input_dir}"
        )

    xyz_files = sorted(args.input_dir.glob("*.xyz"))

    if len(xyz_files) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} XYZ files, "
            f"but found {len(xyz_files)}"
        )

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}\n"
            "Use a new output directory to avoid overwriting results."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_results_path = (
        args.output_dir / "aimnet2_structure_results_raw.csv"
    )
    final_results_path = (
        args.output_dir / "aimnet2_structure_results.csv"
    )
    forces_path = (
        args.output_dir / "aimnet2_atomic_forces.csv"
    )
    errors_path = (
        args.output_dir / "aimnet2_errors.csv"
    )
    metadata_path = (
        args.output_dir / "aimnet2_batch_metadata.json"
    )
    summary_path = (
        args.output_dir / "aimnet2_batch_summary.json"
    )

    metadata = {
        "script": Path(__file__).name,
        "command": " ".join(sys.argv),
        "input_directory": str(args.input_dir.resolve()),
        "output_directory": str(args.output_dir.resolve()),
        "model": args.model,
        "charge": args.charge,
        "expected_structure_count": args.expected_count,
        "expected_atom_count": 78,
        "expected_composition": EXPECTED_COMPOSITION,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "aimnet_version": version("aimnet"),
        "ase_version": ase.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "calculation_type": (
            "Fixed-geometry AIMNet2 single-point energies and forces"
        ),
        "scientific_status": (
            "Exploratory pretrained-model output; "
            "not yet validated against VASP DFT."
        ),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    structure_fieldnames = [
        "structure_index",
        "filename",
        "input_sha256",
        "charge",
        "atom_count",
        "energy_eV",
        "maximum_force_eV_per_A",
        "mean_force_eV_per_A",
        "rms_force_eV_per_A",
        "net_force_x_eV_per_A",
        "net_force_y_eV_per_A",
        "net_force_z_eV_per_A",
        "calculation_time_s",
    ]

    force_fieldnames = [
        "structure_index",
        "filename",
        "atom_index_1based",
        "element",
        "Fx_eV_per_A",
        "Fy_eV_per_A",
        "Fz_eV_per_A",
        "force_magnitude_eV_per_A",
    ]

    error_fieldnames = [
        "structure_index",
        "filename",
        "error_type",
        "error_message",
    ]

    results: list[dict] = []
    errors: list[dict] = []

    batch_start = time.perf_counter()

    # Load the model once and reuse it for all structures.
    calculator = AIMNet2ASE(
        args.model,
        charge=args.charge,
    )

    with (
        raw_results_path.open(
            "w", newline="", encoding="utf-8"
        ) as raw_handle,
        forces_path.open(
            "w", newline="", encoding="utf-8"
        ) as forces_handle,
        errors_path.open(
            "w", newline="", encoding="utf-8"
        ) as errors_handle,
    ):
        raw_writer = csv.DictWriter(
            raw_handle,
            fieldnames=structure_fieldnames,
        )
        force_writer = csv.DictWriter(
            forces_handle,
            fieldnames=force_fieldnames,
        )
        error_writer = csv.DictWriter(
            errors_handle,
            fieldnames=error_fieldnames,
        )

        raw_writer.writeheader()
        force_writer.writeheader()
        error_writer.writeheader()

        for structure_index, xyz_path in enumerate(
            xyz_files,
            start=1,
        ):
            print(
                f"[{structure_index:03d}/{len(xyz_files):03d}] "
                f"{xyz_path.name}",
                flush=True,
            )

            try:
                atoms, symbols = validate_structure(xyz_path)
                atoms.calc = calculator

                calculation_start = time.perf_counter()

                energy = float(
                    atoms.get_potential_energy()
                )
                forces = np.asarray(
                    atoms.get_forces(),
                    dtype=float,
                )

                calculation_time = (
                    time.perf_counter() - calculation_start
                )

                if not np.isfinite(energy):
                    raise ValueError("Non-finite energy returned")

                if not np.isfinite(forces).all():
                    raise ValueError("Non-finite forces returned")

                force_norms = np.linalg.norm(
                    forces,
                    axis=1,
                )
                net_force = forces.sum(axis=0)

                row = {
                    "structure_index": structure_index,
                    "filename": xyz_path.name,
                    "input_sha256": sha256_file(xyz_path),
                    "charge": args.charge,
                    "atom_count": len(atoms),
                    "energy_eV": energy,
                    "maximum_force_eV_per_A": float(
                        force_norms.max()
                    ),
                    "mean_force_eV_per_A": float(
                        force_norms.mean()
                    ),
                    "rms_force_eV_per_A": float(
                        np.sqrt(np.mean(force_norms**2))
                    ),
                    "net_force_x_eV_per_A": float(net_force[0]),
                    "net_force_y_eV_per_A": float(net_force[1]),
                    "net_force_z_eV_per_A": float(net_force[2]),
                    "calculation_time_s": calculation_time,
                }

                results.append(row)
                raw_writer.writerow(row)
                raw_handle.flush()

                for atom_index, (
                    symbol,
                    force,
                    force_norm,
                ) in enumerate(
                    zip(symbols, forces, force_norms),
                    start=1,
                ):
                    force_writer.writerow(
                        {
                            "structure_index": structure_index,
                            "filename": xyz_path.name,
                            "atom_index_1based": atom_index,
                            "element": symbol,
                            "Fx_eV_per_A": float(force[0]),
                            "Fy_eV_per_A": float(force[1]),
                            "Fz_eV_per_A": float(force[2]),
                            "force_magnitude_eV_per_A": float(
                                force_norm
                            ),
                        }
                    )

                forces_handle.flush()

            except Exception as exc:
                error = {
                    "structure_index": structure_index,
                    "filename": xyz_path.name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }

                errors.append(error)
                error_writer.writerow(error)
                errors_handle.flush()

                print(
                    f"FAILED: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    total_wall_time = time.perf_counter() - batch_start

    if results:
        minimum_energy = min(
            row["energy_eV"] for row in results
        )

        final_fieldnames = structure_fieldnames + [
            "relative_energy_eV"
        ]

        with final_results_path.open(
            "w", newline="", encoding="utf-8"
        ) as final_handle:
            final_writer = csv.DictWriter(
                final_handle,
                fieldnames=final_fieldnames,
            )
            final_writer.writeheader()

            for row in results:
                final_row = dict(row)
                final_row["relative_energy_eV"] = (
                    row["energy_eV"] - minimum_energy
                )
                final_writer.writerow(final_row)

        energies = np.asarray(
            [row["energy_eV"] for row in results],
            dtype=float,
        )

        maximum_forces = np.asarray(
            [
                row["maximum_force_eV_per_A"]
                for row in results
            ],
            dtype=float,
        )

        summary = {
            "model": args.model,
            "charge": args.charge,
            "structures_expected": args.expected_count,
            "structures_completed": len(results),
            "structures_failed": len(errors),
            "force_rows_expected_if_complete": (
                args.expected_count * 78
            ),
            "energy_minimum_eV": float(energies.min()),
            "energy_maximum_eV": float(energies.max()),
            "energy_mean_eV": float(energies.mean()),
            "energy_range_eV": float(
                energies.max() - energies.min()
            ),
            "maximum_force_minimum_eV_per_A": float(
                maximum_forces.min()
            ),
            "maximum_force_maximum_eV_per_A": float(
                maximum_forces.max()
            ),
            "maximum_force_mean_eV_per_A": float(
                maximum_forces.mean()
            ),
            "total_batch_wall_time_s": total_wall_time,
            "status": (
                "complete"
                if len(results) == args.expected_count
                and not errors
                else "incomplete"
            ),
            "note": (
                "Exploratory charge-aware AIMNet2 outputs. "
                "Accuracy has not yet been established against VASP."
            ),
        }

    else:
        summary = {
            "model": args.model,
            "charge": args.charge,
            "structures_expected": args.expected_count,
            "structures_completed": 0,
            "structures_failed": len(errors),
            "total_batch_wall_time_s": total_wall_time,
            "status": "failed",
        }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print(json.dumps(summary, indent=2))
    print()
    print(f"Results saved to: {args.output_dir.resolve()}")

    if summary["status"] != "complete":
        raise SystemExit(
            "Batch did not complete successfully. "
            "Inspect aimnet2_errors.csv."
        )


if __name__ == "__main__":
    main()