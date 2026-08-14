from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MEMBER_NAMES = [
    "aimnet2_charge_plus1_member0",
    "aimnet2_charge_plus1_member1",
    "aimnet2_charge_plus1_member2",
    "aimnet2_charge_plus1_member3",
]


def read_csv_by_filename(path: Path) -> dict[str, dict[str, str]]:
    """Read a structure-level CSV and index rows by XYZ filename."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    indexed = {row["filename"]: row for row in rows}

    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate filenames found in {path}")

    return indexed


def read_force_csv(
    path: Path,
) -> dict[tuple[str, int], dict[str, str]]:
    """Read an atomic-force CSV indexed by filename and atom number."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    indexed: dict[tuple[str, int], dict[str, str]] = {}

    for row in rows:
        key = (
            row["filename"],
            int(row["atom_index_1based"]),
        )

        if key in indexed:
            raise ValueError(f"Duplicate force record {key} in {path}")

        indexed[key] = row

    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine four AIMNet2 ensemble-member batches and "
            "calculate energy and force disagreement."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing the four member folders",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="New directory for ensemble-analysis outputs",
    )

    args = parser.parse_args()

    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.outdir}"
        )

    args.outdir.mkdir(parents=True, exist_ok=True)

    member_dirs = [
        args.root / member_name
        for member_name in MEMBER_NAMES
    ]

    for member_dir in member_dirs:
        if not member_dir.is_dir():
            raise FileNotFoundError(
                f"Missing member directory: {member_dir}"
            )

    structure_sets: list[dict[str, dict[str, str]]] = []
    force_sets: list[
        dict[tuple[str, int], dict[str, str]]
    ] = []
    metadata: list[dict] = []

    for member_dir in member_dirs:
        structure_sets.append(
            read_csv_by_filename(
                member_dir / "aimnet2_structure_results.csv"
            )
        )

        force_sets.append(
            read_force_csv(
                member_dir / "aimnet2_atomic_forces.csv"
            )
        )

        metadata.append(
            json.loads(
                (
                    member_dir / "aimnet2_batch_metadata.json"
                ).read_text(encoding="utf-8")
            )
        )

    reference_filenames = sorted(structure_sets[0])

    if len(reference_filenames) != 100:
        raise ValueError(
            f"Expected 100 structures, found "
            f"{len(reference_filenames)}"
        )

    for member_index, structure_set in enumerate(
        structure_sets,
        start=0,
    ):
        if sorted(structure_set) != reference_filenames:
            raise ValueError(
                f"Member {member_index} has a different "
                "structure set."
            )

    # Confirm that identical XYZ files were used by all members.
    for filename in reference_filenames:
        hashes = {
            structure_set[filename]["input_sha256"]
            for structure_set in structure_sets
        }

        if len(hashes) != 1:
            raise ValueError(
                f"Input hash mismatch for {filename}: {hashes}"
            )

    expected_force_keys = {
        (filename, atom_index)
        for filename in reference_filenames
        for atom_index in range(1, 79)
    }

    for member_index, force_set in enumerate(force_sets):
        if set(force_set) != expected_force_keys:
            missing = expected_force_keys - set(force_set)
            extra = set(force_set) - expected_force_keys

            raise ValueError(
                f"Force-record mismatch for member {member_index}. "
                f"Missing={len(missing)}, extra={len(extra)}"
            )

    structure_rows: list[dict] = []
    atomic_rows: list[dict] = []

    for filename in reference_filenames:
        energies = np.asarray(
            [
                float(structure_set[filename]["energy_eV"])
                for structure_set in structure_sets
            ],
            dtype=float,
        )

        member_relative_energies = np.asarray(
            [
                float(
                    structure_set[filename][
                        "relative_energy_eV"
                    ]
                )
                for structure_set in structure_sets
            ],
            dtype=float,
        )

        mean_energy = float(energies.mean())
        energy_std = float(energies.std(ddof=1))

        structure_force_disagreements: list[float] = []
        ensemble_mean_force_norms: list[float] = []
        component_stds: list[float] = []

        for atom_index in range(1, 79):
            member_vectors = np.asarray(
                [
                    [
                        float(
                            force_set[(filename, atom_index)][
                                "Fx_eV_per_A"
                            ]
                        ),
                        float(
                            force_set[(filename, atom_index)][
                                "Fy_eV_per_A"
                            ]
                        ),
                        float(
                            force_set[(filename, atom_index)][
                                "Fz_eV_per_A"
                            ]
                        ),
                    ]
                    for force_set in force_sets
                ],
                dtype=float,
            )

            symbols = {
                force_set[(filename, atom_index)]["element"]
                for force_set in force_sets
            }

            if len(symbols) != 1:
                raise ValueError(
                    f"Element mismatch for {filename}, "
                    f"atom {atom_index}"
                )

            symbol = next(iter(symbols))

            mean_vector = member_vectors.mean(axis=0)
            component_std = member_vectors.std(
                axis=0,
                ddof=1,
            )

            deviations = member_vectors - mean_vector

            rms_vector_disagreement = float(
                np.sqrt(
                    np.mean(
                        np.sum(deviations**2, axis=1)
                    )
                )
            )

            mean_force_norm = float(
                np.linalg.norm(mean_vector)
            )

            structure_force_disagreements.append(
                rms_vector_disagreement
            )
            ensemble_mean_force_norms.append(
                mean_force_norm
            )
            component_stds.extend(component_std.tolist())

            atomic_rows.append(
                {
                    "filename": filename,
                    "atom_index_1based": atom_index,
                    "element": symbol,
                    "mean_Fx_eV_per_A": float(mean_vector[0]),
                    "mean_Fy_eV_per_A": float(mean_vector[1]),
                    "mean_Fz_eV_per_A": float(mean_vector[2]),
                    "std_Fx_eV_per_A": float(component_std[0]),
                    "std_Fy_eV_per_A": float(component_std[1]),
                    "std_Fz_eV_per_A": float(component_std[2]),
                    "ensemble_mean_force_magnitude_eV_per_A":
                        mean_force_norm,
                    "rms_vector_disagreement_eV_per_A":
                        rms_vector_disagreement,
                }
            )

        structure_rows.append(
            {
                "filename": filename,
                "input_sha256":
                    structure_sets[0][filename]["input_sha256"],
                "charge": 1,
                "atom_count": 78,
                "member0_energy_eV": float(energies[0]),
                "member1_energy_eV": float(energies[1]),
                "member2_energy_eV": float(energies[2]),
                "member3_energy_eV": float(energies[3]),
                "ensemble_mean_energy_eV": mean_energy,
                "ensemble_energy_std_eV": energy_std,
                "mean_member_relative_energy_eV": float(
                    member_relative_energies.mean()
                ),
                "member_relative_energy_std_eV": float(
                    member_relative_energies.std(ddof=1)
                ),
                "ensemble_mean_maximum_force_eV_per_A":
                    float(max(ensemble_mean_force_norms)),
                "force_disagreement_rms_eV_per_A": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                structure_force_disagreements
                            )
                        )
                    )
                ),
                "force_disagreement_max_eV_per_A": float(
                    max(structure_force_disagreements)
                ),
                "mean_force_component_std_eV_per_A": float(
                    np.mean(component_stds)
                ),
                "maximum_force_component_std_eV_per_A": float(
                    np.max(component_stds)
                ),
            }
        )

    # Relative energy based on the ensemble-mean energy.
    minimum_mean_energy = min(
        row["ensemble_mean_energy_eV"]
        for row in structure_rows
    )

    for row in structure_rows:
        row["ensemble_mean_relative_energy_eV"] = (
            row["ensemble_mean_energy_eV"]
            - minimum_mean_energy
        )

    structure_rows.sort(key=lambda row: row["filename"])
    atomic_rows.sort(
        key=lambda row: (
            row["filename"],
            row["atom_index_1based"],
        )
    )

    structure_output = (
        args.outdir / "aimnet2_ensemble_structure_summary.csv"
    )
    atomic_output = (
        args.outdir / "aimnet2_ensemble_atomic_forces.csv"
    )

    with structure_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(structure_rows[0]),
        )
        writer.writeheader()
        writer.writerows(structure_rows)

    with atomic_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(atomic_rows[0]),
        )
        writer.writeheader()
        writer.writerows(atomic_rows)

    energy_uncertainty_order = sorted(
        structure_rows,
        key=lambda row:
            row["member_relative_energy_std_eV"],
        reverse=True,
    )

    force_uncertainty_order = sorted(
        structure_rows,
        key=lambda row:
            row["force_disagreement_rms_eV_per_A"],
        reverse=True,
    )

    selected_names = {
        row["filename"]
        for row in energy_uncertainty_order[:10]
    } | {
        row["filename"]
        for row in force_uncertainty_order[:10]
    }

    energy_rank = {
        row["filename"]: rank
        for rank, row in enumerate(
            energy_uncertainty_order,
            start=1,
        )
    }

    force_rank = {
        row["filename"]: rank
        for rank, row in enumerate(
            force_uncertainty_order,
            start=1,
        )
    }

    selected_rows = []

    for row in structure_rows:
        if row["filename"] not in selected_names:
            continue

        selected_rows.append(
            {
                "filename": row["filename"],
                "energy_uncertainty_rank":
                    energy_rank[row["filename"]],
                "force_uncertainty_rank":
                    force_rank[row["filename"]],
                "member_relative_energy_std_eV":
                    row["member_relative_energy_std_eV"],
                "force_disagreement_rms_eV_per_A":
                    row["force_disagreement_rms_eV_per_A"],
                "force_disagreement_max_eV_per_A":
                    row["force_disagreement_max_eV_per_A"],
                "ensemble_mean_relative_energy_eV":
                    row["ensemble_mean_relative_energy_eV"],
            }
        )

    selected_rows.sort(
        key=lambda row: min(
            row["energy_uncertainty_rank"],
            row["force_uncertainty_rank"],
        )
    )

    uncertain_output = (
        args.outdir / "top_uncertain_structures.csv"
    )

    with uncertain_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(selected_rows[0]),
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    energy_stds = np.asarray(
        [
            row["member_relative_energy_std_eV"]
            for row in structure_rows
        ],
        dtype=float,
    )

    force_disagreements = np.asarray(
        [
            row["force_disagreement_rms_eV_per_A"]
            for row in structure_rows
        ],
        dtype=float,
    )

    summary = {
        "status": "complete",
        "ensemble_members": [
            item["model"] for item in metadata
        ],
        "charge": 1,
        "structure_count": len(structure_rows),
        "atomic_force_rows": len(atomic_rows),
        "energy_uncertainty": {
            "metric":
                "sample standard deviation of member-relative energies",
            "mean_eV": float(energy_stds.mean()),
            "median_eV": float(np.median(energy_stds)),
            "maximum_eV": float(energy_stds.max()),
            "maximum_structure":
                energy_uncertainty_order[0]["filename"],
        },
        "force_uncertainty": {
            "metric":
                "RMS vector disagreement among four members",
            "mean_eV_per_A": float(
                force_disagreements.mean()
            ),
            "median_eV_per_A": float(
                np.median(force_disagreements)
            ),
            "maximum_eV_per_A": float(
                force_disagreements.max()
            ),
            "maximum_structure":
                force_uncertainty_order[0]["filename"],
        },
        "scientific_interpretation": (
            "Ensemble disagreement is an uncertainty indicator, "
            "not a calibrated error against DFT."
        ),
    }

    (
        args.outdir / "aimnet2_ensemble_analysis_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print()
    print(f"Saved ensemble analysis to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()