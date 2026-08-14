from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "mace": "MACE-OFF23",
    "aimnet": "AIMNet2 ensemble",
    "uma": "UMA/OMol",
}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return pearson(x_rank, y_rank)


def vector_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    output = np.full(len(a), np.nan, dtype=float)
    mask = denominator > 1e-14
    output[mask] = np.sum(a[mask] * b[mask], axis=1) / denominator[mask]
    return output


def pair_metrics(
    x: np.ndarray,
    y: np.ndarray,
    metric_type: str,
) -> dict[str, float]:
    difference = y - x
    result = {
        "pearson_r": pearson(x, y),
        "spearman_rho": spearman(x, y),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_difference": float(
            np.max(np.abs(difference))
        ),
    }
    result["metric_type"] = metric_type
    return result


def scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    out: Path,
    point_size: float = 25,
) -> None:
    fig = plt.figure(figsize=(6.8, 5.6))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=point_size, alpha=0.65)

    minimum = float(min(np.min(x), np.min(y)))
    maximum = float(max(np.max(x), np.max(y)))
    padding = 0.04 * max(maximum - minimum, 1e-8)
    limits = [minimum - padding, maximum + padding]
    ax.plot(limits, limits, linestyle="--", linewidth=1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MACE-OFF23, the four-member AIMNet2 ensemble mean, "
            "and UMA/OMol on the same 100 charged N16 structures."
        )
    )
    parser.add_argument("--mace-results", required=True, type=Path)
    parser.add_argument("--mace-forces", required=True, type=Path)
    parser.add_argument("--aimnet-results", required=True, type=Path)
    parser.add_argument("--aimnet-forces", required=True, type=Path)
    parser.add_argument("--uma-results", required=True, type=Path)
    parser.add_argument("--uma-forces", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()

    for path in [
        args.mace_results,
        args.mace_forces,
        args.aimnet_results,
        args.aimnet_forces,
        args.uma_results,
        args.uma_forces,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.outdir}. "
            "Use a new output directory."
        )
    args.outdir.mkdir(parents=True, exist_ok=True)

    mace_s = pd.read_csv(args.mace_results)
    aim_s = pd.read_csv(args.aimnet_results)
    uma_s = pd.read_csv(args.uma_results)

    mace_s = mace_s[
        mace_s["calculation_ok"].astype(str).str.lower().eq("true")
    ].copy()
    uma_s = uma_s[uma_s["status"].eq("complete")].copy()

    structure = (
        mace_s[
            [
                "filename",
                "sha256",
                "relative_energy_eV",
                "maximum_force_eV_per_A",
                "calculation_seconds",
            ]
        ]
        .rename(
            columns={
                "sha256": "mace_sha256",
                "relative_energy_eV": "mace_relative_energy_eV",
                "maximum_force_eV_per_A": "mace_maximum_force_eV_per_A",
                "calculation_seconds": "mace_seconds",
            }
        )
        .merge(
            aim_s[
                [
                    "filename",
                    "input_sha256",
                    "ensemble_mean_relative_energy_eV",
                    "ensemble_mean_maximum_force_eV_per_A",
                    "member_relative_energy_std_eV",
                    "force_disagreement_rms_eV_per_A",
                ]
            ].rename(
                columns={
                    "input_sha256": "aimnet_sha256",
                    "ensemble_mean_relative_energy_eV": (
                        "aimnet_relative_energy_eV"
                    ),
                    "ensemble_mean_maximum_force_eV_per_A": (
                        "aimnet_maximum_force_eV_per_A"
                    ),
                }
            ),
            on="filename",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            uma_s[
                [
                    "filename",
                    "input_sha256",
                    "relative_energy_eV",
                    "maximum_force_eV_per_A",
                    "calculation_seconds",
                ]
            ].rename(
                columns={
                    "input_sha256": "uma_sha256",
                    "relative_energy_eV": "uma_relative_energy_eV",
                    "maximum_force_eV_per_A": (
                        "uma_maximum_force_eV_per_A"
                    ),
                    "calculation_seconds": "uma_seconds",
                }
            ),
            on="filename",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("filename")
        .reset_index(drop=True)
    )

    if len(structure) != 100:
        raise ValueError(
            f"Expected 100 matched structures; found {len(structure)}."
        )

    hash_match = (
        structure["mace_sha256"].eq(structure["aimnet_sha256"])
        & structure["mace_sha256"].eq(structure["uma_sha256"])
    )
    if not hash_match.all():
        bad = structure.loc[~hash_match, "filename"].tolist()
        raise ValueError(f"Input SHA256 mismatch for: {bad}")

    mace_f = pd.read_csv(args.mace_forces).rename(
        columns={"atom_index_1_based": "atom_index_1based"}
    )
    aim_f = pd.read_csv(args.aimnet_forces)
    uma_f = pd.read_csv(args.uma_forces)

    forces = (
        mace_f[
            [
                "filename",
                "atom_index_1based",
                "element",
                "force_x_eV_per_A",
                "force_y_eV_per_A",
                "force_z_eV_per_A",
            ]
        ]
        .rename(
            columns={
                "force_x_eV_per_A": "mace_fx",
                "force_y_eV_per_A": "mace_fy",
                "force_z_eV_per_A": "mace_fz",
            }
        )
        .merge(
            aim_f[
                [
                    "filename",
                    "atom_index_1based",
                    "element",
                    "mean_Fx_eV_per_A",
                    "mean_Fy_eV_per_A",
                    "mean_Fz_eV_per_A",
                    "rms_vector_disagreement_eV_per_A",
                ]
            ].rename(
                columns={
                    "element": "aimnet_element",
                    "mean_Fx_eV_per_A": "aimnet_fx",
                    "mean_Fy_eV_per_A": "aimnet_fy",
                    "mean_Fz_eV_per_A": "aimnet_fz",
                    "rms_vector_disagreement_eV_per_A": (
                        "aimnet_ensemble_force_disagreement"
                    ),
                }
            ),
            on=["filename", "atom_index_1based"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            uma_f[
                [
                    "filename",
                    "atom_index_1based",
                    "element",
                    "force_x_eV_per_A",
                    "force_y_eV_per_A",
                    "force_z_eV_per_A",
                ]
            ].rename(
                columns={
                    "element": "uma_element",
                    "force_x_eV_per_A": "uma_fx",
                    "force_y_eV_per_A": "uma_fy",
                    "force_z_eV_per_A": "uma_fz",
                }
            ),
            on=["filename", "atom_index_1based"],
            how="inner",
            validate="one_to_one",
        )
        .sort_values(["filename", "atom_index_1based"])
        .reset_index(drop=True)
    )

    if len(forces) != 7800:
        raise ValueError(
            f"Expected 7,800 matched atomic force rows; found {len(forces)}."
        )
    if not (
        forces["element"].eq(forces["aimnet_element"])
        & forces["element"].eq(forces["uma_element"])
    ).all():
        raise ValueError("Element labels do not match across model outputs.")

    forces["molecular_group"] = np.where(
        forces["atom_index_1based"] <= 30,
        "Tacrine",
        "Water",
    )

    structure.to_csv(
        args.outdir / "three_model_structure_comparison.csv",
        index=False,
    )
    forces.to_csv(
        args.outdir / "three_model_atomic_force_comparison.csv",
        index=False,
    )

    model_pairs = [
        ("mace", "aimnet"),
        ("mace", "uma"),
        ("aimnet", "uma"),
    ]

    pairwise_rows = []
    group_rows = []

    for model_x, model_y in model_pairs:
        energy_x = structure[
            f"{model_x}_relative_energy_eV"
        ].to_numpy(dtype=float)
        energy_y = structure[
            f"{model_y}_relative_energy_eV"
        ].to_numpy(dtype=float)

        energy_metrics = pair_metrics(
            energy_x,
            energy_y,
            "relative_energy_eV",
        )
        energy_metrics.update(
            {"model_x": model_x, "model_y": model_y}
        )
        pairwise_rows.append(energy_metrics)

        vector_x = forces[
            [f"{model_x}_fx", f"{model_x}_fy", f"{model_x}_fz"]
        ].to_numpy(dtype=float)
        vector_y = forces[
            [f"{model_y}_fx", f"{model_y}_fy", f"{model_y}_fz"]
        ].to_numpy(dtype=float)

        component_metrics = pair_metrics(
            vector_x.reshape(-1),
            vector_y.reshape(-1),
            "force_component_eV_per_A",
        )
        component_metrics.update(
            {"model_x": model_x, "model_y": model_y}
        )
        component_metrics[
            "mean_force_vector_cosine_similarity"
        ] = float(np.nanmean(vector_cosines(vector_x, vector_y)))
        component_metrics[
            "mean_force_vector_difference_eV_per_A"
        ] = float(np.mean(np.linalg.norm(vector_y - vector_x, axis=1)))
        pairwise_rows.append(component_metrics)

        for grouping in ["element", "molecular_group"]:
            for group_name, group in forces.groupby(grouping, sort=True):
                gx = group[
                    [
                        f"{model_x}_fx",
                        f"{model_x}_fy",
                        f"{model_x}_fz",
                    ]
                ].to_numpy(dtype=float)
                gy = group[
                    [
                        f"{model_y}_fx",
                        f"{model_y}_fy",
                        f"{model_y}_fz",
                    ]
                ].to_numpy(dtype=float)
                difference = gy - gx
                group_rows.append(
                    {
                        "model_x": model_x,
                        "model_y": model_y,
                        "grouping": grouping,
                        "group": group_name,
                        "atomic_rows": len(group),
                        "component_mae_eV_per_A": float(
                            np.mean(np.abs(difference.reshape(-1)))
                        ),
                        "component_rmse_eV_per_A": float(
                            np.sqrt(np.mean(difference.reshape(-1) ** 2))
                        ),
                        "mean_vector_cosine_similarity": float(
                            np.nanmean(vector_cosines(gx, gy))
                        ),
                        "mean_vector_difference_eV_per_A": float(
                            np.mean(np.linalg.norm(difference, axis=1))
                        ),
                    }
                )

        scatter_plot(
            energy_x,
            energy_y,
            f"{MODEL_LABELS[model_x]} relative energy (eV)",
            f"{MODEL_LABELS[model_y]} relative energy (eV)",
            (
                f"Relative-energy agreement: "
                f"{MODEL_LABELS[model_x]} vs {MODEL_LABELS[model_y]}"
            ),
            args.outdir
            / f"energy_{model_x}_vs_{model_y}.png",
        )

        scatter_plot(
            vector_x.reshape(-1),
            vector_y.reshape(-1),
            f"{MODEL_LABELS[model_x]} force component (eV A$^{{-1}}$)",
            f"{MODEL_LABELS[model_y]} force component (eV A$^{{-1}}$)",
            (
                f"Atomic force-component agreement: "
                f"{MODEL_LABELS[model_x]} vs {MODEL_LABELS[model_y]}"
            ),
            args.outdir
            / f"forces_{model_x}_vs_{model_y}.png",
            point_size=6,
        )

    pairwise = pd.DataFrame(pairwise_rows)
    grouped = pd.DataFrame(group_rows)
    pairwise.to_csv(
        args.outdir / "three_model_pairwise_metrics.csv",
        index=False,
    )
    grouped.to_csv(
        args.outdir / "three_model_group_metrics.csv",
        index=False,
    )

    fig = plt.figure(figsize=(9.0, 5.8))
    ax = fig.add_subplot(111)
    x = np.arange(len(structure))
    ax.plot(
        x,
        structure["mace_relative_energy_eV"],
        label=MODEL_LABELS["mace"],
    )
    ax.plot(
        x,
        structure["aimnet_relative_energy_eV"],
        label=MODEL_LABELS["aimnet"],
    )
    ax.plot(
        x,
        structure["uma_relative_energy_eV"],
        label=MODEL_LABELS["uma"],
    )
    ax.set_xlabel("N16 structure index")
    ax.set_ylabel("Relative energy (eV)")
    ax.set_title("Relative energies across 100 matched N16 structures")
    ax.grid(True, linewidth=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        args.outdir / "relative_energy_profiles_three_models.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)

    timing = pd.DataFrame(
        {
            "model": ["MACE-OFF23", "UMA/OMol"],
            "mean_seconds_per_structure": [
                float(structure["mace_seconds"].mean()),
                float(structure["uma_seconds"].mean()),
            ],
        }
    )
    timing.to_csv(
        args.outdir / "observed_inference_timings.csv",
        index=False,
    )

    summary = {
        "status": "complete",
        "structure_count": len(structure),
        "atomic_force_rows": len(forces),
        "input_hashes_match": bool(hash_match.all()),
        "pairwise_metrics": pairwise.to_dict(orient="records"),
        "scientific_interpretation": (
            "These pairwise statistics quantify agreement among pretrained "
            "models. They do not establish accuracy against an independent "
            "ab-initio reference. Raw absolute energies were not compared."
        ),
        "timing_caution": (
            "Observed wall times are workflow measurements and may not be "
            "directly comparable when software, model loading, or hardware differ."
        ),
    }
    (args.outdir / "three_model_comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nSaved outputs to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
