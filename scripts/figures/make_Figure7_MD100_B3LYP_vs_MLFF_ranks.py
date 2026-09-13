from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau


EXPECTED = {
    "MACE-OFF23-medium": {
        "column": "mace_rank",
        "rho": 0.9122952295229522,
        "tau": 0.7535353535353536,
    },
    "AIMNet2 ensemble": {
        "column": "aimnet2_rank",
        "rho": 0.9585598559855985,
        "tau": 0.8359595959595961,
    },
    "UMA/OMol": {
        "column": "uma_omol_rank",
        "rho": 0.9573957395739572,
        "tau": 0.8327272727272729,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the dissertation B3LYP-versus-MLFF energetic-rank "
            "comparison from public derived data."
        )
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory in which PNG and PDF outputs will be written.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_file = (
        repo_root
        / "data"
        / "b3lyp_md100"
        / "md100_b3lyp_mlff_structure_ranking.csv"
    )

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        raise FileNotFoundError(f"Required public data file not found: {data_file}")

    data = pd.read_csv(data_file)

    required = {
        "structure_id",
        "filename",
        "b3lyp_rank",
        "mace_rank",
        "aimnet2_rank",
        "uma_omol_rank",
    }

    missing = required.difference(data.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    if len(data) != 100:
        raise RuntimeError(f"Expected 100 structures; found {len(data)}")

    expected_ranks = set(range(1, 101))

    for column in [
        "b3lyp_rank",
        "mace_rank",
        "aimnet2_rank",
        "uma_omol_rank",
    ]:
        values = pd.to_numeric(data[column], errors="raise")
        if values.isna().any() or not values.eq(values.round()).all():
            raise RuntimeError(f"{column} contains non-integer or missing ranks.")
        ranks = set(values)
        if ranks != expected_ranks:
            raise RuntimeError(
                f"{column} does not contain each integer rank from 1 to 100 exactly once."
            )

    # Locked structural checks from the final benchmark.
    b3lyp_min = data.loc[data["b3lyp_rank"] == 1].iloc[0]
    if int(b3lyp_min["structure_id"]) != 57:
        raise RuntimeError("Unexpected B3LYP minimum; expected structure 57.")

    for column in ["mace_rank", "aimnet2_rank", "uma_omol_rank"]:
        if int(b3lyp_min[column]) != 2:
            raise RuntimeError(
                f"B3LYP minimum should be rank 2 under {column}; "
                f"found {b3lyp_min[column]}."
            )

    for column in ["mace_rank", "aimnet2_rank", "uma_omol_rank"]:
        model_min = data.loc[data[column] == 1].iloc[0]

        if int(model_min["structure_id"]) != 13:
            raise RuntimeError(
                f"Unexpected model minimum under {column}; expected structure 13."
            )

        if int(model_min["b3lyp_rank"]) != 2:
            raise RuntimeError(
                f"Structure 13 should be B3LYP rank 2; "
                f"found {model_min['b3lyp_rank']}."
            )

    calculated = {}

    for model, spec in EXPECTED.items():
        column = spec["column"]

        rho, _ = spearmanr(data["b3lyp_rank"], data[column])
        tau, _ = kendalltau(
            data["b3lyp_rank"],
            data[column],
            variant="b",
        )

        if abs(rho - spec["rho"]) > 1e-12:
            raise RuntimeError(
                f"{model} Spearman check failed: "
                f"{rho} != {spec['rho']}"
            )

        if abs(tau - spec["tau"]) > 1e-12:
            raise RuntimeError(
                f"{model} Kendall tau-b check failed: "
                f"{tau} != {spec['tau']}"
            )

        calculated[model] = (rho, tau)

    models = [
        ("MACE-OFF23-medium", "mace_rank", "A"),
        ("AIMNet2 ensemble", "aimnet2_rank", "B"),
        ("UMA/OMol", "uma_omol_rank", "C"),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.7),
        sharex=True,
        sharey=True,
    )

    for ax, (label, column, panel) in zip(axes, models):
        rho, tau = calculated[label]

        ax.scatter(
            data["b3lyp_rank"],
            data[column],
            s=17,
            alpha=0.78,
            edgecolors="none",
        )

        ax.plot(
            [1, 100],
            [1, 100],
            linestyle="--",
            linewidth=0.9,
        )

        ax.set_xlim(0, 101)
        ax.set_ylim(0, 101)
        ax.set_xticks([1, 25, 50, 75, 100])
        ax.set_yticks([1, 25, 50, 75, 100])

        ax.set_title(
            f"{panel}  {label}",
            fontsize=8.5,
        )

        ax.text(
            0.04,
            0.96,
            f"Spearman $\\rho$ = {rho:.3f}\n"
            f"Kendall $\\tau_b$ = {tau:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.4,
        )

        ax.set_xlabel(
            "B3LYP relative-energy rank",
            fontsize=8,
        )

        ax.tick_params(labelsize=7)

    axes[0].set_ylabel(
        "MLFF relative-energy rank",
        fontsize=8,
    )

    fig.text(
        0.5,
        -0.01,
        "Rank 1 denotes the lowest-energy configuration under each method.",
        ha="center",
        fontsize=7,
    )

    fig.tight_layout(
        pad=0.8,
        w_pad=0.7,
        rect=[0, 0.05, 1, 1],
    )

    png = outdir / "Figure7_MD100_B3LYP_vs_MLFF_ranks.png"
    pdf = outdir / "Figure7_MD100_B3LYP_vs_MLFF_ranks.pdf"

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("MD100 Figure 7 reproduction checks PASSED")
    print(f"Structures: {len(data)}")
    print("B3LYP minimum: structure 57")
    print("Shared MLFF minimum: structure 13")
    print()

    for model, (rho, tau) in calculated.items():
        print(
            f"{model}: "
            f"Spearman rho={rho:.6f}, "
            f"Kendall tau-b={tau:.6f}"
        )

    print()
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")


if __name__ == "__main__":
    main()
