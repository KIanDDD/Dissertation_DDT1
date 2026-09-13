from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ============================================================
# CHE701P — FINAL SECTION 3.2 FIGURES 9 AND 10
#
# Figure 9:
#   (a) B3LYP and pretrained-MLFF relative-energy profiles across
#       the 244 sequential Gaussian geometries.
#   (b) Absolute B3LYP-referenced relative-energy error.
#
# Figure 10:
#   (a) Overall and molecular-group RMS atomic force-vector error.
#   (b) Element-resolved RMS atomic force-vector error.
#
# Scientific boundary:
#   The 244 structures form one sequential, unconverged Gaussian
#   optimisation path. These figures support path-local comparison,
#   not independent validation or universal model accuracy.
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "b3lyp_benchmark"

ENERGY_CSV = DATA_DIR / "energy_predictions_and_errors.csv"
ENERGY_METRICS_CSV = DATA_DIR / "energy_metrics.csv"
FORCE_OVERALL_CSV = DATA_DIR / "force_overall_metrics.csv"
FORCE_GROUP_CSV = DATA_DIR / "force_molecular_group_metrics.csv"
FORCE_ELEMENT_CSV = DATA_DIR / "force_element_metrics.csv"

OUTDIR = Path(__file__).resolve().parent
OUTDIR.mkdir(parents=True, exist_ok=True)

energy = pd.read_csv(ENERGY_CSV)
energy_metrics = pd.read_csv(ENERGY_METRICS_CSV)
force_overall = pd.read_csv(FORCE_OVERALL_CSV)
force_group = pd.read_csv(FORCE_GROUP_CSV)
force_element = pd.read_csv(FORCE_ELEMENT_CSV)

# ------------------------------------------------------------
# Strict QC
# ------------------------------------------------------------
if len(energy) != 244:
    raise ValueError(f"Expected 244 geometries; found {len(energy)}")

steps = energy["gaussian_step_1based"].astype(int).tolist()
if steps != list(range(1, 245)):
    raise ValueError("Gaussian steps are not exactly 1–244.")

ref = energy.loc[energy["gaussian_step_1based"] == 227]
if len(ref) != 1:
    raise ValueError("Expected exactly one row for step 227.")

if not np.isclose(
    float(ref["b3lyp_relative_energy_eV"].iloc[0]),
    0.0,
    atol=1e-12,
):
    raise ValueError("B3LYP relative energy at step 227 is not zero.")

peak_idx = int(energy["b3lyp_relative_energy_eV"].idxmax())
peak_step = int(energy.loc[peak_idx, "gaussian_step_1based"])
peak_energy = float(energy.loc[peak_idx, "b3lyp_relative_energy_eV"])

if peak_step != 234:
    raise ValueError(
        f"Expected late B3LYP excursion at step 234; found {peak_step}"
    )

expected_minima = {
    "MACE-OFF23": 147,
    "AIMNet2 ensemble": 7,
    "UMA/OMol": 209,
}
for model, expected_step in expected_minima.items():
    row = energy_metrics.loc[energy_metrics["model"] == model]
    if len(row) != 1:
        raise ValueError(f"Missing unique energy-metrics row for {model}")
    got = int(row["predicted_minimum_step"].iloc[0])
    if got != expected_step:
        raise ValueError(
            f"{model} predicted minimum {got} != expected {expected_step}"
        )

# ------------------------------------------------------------
# House style
# ------------------------------------------------------------
B3LYP = "#222222"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRID = "#E6E6E6"
INK = "#1A1A1A"
AXIS = "#454545"
REFERENCE = "#A9A9A9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"
    ],
    "font.size": 9.2,
    "axes.linewidth": 0.85,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# ============================================================
# FIGURE 9
# ============================================================
x = energy["gaussian_step_1based"].to_numpy(int)
b3 = energy["b3lyp_relative_energy_eV"].to_numpy(float)

fig, axes = plt.subplots(
    2, 1,
    figsize=(7.55, 5.95),
    sharex=True,
    gridspec_kw={
        "height_ratios": [1.15, 0.90],
        "hspace": 0.18,
    },
)

series = [
    ("B3LYP/6-31G(d)", "b3lyp_relative_energy_eV", B3LYP, 1.85),
    ("MACE-OFF23", "mace_relative_energy_eV", BLUE, 1.15),
    ("AIMNet2 ensemble", "aimnet2_relative_energy_eV", ORANGE, 1.15),
    ("UMA/OMol", "uma_relative_energy_eV", GREEN, 1.15),
]

ax = axes[0]
handles, labels = [], []

for label, col, colour, linewidth in series:
    line, = ax.plot(
        x,
        energy[col].to_numpy(float),
        color=colour,
        linewidth=linewidth,
        zorder=5 if label.startswith("B3LYP") else 3,
    )
    handles.append(line)
    labels.append(label)

ax.axvline(
    227,
    color=REFERENCE,
    linewidth=0.85,
    zorder=1,
)

ax.set_title(
    "(a) Relative-energy profiles",
    loc="left",
    fontsize=9.0,
    pad=6,
)
ax.set_ylabel("Relative energy / eV")
ax.set_ylim(-0.055, 1.43)
ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
ax.set_axisbelow(True)

ax = axes[1]

error_map = [
    ("MACE-OFF23", "mace_relative_energy_eV", BLUE),
    ("AIMNet2 ensemble", "aimnet2_relative_energy_eV", ORANGE),
    ("UMA/OMol", "uma_relative_energy_eV", GREEN),
]

for label, pred_col, colour in error_map:
    error = np.abs(
        energy[pred_col].to_numpy(float) - b3
    )
    ax.plot(
        x,
        error,
        color=colour,
        linewidth=1.20,
        zorder=3,
    )

ax.axvline(
    227,
    color=REFERENCE,
    linewidth=0.85,
    zorder=1,
)

ax.set_title(
    "(b) Absolute B3LYP-referenced relative-energy error",
    loc="left",
    fontsize=9.0,
    pad=6,
)
ax.set_ylabel("Absolute error / eV")
ax.set_xlabel("Gaussian optimisation step")
ax.set_ylim(0.0, 0.62)
ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
ax.set_axisbelow(True)

for ax in axes:
    ax.set_xlim(1, 244)

axes[1].set_xticks([1, 50, 100, 150, 200, 227, 244])

# One shared legend only: avoids panel-title and legend clashes.
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=4,
    frameon=False,
    fontsize=7.9,
    handlelength=2.0,
    columnspacing=1.25,
)

fig.subplots_adjust(
    left=0.105,
    right=0.995,
    bottom=0.09,
    top=0.90,
)

fig.savefig(
    OUTDIR / "Figure9_B3LYP_energy_benchmark_TWO_PANEL_PERFECT_FINAL.png",
    dpi=600,
)
fig.savefig(
    OUTDIR / "Figure9_B3LYP_energy_benchmark_TWO_PANEL_PERFECT_FINAL.svg"
)
fig.savefig(
    OUTDIR / "Figure9_B3LYP_energy_benchmark_TWO_PANEL_PERFECT_FINAL.pdf"
)
plt.close(fig)

# ============================================================
# FIGURE 10
# ============================================================
models = ["MACE-OFF23", "AIMNet2 ensemble", "UMA/OMol"]
colours = [BLUE, ORANGE, GREEN]

overall_lookup = {
    row["model"]: float(row["rms_vector_error_eV_per_A"])
    for _, row in force_overall.iterrows()
}
group_lookup = {
    (row["model"], str(row["molecular_group"])):
        float(row["rms_vector_error_eV_per_A"])
    for _, row in force_group.iterrows()
}
element_lookup = {
    (row["model"], str(row["element"])):
        float(row["rms_vector_error_eV_per_A"])
    for _, row in force_element.iterrows()
}

left_categories = ["Overall", "Tacrinium", "Water"]
right_categories = ["C", "H", "N", "O"]

left_values = {
    model: [
        overall_lookup[model],
        group_lookup[(model, "tacrine")],
        group_lookup[(model, "water")],
    ]
    for model in models
}

right_values = {
    model: [
        element_lookup[(model, element)]
        for element in right_categories
    ]
    for model in models
}

fig, axes = plt.subplots(
    1, 2,
    figsize=(7.55, 3.45),
    sharex=True,
    gridspec_kw={
        "width_ratios": [0.92, 1.08],
        "wspace": 0.34,
    },
)

bar_height = 0.22
offsets = np.array([-bar_height, 0.0, bar_height])

for ax, categories, values_by_model, title in [
    (
        axes[0],
        left_categories,
        left_values,
        "(a) Overall and molecular-group error",
    ),
    (
        axes[1],
        right_categories,
        right_values,
        "(b) Element-resolved error",
    ),
]:
    y = np.arange(len(categories), dtype=float)

    for offset, model, colour in zip(
        offsets, models, colours
    ):
        values = np.asarray(
            values_by_model[model],
            dtype=float,
        )

        bars = ax.barh(
            y + offset,
            values,
            height=bar_height,
            color=colour,
            edgecolor="white",
            linewidth=0.45,
            label=model,
            zorder=3,
        )

        for bar, value in zip(bars, values):
            ax.text(
                value + 0.010,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=7.35,
                color=INK,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=8.7, pad=7)
    ax.set_xlim(0.0, 0.80)
    ax.xaxis.set_major_locator(
        MultipleLocator(0.2)
    )
    ax.grid(
        axis="x",
        color=GRID,
        linewidth=0.55,
        zorder=0,
    )
    ax.set_axisbelow(True)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=3,
    frameon=False,
    fontsize=7.8,
    handlelength=1.6,
    columnspacing=1.35,
)

fig.supxlabel(
    r"RMS atomic force-vector error / eV Å$^{-1}$",
    fontsize=8.5,
    y=0.035,
)

fig.subplots_adjust(
    left=0.105,
    right=0.99,
    bottom=0.18,
    top=0.80,
)

fig.savefig(
    OUTDIR / "Figure10_B3LYP_force_error_TWO_PANEL_PERFECT_FINAL.png",
    dpi=600,
)
fig.savefig(
    OUTDIR / "Figure10_B3LYP_force_error_TWO_PANEL_PERFECT_FINAL.svg"
)
fig.savefig(
    OUTDIR / "Figure10_B3LYP_force_error_TWO_PANEL_PERFECT_FINAL.pdf"
)
plt.close(fig)

print("QC PASS")
print(
    f"244 sequential geometries; reference step 227; "
    f"late B3LYP excursion step {peak_step} = {peak_energy:.3f} eV."
)
print("Created:")
print(
    OUTDIR / "Figure9_B3LYP_energy_benchmark_TWO_PANEL_PERFECT_FINAL.png"
)
print(
    OUTDIR / "Figure10_B3LYP_force_error_TWO_PANEL_PERFECT_FINAL.png"
)

