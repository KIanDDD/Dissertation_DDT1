from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSV = REPO_ROOT / "data" / "b3lyp_benchmark" / "disagreement_recovery.csv"
OUTDIR = Path(__file__).resolve().parent

INDICATORS = [
    "AIMNet2 energy ensemble std",
    "AIMNet2 force disagreement RMS",
    "Cross-model force disagreement RMS",
]
LABELS = {
    "AIMNet2 energy ensemble std": "AIMNet2 energy disagreement",
    "AIMNet2 force disagreement RMS": "AIMNet2 force RMS disagreement",
    "Cross-model force disagreement RMS": "Cross-model force RMS disagreement",
}
FRACTIONS = [0.05, 0.10, 0.20]
EXPECTED_N = [13, 25, 49]
EXPECTED_RECOVERY = {
    "AIMNet2 energy ensemble std": [61.5, 44.0, 32.7],
    "AIMNet2 force disagreement RMS": [30.8, 48.0, 49.0],
    "Cross-model force disagreement RMS": [7.7, 12.0, 10.2],
}

if not SOURCE_CSV.is_file():
    raise FileNotFoundError(SOURCE_CSV)
df = pd.read_csv(SOURCE_CSV)
required = {"indicator", "selection_fraction", "n_selected", "recovery_fraction", "random_expected_fraction"}
missing = required.difference(df.columns)
if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
if len(df) != 9: raise ValueError(f"Expected 9 rows; found {len(df)}")
if set(df["indicator"]) != set(INDICATORS): raise ValueError("Unexpected disagreement indicators")
if not np.allclose(sorted(df["selection_fraction"].unique().astype(float)), FRACTIONS):
    raise ValueError("Unexpected prioritisation fractions")
ordered = {}
for indicator in INDICATORS:
    sub = df[df["indicator"] == indicator].set_index("selection_fraction").reindex(FRACTIONS)
    if sub.isnull().any().any(): raise ValueError(f"Missing data for {indicator}")
    ordered[indicator] = sub
for indicator in INDICATORS:
    got_n = ordered[indicator]["n_selected"].astype(int).tolist()
    if got_n != EXPECTED_N: raise ValueError(f"{indicator}: n_selected {got_n} != {EXPECTED_N}")
    got = ordered[indicator]["recovery_fraction"].to_numpy(float) * 100.0
    exp = np.asarray(EXPECTED_RECOVERY[indicator], float)
    if not np.allclose(got, exp, atol=0.06): raise ValueError(f"Locked recovery values changed for {indicator}")
# The source table stores the nominal selection budgets (5%, 10%, 20%).
# On a finite set of 244 geometries, these correspond to realised selections
# of 13, 25 and 49 geometries. The dissertation random-selection baseline
# therefore uses the realised fractions n_selected / 244:
# 5.3%, 10.2% and 20.1%, respectively.
nominal_random = ordered[INDICATORS[0]]["random_expected_fraction"].to_numpy(float)
if not np.allclose(nominal_random, FRACTIONS, atol=1e-12):
    raise ValueError("Source-table nominal random fractions have changed")
random_pct = np.asarray(EXPECTED_N, dtype=float) / 244.0 * 100.0

BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
GRID, INK = "#E8E8E8", "#1A1A1A"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 9.2, "axes.labelsize": 9.2, "xtick.labelsize": 8.6,
    "ytick.labelsize": 8.6, "legend.fontsize": 7.8, "axes.linewidth": 0.85,
    "axes.edgecolor": "#454545", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#454545", "ytick.color": "#454545",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "savefig.bbox": "tight",
    "svg.fonttype": "none", "pdf.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(7.55, 3.95))
x = np.arange(len(FRACTIONS)); width = 0.22
colours = [BLUE, ORANGE, GREEN]; offsets = [-width, 0.0, width]
bar_containers = []
for offset, indicator, colour in zip(offsets, INDICATORS, colours):
    values = ordered[indicator]["recovery_fraction"].to_numpy(float) * 100.0
    bars = ax.bar(x + offset, values, width=width, color=colour, edgecolor="white", linewidth=0.45,
                  label=LABELS[indicator], zorder=3)
    bar_containers.append(bars)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+1.15, f"{value:.1f}%",
                ha="center", va="bottom", fontsize=7.8, color=INK)
# Final dissertation choice: black dashed baseline, no markers or extra decoration.
random_line, = ax.plot(x, random_pct, color="black", linestyle="--", linewidth=1.45,
                       label="Random-selection expectation", zorder=4)
ax.set_xticks(x)
ax.set_xticklabels(["5%\n(n = 13)", "10%\n(n = 25)", "20%\n(n = 49)"])
ax.set_xlabel("Prioritisation budget")
ax.set_ylabel("Highest-error configurations recovered (%)")
ax.set_ylim(0, 70); ax.set_yticks(np.arange(0, 71, 10))
ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0); ax.set_axisbelow(True)
handles = [bar_containers[0], bar_containers[1], bar_containers[2], random_line]
labels = [LABELS[i] for i in INDICATORS] + ["Random-selection expectation"]
ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
          frameon=False, columnspacing=1.35, handlelength=1.7)
fig.subplots_adjust(left=0.105, right=0.995, bottom=0.20, top=0.79)
for ext in ("png", "svg", "pdf"):
    kwargs = {"dpi": 600} if ext == "png" else {}
    fig.savefig(OUTDIR / f"Figure11_retrospective_recovery_FINAL.{ext}", **kwargs)
plt.close(fig)
print("QC PASS: Figure 11 regenerated from disagreement_recovery.csv")
print("Random baseline (%):", [round(v, 1) for v in random_pct])
