from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Repository-adapted copy of the final Section 3.1 plotting logic.
# Only paths/output numbering are changed; locked data/QC are unchanged.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data" / "inter_model"
OUTDIR = Path(__file__).resolve().parent

STRUCTURE_CSV = DATA / "three_model_structure_comparison.csv"
METRICS_CSV = DATA / "three_model_pairwise_metrics.csv"
GROUP_CSV = DATA / "three_model_group_metrics.csv"
ATOMIC_CSV = DATA / "three_model_atomic_force_comparison.csv"

for p in (STRUCTURE_CSV, METRICS_CSV, GROUP_CSV, ATOMIC_CSV):
    if not p.is_file():
        raise FileNotFoundError(p)

structures = pd.read_csv(STRUCTURE_CSV)
metrics = pd.read_csv(METRICS_CSV)
groups = pd.read_csv(GROUP_CSV)
atomic = pd.read_csv(ATOMIC_CSV)

if len(structures) != 100 or structures["filename"].nunique() != 100:
    raise ValueError("Expected exactly 100 unique MD-derived N16 structures.")
if not ((structures["mace_sha256"] == structures["aimnet_sha256"]).all()
        and (structures["mace_sha256"] == structures["uma_sha256"]).all()):
    raise ValueError("Cross-model structure SHA256 hashes do not match.")

def selected_rank(filename):
    m = re.search(r"cluster_N16_(\d+)_", str(filename))
    if not m:
        raise ValueError(f"Cannot recover selected rank from {filename}")
    return int(m.group(1))

structures["selected_rank"] = structures["filename"].map(selected_rank)
structures = structures.sort_values("selected_rank").reset_index(drop=True)
if structures["selected_rank"].tolist() != list(range(1, 101)):
    raise ValueError("Selected N16 ranks are not exactly 1-100.")
for col in ("mace_relative_energy_eV", "aimnet_relative_energy_eV", "uma_relative_energy_eV"):
    min_rank = int(structures.loc[structures[col].idxmin(), "selected_rank"])
    if min_rank != 13:
        raise ValueError(f"Expected independently referenced minimum at structure 13 for {col}; found {min_rank}.")

BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
GRID, INK, AXIS, IDENTITY = "#E8E8E8", "#1A1A1A", "#4A4A4A", "#A9A9A9"
COLORS = [BLUE, ORANGE, GREEN]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 9.2,
    "axes.linewidth": 0.8,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# Figure 7: pairwise relative-energy agreement
pairs_energy = [
    ("mace", "aimnet", "MACE-OFF23", "AIMNet2 ensemble mean"),
    ("mace", "uma", "MACE-OFF23", "UMA/OMol"),
    ("aimnet", "uma", "AIMNet2 ensemble mean", "UMA/OMol"),
]
energy = {
    "mace": structures["mace_relative_energy_eV"].to_numpy(float),
    "aimnet": structures["aimnet_relative_energy_eV"].to_numpy(float),
    "uma": structures["uma_relative_energy_eV"].to_numpy(float),
}
em = metrics[metrics["metric_type"].astype(str).eq("relative_energy_eV")]
stats = {}
for mx, my, _, _ in pairs_energy:
    row = em[(em["model_x"] == mx) & (em["model_y"] == my)]
    if len(row) != 1:
        raise ValueError(f"Expected one energy metric row for {mx}/{my}")
    row = row.iloc[0]
    stats[(mx, my)] = (float(row["pearson_r"]), float(row["spearman_rho"]), float(row["rmse"]))
all_values = np.concatenate(list(energy.values()))
upper = float(np.ceil((all_values.max() + 0.05) * 10) / 10)
lim = (-0.035, upper)
fig, axes = plt.subplots(1, 3, figsize=(7.7, 3.08))
for ax, (mx, my, lx, ly), colour, letter in zip(axes, pairs_energy, COLORS, ("a", "b", "c")):
    r, rho, rmse = stats[(mx, my)]
    ax.grid(True, color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.plot(lim, lim, color=IDENTITY, linewidth=1.05, zorder=1)
    ax.scatter(energy[mx], energy[my], s=24, facecolor=colour, edgecolor="white", linewidth=0.45, alpha=0.90, zorder=3)
    ax.text(0.055, 0.945, f"$r$ = {r:.3f}\n$\\rho$ = {rho:.3f}\nPairwise RMSE = {rmse:.3f} eV",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.25, linespacing=1.42)
    ax.set_title(f"({letter}) {lx} vs {ly}", fontsize=8.15, pad=7)
    ax.set_xlabel(f"{lx} relative energy / eV", fontsize=7.65)
    ax.set_ylabel(f"{ly} relative energy / eV", fontsize=7.65)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.5)); ax.yaxis.set_major_locator(MultipleLocator(0.5))
fig.subplots_adjust(left=0.072, right=0.997, bottom=0.225, top=0.84, wspace=0.40)
for ext in ("png", "svg", "pdf"):
    kwargs = {"dpi": 600} if ext == "png" else {}
    fig.savefig(OUTDIR / f"Figure7_pairwise_relative_energy_agreement_FINAL.{ext}", **kwargs)
plt.close(fig)

# Figure 8: chemically resolved pairwise atomic-force disagreement
pairs_force = [
    ("mace", "aimnet", "MACE-OFF23-AIMNet2"),
    ("mace", "uma", "MACE-OFF23-UMA/OMol"),
    ("aimnet", "uma", "AIMNet2-UMA/OMol"),
]
elements = ["C", "H", "N", "O"]
element_values = {}
for mx, my, label in pairs_force:
    sub = groups[(groups["model_x"] == mx) & (groups["model_y"] == my) & (groups["grouping"] == "element")].set_index("group")
    element_values[label] = {e: float(sub.loc[e, "mean_vector_difference_eV_per_A"]) for e in elements}

def mean_pairwise_vector_difference(mx, my, atom_index):
    sub = atomic[atomic["atom_index_1based"] == atom_index]
    A = sub[[f"{mx}_fx", f"{mx}_fy", f"{mx}_fz"]].to_numpy(float)
    B = sub[[f"{my}_fx", f"{my}_fy", f"{my}_fz"]].to_numpy(float)
    return float(np.linalg.norm(A - B, axis=1).mean())

site_values = {}
for mx, my, label in pairs_force:
    site_values[label] = {
        "Protonated ring N": mean_pairwise_vector_difference(mx, my, 1),
        "Exocyclic amino N": mean_pairwise_vector_difference(mx, my, 2),
    }
categories = ["C", "H", "N", "O", "Protonated ring N", "Exocyclic amino N"]
y = np.array([0.0, 1.0, 2.0, 3.0, 4.45, 5.45])
force_values = {}
for _, _, label in pairs_force:
    force_values[label] = [
        element_values[label]["C"], element_values[label]["H"], element_values[label]["N"], element_values[label]["O"],
        site_values[label]["Protonated ring N"], site_values[label]["Exocyclic amino N"],
    ]
fig, ax = plt.subplots(figsize=(7.7, 3.35))
bar_h = 0.21
for offset, (_, _, label), colour in zip(np.array([-bar_h, 0.0, bar_h]), pairs_force, COLORS):
    vals = np.asarray(force_values[label], dtype=float)
    bars = ax.barh(y + offset, vals, height=bar_h, color=colour, edgecolor="white", linewidth=0.45, label=label, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.010, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", ha="left", fontsize=7.45)
ax.set_yticks(y); ax.set_yticklabels(categories); ax.invert_yaxis()
ax.set_xlim(0.0, 0.90); ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.set_xlabel(r"Mean pairwise atomic-force disagreement, $\langle |\mathbf{F}_A-\mathbf{F}_B| \rangle$ / eV Å$^{-1}$", fontsize=8.3)
ax.grid(axis="x", color=GRID, linewidth=0.55, zorder=0); ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False, fontsize=7.7, handlelength=1.6, columnspacing=1.35)
fig.subplots_adjust(left=0.22, right=0.995, bottom=0.19, top=0.82)
for ext in ("png", "svg", "pdf"):
    kwargs = {"dpi": 600} if ext == "png" else {}
    fig.savefig(OUTDIR / f"Figure8_chemically_resolved_force_disagreement_FINAL.{ext}", **kwargs)
plt.close(fig)
print("QC PASS: Figures 7 and 8 regenerated from public locked derived tables.")
