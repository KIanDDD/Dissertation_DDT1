from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Arc

# ================================================================
# Figure 1 — Evolution of machine-learned force fields
# Reproduces the approved four-card dissertation figure.
# Outputs SVG (preferred for Word), 600-dpi PNG, and PDF.
# ================================================================

OUTDIR = Path(__file__).resolve().parent
SVG_OUT = OUTDIR / "Figure1_MLFF_evolution_EXACT.svg"
PNG_OUT = OUTDIR / "Figure1_MLFF_evolution_EXACT_600dpi.png"
PDF_OUT = OUTDIR / "Figure1_MLFF_evolution_EXACT.pdf"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Canvas designed for full-width insertion in Word.
fig, ax = plt.subplots(figsize=(13.2, 5.0))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Muted scientific palette.
FILLS = ["#EEF4F8", "#EEF7F4", "#F3F1F8", "#F8F3EC"]
BANDS = ["#365E7D", "#3C7163", "#665C87", "#8B6845"]
EDGE = "#303030"
TEXT = "#1F1F1F"
ARROW = "#4B4B4B"

# Four equal cards.
CARD_X = [0.020, 0.270, 0.520, 0.770]
CARD_Y = 0.200
CARD_W = 0.205
CARD_H = 0.690

YEAR_Y = CARD_Y + CARD_H - 0.057
TITLE_Y = 0.700
ICON_Y = 0.505
MODEL_Y = 0.385
DESCRIPTION_Y = 0.255


def add_card(x, year, title, model, description, fill, band):
    """Draw one milestone card with fixed typography and spacing."""
    ax.add_patch(FancyBboxPatch(
        (x, CARD_Y), CARD_W, CARD_H,
        boxstyle="round,pad=0.007,rounding_size=0.016",
        facecolor=fill, edgecolor=EDGE, linewidth=1.15, zorder=1
    ))

    ax.add_patch(FancyBboxPatch(
        (x, CARD_Y + CARD_H - 0.115), CARD_W, 0.115,
        boxstyle="round,pad=0.007,rounding_size=0.016",
        facecolor=band, edgecolor=band, linewidth=0, zorder=2
    ))

    ax.text(
        x + CARD_W/2, YEAR_Y, year,
        ha="center", va="center", fontsize=17.5,
        fontweight="bold", color="white", zorder=5
    )

    ax.text(
        x + CARD_W/2, TITLE_Y, title,
        ha="center", va="top", fontsize=14.6,
        fontweight="bold", color=TEXT, linespacing=1.02, zorder=5
    )

    ax.text(
        x + CARD_W/2, MODEL_Y, model,
        ha="center", va="center", fontsize=13.3,
        fontweight="bold", color=TEXT, linespacing=1.0, zorder=5
    )

    ax.text(
        x + CARD_W/2, DESCRIPTION_Y, description,
        ha="center", va="center", fontsize=11.8,
        color=TEXT, linespacing=1.08, zorder=5
    )


# -------------------- Cards --------------------
add_card(
    CARD_X[0], "2007",
    "High-dimensional\nneural potentials",
    "Behler–Parrinello",
    "Local environments\n+ atomic-energy sum",
    FILLS[0], BANDS[0]
)

add_card(
    CARD_X[1], "2017",
    "Transferable\nmolecular potentials",
    "ANI-1",
    "Broader organic\nchemical space",
    FILLS[1], BANDS[1]
)

add_card(
    CARD_X[2], "2022",
    "Equivariant\narchitectures",
    "NequIP · MACE",
    "Geometry-aware\nmessage passing",
    FILLS[2], BANDS[2]
)

add_card(
    CARD_X[3], "2025",
    "Pretrained\nreusable models",
    "MACE-OFF · AIMNet2\nUMA",
    "Reuse across broader\nmolecular chemistry",
    FILLS[3], BANDS[3]
)

# -------------------- Icon 1: local environment --------------------
cx, cy = CARD_X[0] + CARD_W/2, ICON_Y
ax.add_patch(Circle((cx, cy), 0.015, facecolor=BANDS[0], edgecolor="none", zorder=5))
for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
    px = cx + 0.050*np.cos(angle)
    py = cy + 0.050*np.sin(angle)
    ax.plot([cx, px], [cy, py], color=BANDS[0], linewidth=1.15, zorder=3)
    ax.add_patch(Circle((px, py), 0.0085, facecolor="white",
                        edgecolor=BANDS[0], linewidth=1.15, zorder=4))
ax.add_patch(Circle((cx, cy), 0.068, fill=False, edgecolor=BANDS[0],
                    linewidth=1.05, linestyle="--", zorder=3))

# -------------------- Icon 2: transferable molecular chemistry --------------------
cx, cy = CARD_X[1] + CARD_W/2, ICON_Y
molecules = [(-0.052, 0.000, 0.85), (0.000, 0.025, 1.00), (0.052, -0.004, 0.78)]
for dx, dy, scale in molecules:
    pts = np.array([
        [-0.020,  0.000],
        [ 0.000,  0.020],
        [ 0.023,  0.004],
        [ 0.010, -0.023],
    ]) * scale
    pts[:, 0] += cx + dx
    pts[:, 1] += cy + dy
    for i, j in [(0, 1), (1, 2), (1, 3)]:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                color=BANDS[1], linewidth=1.20, zorder=3)
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.0067, facecolor="white",
                            edgecolor=BANDS[1], linewidth=1.10, zorder=4))

# -------------------- Icon 3: equivariant message passing --------------------
cx, cy = CARD_X[2] + CARD_W/2, ICON_Y
nodes = np.array([
    [cx - 0.052, cy - 0.020],
    [cx - 0.020, cy + 0.032],
    [cx + 0.043, cy + 0.020],
    [cx + 0.052, cy - 0.036],
    [cx + 0.000, cy - 0.050],
])
for i, j in [(0,1), (1,2), (2,3), (3,4), (4,0), (1,4)]:
    ax.add_patch(FancyArrowPatch(
        (nodes[i,0], nodes[i,1]), (nodes[j,0], nodes[j,1]),
        arrowstyle="-|>", mutation_scale=8.5, linewidth=1.05,
        color=BANDS[2], shrinkA=7, shrinkB=7, zorder=3
    ))
for px, py in nodes:
    ax.add_patch(Circle((px, py), 0.0080, facecolor="white",
                        edgecolor=BANDS[2], linewidth=1.10, zorder=4))
ax.add_patch(Arc((cx, cy), 0.145, 0.120, theta1=205, theta2=312,
                 linewidth=1.05, color=BANDS[2], zorder=2))

# -------------------- Icon 4: pretrained reuse --------------------
cx, cy = CARD_X[3] + CARD_W/2, ICON_Y

# Central MLFF box
ax.add_patch(FancyBboxPatch(
    (cx - 0.026, cy - 0.019), 0.052, 0.038,
    boxstyle="round,pad=0.003,rounding_size=0.005",
    facecolor="white", edgecolor=BANDS[3], linewidth=1.10, zorder=4
))
ax.text(cx, cy, "MLFF", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=BANDS[3], zorder=5)

# Training-data stack
for k in range(3):
    ax.add_patch(FancyBboxPatch(
        (cx - 0.080, cy + 0.035 + k*0.010), 0.042, 0.016,
        boxstyle="round,pad=0.0015,rounding_size=0.002",
        facecolor="white", edgecolor=BANDS[3], linewidth=0.90, zorder=3
    ))
ax.add_patch(FancyArrowPatch(
    (cx - 0.047, cy + 0.035), (cx - 0.025, cy + 0.014),
    arrowstyle="-|>", mutation_scale=8.5, linewidth=1.00,
    color=BANDS[3], zorder=3
))

# Three reuse targets
targets = [
    (cx + 0.072, cy + 0.045),
    (cx + 0.078, cy + 0.000),
    (cx + 0.067, cy - 0.050),
]
for ox, oy in targets:
    ax.add_patch(FancyArrowPatch(
        (cx + 0.028, cy), (ox - 0.016, oy),
        arrowstyle="-|>", mutation_scale=8.5, linewidth=1.00,
        color=BANDS[3], zorder=3
    ))
    ax.add_patch(Circle((ox, oy), 0.0090, facecolor="white",
                        edgecolor=BANDS[3], linewidth=1.00, zorder=4))
    ax.add_patch(Circle((ox + 0.018, oy + 0.008), 0.0055,
                        facecolor="white", edgecolor=BANDS[3],
                        linewidth=0.90, zorder=4))
    ax.plot([ox + 0.008, ox + 0.014], [oy + 0.003, oy + 0.006],
            color=BANDS[3], linewidth=0.90, zorder=3)

# -------------------- Strong inter-card arrows --------------------
for i in range(3):
    start_x = CARD_X[i] + CARD_W + 0.006
    end_x = CARD_X[i + 1] - 0.006
    ax.add_patch(FancyArrowPatch(
        (start_x, 0.520), (end_x, 0.520),
        arrowstyle="-|>", mutation_scale=20, linewidth=2.05,
        color=ARROW, shrinkA=0, shrinkB=0, zorder=6
    ))

# ================================================================
# BOTTOM SYNTHESIS ARROW — FINAL POLISHED VERSION
# ================================================================

bottom_y = 0.105
arrow_start_x = 0.382
arrow_end_x = 0.900

ax.text(
    0.020,
    bottom_y,
    "Towards broader chemical scope and reuse",
    ha="left",
    va="center",
    fontsize=13.2,
    fontweight="bold",
    color=TEXT,
    zorder=5,
)

ax.add_patch(
    FancyArrowPatch(
        (arrow_start_x, bottom_y),
        (arrow_end_x, bottom_y),
        arrowstyle="-|>",
        mutation_scale=17,
        linewidth=2.15,
        color=ARROW,
        shrinkA=0,
        shrinkB=0,
        zorder=3,
    )
)
# -------------------- Export --------------------
fig.tight_layout(pad=0.35)
fig.savefig(SVG_OUT, format="svg", bbox_inches="tight", facecolor="white")
fig.savefig(PNG_OUT, format="png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(PDF_OUT, format="pdf", bbox_inches="tight", facecolor="white")

print("Figure created successfully:")
print(f"  SVG: {SVG_OUT}")
print(f"  PNG: {PNG_OUT}")
print(f"  PDF: {PDF_OUT}")

plt.show()
