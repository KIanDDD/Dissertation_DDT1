from pathlib import Path
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# ================================================================
# FIGURE 3 — FINAL
# Computational workflow for protonated tacrine microhydration
# ================================================================

OUTDIR = Path(__file__).resolve().parent
SVG_OUT = OUTDIR / "Figure3_Computational_Workflow_FINAL_NO_OVERFLOW.svg"
PNG_OUT = OUTDIR / "Figure3_Computational_Workflow_FINAL_NO_OVERFLOW_600dpi.png"

# Exact MD-derived N16 structure used for the example cluster.
# Embedded so this script is self-contained and produces the same figure
# without requiring a separate XYZ file.
N16_XYZ = r"""78
N16 tacrine microhydration cluster; source_frame=1209; time_ps=1211.418000; n_waters=16; atoms=78; max_selected_waterO_min_distance_A=3.756344; selection=nearest_16_water_oxygens_to_tacrine_heavy_atoms
N   0.05327765  1.39916541 -0.37171629
N   0.02789942 -2.49489091  0.77594682
C   2.26129595 -1.27622388 -2.69779089
C   1.83933608 -1.77969240 -1.37458781
C   2.86689727  0.05882956 -2.48338011
C   0.96753852 -0.79584382 -0.56491640
C   1.73574702  1.01821734 -2.15469244
C   0.91709010  0.53997637 -0.96017054
C   0.00404612 -1.20599435  0.39708492
C  -0.89739641 -0.27102826  1.01661941
C  -0.81374772  1.12722518  0.64441272
C  -1.88335737 -0.55234407  2.02159568
C  -1.69090780  2.10104872  1.15188333
C  -2.71087201  0.42464377  2.50654027
C  -2.67684682  1.70691134  2.09317133
H   2.89109198 -1.94001840 -3.29010275
H   1.36571185 -1.17040037 -3.31002501
H   1.36085574 -2.74984525 -1.50860146
H   2.76258151 -1.98436902 -0.83253077
H   3.59244315  0.03485037 -1.67029456
H   3.44918124  0.30962207 -3.37003020
H   2.11684481  2.03193022 -2.03119639
H   1.06646601  1.11918666 -3.00907400
H  -1.96387132 -1.54045938  2.45001861
H  -1.49688752  3.13202311  0.89527342
H  -0.80084642 -2.98163389  1.08636686
H   0.65426127 -3.00739263  0.17170211
H  -3.32706483  0.22977187  3.37183329
H  -3.34850057  2.38968779  2.59226796
H   0.10294501  2.34142806 -0.73195628
O  -1.52237447 -4.91580175  1.36488172
H  -1.01504453 -5.42025636  0.66621038
H  -2.49634202 -5.12939237  1.28893396
O  -0.25892289  3.94846274 -1.80329684
H   0.54726855  4.31513812 -2.26762846
H  -0.97301038  4.64786174 -1.77298573
O  -2.26752218 -0.42565988 -1.81077936
H  -2.25842222 -1.07889913 -2.56787661
H  -2.59842714  0.45939852 -2.13815287
O  -5.82944997  1.37234809  2.04766723
H  -5.94600519  0.68776728  1.32810947
H  -5.78486665  2.28311850  1.63716671
O   0.34085719  4.20149829  2.41845056
H   0.93875186  4.72212245  3.02793595
H   0.79727046  4.06674601  1.53894588
O  -1.28719362  3.21622874  4.61369940
H  -0.56446107  3.28694560  5.30120054
H  -0.87712415  3.16462351  3.70310613
O   2.83659045 -4.59192441  0.03383991
H   3.82959572 -4.56374524 -0.08081988
H   2.40901629 -4.86253999 -0.82868365
O   2.42308680  3.73272826  0.60741731
H   2.63663069  2.99174334  1.24408123
H   3.27162139  4.16552855  0.30299589
O  -3.78704580  1.80661227 -1.61363390
H  -3.53769048  1.97068812 -0.65922240
H  -4.33226999  2.57346751 -1.95225218
O  -5.05197652  4.40422179  2.24256798
H  -5.85341676  4.99131896  2.35663053
H  -4.60282262  4.27779032  3.12703201
O  -3.36830457 -2.92498372  4.32170400
H  -3.67808278 -3.23461221  3.42272278
H  -4.15741666 -2.81197236  4.92546746
O   1.66462866 -5.25763391 -2.43697813
H   0.98716609 -5.83092092 -2.89783266
H   2.57782618 -5.48177217 -2.77731970
O  -3.86160215  4.01816394  4.74670729
H  -2.89659150  3.75845076  4.78281387
H  -4.42935212  3.21996714  4.94805452
O  -5.44438203  1.67942835  4.74671849
H  -5.69949373  1.57762934  3.78518054
H  -6.24911722  1.53402068  5.32226810
O  -1.26655897 -2.93306230 -3.14338091
H  -0.48826917 -2.35066865 -2.90868357
H  -1.11478456 -3.34638189 -4.04122904
O   2.54906909  1.62342288  2.42664954
H   3.28861872  1.08024909  2.02912208
H   1.91007487  1.01892306  2.90232464
"""

# -----------------------------
# Exact tacrinium 2D structure
# -----------------------------
# Same protonated tacrine connectivity used previously for Figure 2.
TACRINIUM_SMILES = "C1CCC2=[NH+]C3=CC=CC=C3C(=C2C1)N"
mol = Chem.MolFromSmiles(TACRINIUM_SMILES)
if mol is None:
    raise RuntimeError("RDKit could not parse tacrinium.")

try:
    rdDepictor.SetPreferCoordGen(True)
except Exception:
    pass
rdDepictor.Compute2DCoords(mol)

drawer = rdMolDraw2D.MolDraw2DCairo(2200, 1200)
opts = drawer.drawOptions()
opts.clearBackground = False
opts.padding = 0.008
opts.bondLineWidth = 15.0        # deliberately thicker for dissertation-size legibility
opts.minFontSize = 54
opts.maxFontSize = 78
opts.addAtomIndices = False
opts.addBondIndices = False
drawer.DrawMolecule(mol)
drawer.FinishDrawing()

tac_img = mpimg.imread(BytesIO(drawer.GetDrawingText()), format="png")

# Tight crop around molecule
rgb = tac_img[..., :3]
mask = np.any(rgb < 0.985, axis=2)
yy, xx = np.where(mask)
if len(xx):
    pad = 14
    tac_img = tac_img[
        max(yy.min()-pad, 0):min(yy.max()+pad+1, tac_img.shape[0]),
        max(xx.min()-pad, 0):min(xx.max()+pad+1, tac_img.shape[1]),
    ]

# -----------------------------
# Load actual N16 cluster
# -----------------------------
def load_xyz_text(text):
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    n = int(lines[0])
    elements, coords = [], []
    for line in lines[2:2+n]:
        p = line.split()
        elements.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    if len(elements) != n:
        raise RuntimeError(f"Expected {n} atoms in embedded N16 structure; found {len(elements)}.")
    return np.array(elements), np.array(coords, float)

elements, xyz = load_xyz_text(N16_XYZ)

# PCA projection purely for a clean 2D depiction of the actual coordinates
xyz0 = xyz - xyz.mean(axis=0)
_, _, vh = np.linalg.svd(xyz0, full_matrices=False)
proj = xyz0 @ vh[:2].T
span = max(np.ptp(proj[:, 0]), np.ptp(proj[:, 1]), 1e-9)
proj = (proj - proj.mean(axis=0)) / (span / 2)

# simple covalent-bond display
cov = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}
bonds = []
for i in range(len(elements)):
    for j in range(i + 1, len(elements)):
        d = np.linalg.norm(xyz[i] - xyz[j])
        cutoff = 1.22 * (cov[elements[i]] + cov[elements[j]])
        if 0.45 < d <= cutoff:
            bonds.append((i, j))

# -----------------------------
# Canvas
# -----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
})

fig, ax = plt.subplots(figsize=(7.30, 7.45))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# -----------------------------
# Palette
# -----------------------------
TEXT = "#121518"
SUB = "#38414A"
FLOW = "#303941"
DIV = "#BBC5CD"

COMMON = "#3E6A87"
COMMON_BG = "#EDF4F8"

GOLD = "#9A713F"
GOLD_BG = "#F8F0E4"

BLUE = "#3475A1"
BLUE_BG = "#EAF4FA"
BLUE_BOX = ["#F7FBFD", "#DDECF5", "#EEF6FA"]

PURPLE = "#6B5788"
PURPLE_BG = "#F2EEF6"
PURPLE_BOX = ["#FAF8FC", "#E6DFF0", "#F1ECF6"]

GREEN = "#4E7F4C"
GREEN_BG = "#ECF5EB"
GREEN_BOX = ["#F7FBF6", "#E0EEDF", "#EDF5EB"]

WHITE = "#FFFFFF"

ATOM_COLOR = {
    "H": "#D7DCE0",
    "C": "#454A4F",
    "N": "#2C67B2",
    "O": "#CB4848",
}
ATOM_SIZE = {"H": 7, "C": 18, "N": 22, "O": 22}

# -----------------------------
# Helpers
# -----------------------------
def rounded(x, y, w, h, fc, ec=FLOW, lw=1.0, radius=0.010, z=1):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.003,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z
    )
    ax.add_patch(p)
    return p

def flow_arrow(x1, y1, x2, y2, color=FLOW, lw=1.75, head=11.5, z=10):
    p = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>",
        mutation_scale=head,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=z
    )
    ax.add_patch(p)
    return p

# ================================================================
# TOP ROW — common preparation and hydration stages
# large visual area + compact text, with no nested boxes or clashes
# ================================================================
CARD_Y = 0.795
CARD_W = 0.294
CARD_H = 0.166
CARD_X = [0.020, 0.353, 0.686]

def top_card(x, title, visual, main, sub):
    y, w, h = CARD_Y, CARD_W, CARD_H
    rounded(x, y, w, h, COMMON_BG, ec=FLOW, lw=1.05, radius=0.010)

    hh = 0.050
    ax.add_patch(FancyBboxPatch(
        (x, y+h-hh), w, hh,
        boxstyle="round,pad=0.003,rounding_size=0.010",
        facecolor=COMMON, edgecolor=COMMON, linewidth=0, zorder=2
    ))
    ax.add_patch(Rectangle(
        (x, y+h-hh), w, hh*0.33,
        facecolor=COMMON, edgecolor="none", zorder=2
    ))
    ax.text(
        x+w/2, y+h-hh/2, title,
        ha="center", va="center",
        fontsize=9.9, fontweight="bold", color=WHITE, zorder=5
    )

    # Upper body reserved entirely for the visual.
    vis_y0 = y + 0.057
    vis_y1 = y + h - hh - 0.006
    cx = x + w/2

    if visual == "rdkit":
        # Larger, thicker depiction; atom labels stay visible.
        ax.imshow(
            tac_img,
            extent=(x+0.008, x+w-0.008, vis_y0-0.006, vis_y1+0.007),
            aspect="auto", zorder=4
        )

    elif visual == "md":
        # Periodic explicit-water schematic, no decorative arrows inside.
        bw, bh = 0.116, 0.050
        bx = cx - bw/2
        by = (vis_y0 + vis_y1)/2 - bh/2 + 0.002

        ax.add_patch(Rectangle(
            (bx, by), bw, bh,
            facecolor="#FBFDFE", edgecolor=COMMON,
            linewidth=1.15, zorder=4
        ))

        pts = [
            (-0.044, 0.015), (-0.029, -0.015), (-0.010, 0.018),
            (0.012, -0.014), (0.032, 0.016), (0.046, -0.012),
            (-0.020, 0.002), (0.023, 0.000),
        ]
        for dx, dy in pts:
            ax.add_patch(Circle(
                (cx+dx, by+bh/2+dy), 0.0043,
                facecolor=WHITE, edgecolor=COMMON,
                linewidth=0.75, zorder=5
            ))

        ax.text(
            cx, by+bh/2, "Tac⁺",
            ha="center", va="center",
            fontsize=5.6, fontweight="bold", color=COMMON, zorder=6
        )
        ax.text(
            bx+bw-0.004, by+bh-0.004, "Cl⁻",
            ha="right", va="top",
            fontsize=5.1, fontweight="bold", color=COMMON, zorder=6
        )

    elif visual == "distance":
        # Actual analysis coordinate: water O ↔ nearest tacrinium heavy atom.
        hy = (vis_y0 + vis_y1)/2 + 0.004
        lx = x + 0.100
        rx = x + w - 0.100

        ax.add_patch(Circle(
            (lx, hy), 0.012,
            facecolor="#4B4F54", edgecolor="#4B4F54", zorder=5
        ))
        ax.add_patch(Circle(
            (rx, hy), 0.012,
            facecolor="#CB4848", edgecolor="#CB4848", zorder=5
        ))
        ax.text(
            rx, hy, "O",
            ha="center", va="center",
            fontsize=5.7, fontweight="bold", color=WHITE, zorder=6
        )

        p = FancyArrowPatch(
            (lx+0.017, hy), (rx-0.017, hy),
            arrowstyle="<->",
            mutation_scale=9.5,
            linewidth=1.25,
            color=COMMON,
            zorder=6
        )
        ax.add_patch(p)

        ax.text(
            (lx+rx)/2, hy+0.014, "dᵢ(t)",
            ha="center", va="bottom",
            fontsize=6.6, fontweight="bold", color=COMMON, zorder=6
        )
        ax.text(
            lx, hy-0.018, "nearest\nheavy atom",
            ha="center", va="top",
            fontsize=5.15, fontweight="bold", color=TEXT,
            linespacing=0.92, zorder=6
        )
        ax.text(
            rx, hy-0.018, "water O",
            ha="center", va="top",
            fontsize=5.15, fontweight="bold", color=TEXT, zorder=6
        )

    # Bottom text zone is completely separate from the visual zone.
    ax.plot(
        [x+0.024, x+w-0.024],
        [y+0.050, y+0.050],
        color=DIV, linewidth=0.65, zorder=4
    )
    ax.text(
        cx, y+0.034, main,
        ha="center", va="center",
        fontsize=7.75, fontweight="bold", color=TEXT, zorder=6
    )
    ax.text(
        cx, y+0.0165, sub,
        ha="center", va="center",
        fontsize=6.30, color=SUB, linespacing=1.02, zorder=6
    )
    return x, y, w, h

ax.text(
    0.020, 0.982, "Common workflow",
    ha="left", va="center",
    fontsize=11.5, fontweight="bold", color=TEXT
)

cards = [
    top_card(
        CARD_X[0],
        "Tacrine preparation",
        "rdkit",
        "LigPrep/Epik · pH 7.4 ± 0.5",
        "retained species: tacrinium (+1)"
    ),
    top_card(
        CARD_X[1],
        "Periodic explicit-water MD",
        "md",
        "OPLS4 · SPC · one Cl⁻",
        "10 ns NPT · 9,982 readable frames"
    ),
    top_card(
        CARD_X[2],
        "Hydration analysis",
        "distance",
        "minimum water-O distance",
        "nearest tacrinium heavy atom · MIC\nfirst minimum = 5.325 Å"
    ),
]

# Top arrows remain fully inside the blank gutters.
for i in range(2):
    x, y, w, h = cards[i]
    nx, ny, nw, nh = cards[i+1]
    flow_arrow(
        x+w+0.007, y+h/2,
        nx-0.007, ny+nh/2,
        lw=1.95, head=12.5
    )

# ================================================================
# CONTROLLED N16 CONSTRUCTION
# ================================================================
HUB_X, HUB_Y, HUB_W, HUB_H = 0.075, 0.602, 0.850, 0.137
rounded(HUB_X, HUB_Y, HUB_W, HUB_H, GOLD_BG, ec=GOLD, lw=1.20, radius=0.011)

hh = 0.041
ax.add_patch(FancyBboxPatch(
    (HUB_X, HUB_Y+HUB_H-hh), HUB_W, hh,
    boxstyle="round,pad=0.003,rounding_size=0.011",
    facecolor=GOLD, edgecolor=GOLD, linewidth=0, zorder=2
))
ax.add_patch(Rectangle(
    (HUB_X, HUB_Y+HUB_H-hh), HUB_W, hh*0.35,
    facecolor=GOLD, edgecolor="none", zorder=2
))
ax.text(
    HUB_X+0.018, HUB_Y+HUB_H-hh/2,
    "Controlled N16 construction",
    ha="left", va="center",
    fontsize=9.7, fontweight="bold", color=WHITE, zorder=5
)

# Actual MD-derived N16 visual at left; no text overlaps atoms.
VIS_LEFT = HUB_X + 0.026
VIS_RIGHT = HUB_X + 0.325
VIS_BOTTOM = HUB_Y + 0.018
VIS_TOP = HUB_Y + HUB_H - hh - 0.010
cx0 = (VIS_LEFT + VIS_RIGHT)/2
cy0 = (VIS_BOTTOM + VIS_TOP)/2 - 0.002
scale_x = 0.118
scale_y = 0.029

for i, j in bonds:
    xi = cx0 + proj[i, 0]*scale_x
    yi = cy0 + proj[i, 1]*scale_y
    xj = cx0 + proj[j, 0]*scale_x
    yj = cy0 + proj[j, 1]*scale_y

    if i < 30 and j < 30:
        col, lw, alpha = "#50555A", 1.00, 0.95
    else:
        col, lw, alpha = "#92999F", 0.55, 0.65

    ax.plot([xi, xj], [yi, yj], color=col, linewidth=lw, alpha=alpha, zorder=4)

for i, el in enumerate(elements):
    xi = cx0 + proj[i, 0]*scale_x
    yi = cy0 + proj[i, 1]*scale_y
    size = ATOM_SIZE[el] * (1.20 if i < 30 else 1.0)
    ax.scatter(
        [xi], [yi],
        s=size,
        c=ATOM_COLOR[el],
        edgecolors=WHITE if el == "H" else "#2D3236",
        linewidths=0.23,
        zorder=5
    )

ax.text(
    VIS_LEFT, VIS_TOP-0.001, "Example MD-derived N16",
    ha="left", va="top",
    fontsize=6.1, fontweight="bold", color=SUB, zorder=6
)

# Separator and right text block
SEP_X = HUB_X + 0.350
ax.plot(
    [SEP_X, SEP_X],
    [HUB_Y+0.014, HUB_Y+HUB_H-hh-0.009],
    color="#D3BB99", linewidth=0.85, zorder=4
)

TX = SEP_X + 0.025
ax.text(
    TX, HUB_Y+0.071,
    "100 approximately uniformly distributed MD frames",
    ha="left", va="center",
    fontsize=8.05, fontweight="bold", color=TEXT, zorder=6
)
ax.text(
    TX, HUB_Y+0.049,
    "16 nearest waters retained with tacrinium",
    ha="left", va="center",
    fontsize=7.75, fontweight="bold", color=TEXT, zorder=6
)
ax.text(
    TX, HUB_Y+0.030,
    "[tacrineH]⁺(H₂O)₁₆ · 78 atoms · net +1",
    ha="left", va="center",
    fontsize=6.75, color=SUB, zorder=6
)
ax.text(
    TX, HUB_Y+0.011,
    "Cl⁻ omitted · non-periodic · fixed composition · not the full hydration envelope",
    ha="left", va="center",
    fontsize=6.20, color=SUB, zorder=6
)

# Hydration → N16 arrow
hyd = cards[2]
flow_arrow(
    hyd[0]+hyd[2]/2, hyd[1]-0.006,
    hyd[0]+hyd[2]/2, HUB_Y+HUB_H+0.006,
    lw=1.90, head=11.8
)

# ================================================================
# BRANCHING TO THREE INDEPENDENT ANALYSES
# ================================================================
LABEL_Y = 0.555
BUS_Y = 0.518

# connector stops before label, resumes after it
ax.plot(
    [0.5, 0.5],
    [HUB_Y-0.004, LABEL_Y+0.014],
    color=FLOW, linewidth=1.50, zorder=3
)

ax.text(
    0.5, LABEL_Y,
    "Independent downstream analyses",
    ha="center", va="center",
    fontsize=8.9, fontweight="bold", color=TEXT, zorder=8
)

flow_arrow(
    0.5, LABEL_Y-0.015,
    0.5, BUS_Y+0.004,
    lw=1.50, head=9.6
)

PANEL_X = [0.020, 0.353, 0.686]
PANEL_W = 0.294
PANEL_Y = 0.045
PANEL_H = 0.425
CENTRES = [x + PANEL_W/2 for x in PANEL_X]

ax.plot(
    [CENTRES[0], CENTRES[2]],
    [BUS_Y, BUS_Y],
    color=FLOW, linewidth=1.60, zorder=3
)

# ================================================================
# A / B / C PANELS
# ================================================================
def analysis_panel(x, color, panel_bg, box_bg, letter, title, rows):
    y, w, h = PANEL_Y, PANEL_W, PANEL_H
    rounded(x, y, w, h, panel_bg, ec=color, lw=1.16, radius=0.011)

    hh = 0.058
    ax.add_patch(FancyBboxPatch(
        (x, y+h-hh), w, hh,
        boxstyle="round,pad=0.003,rounding_size=0.011",
        facecolor=color, edgecolor=color, linewidth=0, zorder=2
    ))
    ax.add_patch(Rectangle(
        (x, y+h-hh), w, hh*0.35,
        facecolor=color, edgecolor="none", zorder=2
    ))

    ax.add_patch(FancyBboxPatch(
        (x+0.014, y+h-0.046), 0.042, 0.033,
        boxstyle="round,pad=0.002,rounding_size=0.004",
        facecolor=WHITE, edgecolor="none", zorder=4
    ))
    ax.text(
        x+0.035, y+h-0.0295, letter,
        ha="center", va="center",
        fontsize=8.5, fontweight="bold", color=color, zorder=5
    )
    ax.text(
        x+0.066, y+h-hh/2, title,
        ha="left", va="center",
        fontsize=8.85, fontweight="bold", color=WHITE, zorder=5
    )

    inner_x = x + 0.015
    inner_w = w - 0.030
    bh = 0.098
    ys = [y+0.234, y+0.128, y+0.022]

    for k, ((main, sub), fc) in enumerate(zip(rows, box_bg)):
        rounded(inner_x, ys[k], inner_w, bh, fc, ec=color, lw=0.88, radius=0.006, z=3)
        ax.text(
            inner_x+inner_w/2, ys[k]+0.062, main,
            ha="center", va="center",
            fontsize=7.70, fontweight="bold", color=TEXT, zorder=5
        )
        ax.text(
            inner_x+inner_w/2, ys[k]+0.028, sub,
            ha="center", va="center",
            fontsize=6.60, color=SUB,
            linespacing=1.00, zorder=5
        )

        if k < 2:
            flow_arrow(
                x+w/2, ys[k]-0.005,
                x+w/2, ys[k+1]+bh+0.005,
                color=color, lw=1.38, head=9.1, z=9
            )

# bus → panels
for cx in CENTRES:
    flow_arrow(
        cx, BUS_Y,
        cx, PANEL_Y+PANEL_H+0.009,
        lw=1.40, head=9.1
    )

analysis_panel(
    PANEL_X[0], BLUE, BLUE_BG, BLUE_BOX,
    "A", "MLFF screening",
    [
        ("100 MD-derived N16 structures", "identical composition\nand atom order"),
        ("Fixed-geometry MLFF inference", "MACE-OFF23-medium · AIMNet2 ×4\nUMA/OMol"),
        ("Screening outputs", "relative energies · forces\nagreement/disagreement"),
    ]
)

analysis_panel(
    PANEL_X[1], PURPLE, PURPLE_BG, PURPLE_BOX,
    "B", "B3LYP path benchmark",
    [
        ("244 Gaussian-path N16", "supervisor-supplied\nsequential geometries"),
        ("Path-local comparison", "B3LYP/6-31G(d)\n+ same pretrained MLFFs"),
        ("Benchmark outputs", "energy/force deviations\ngroup- and element-resolved"),
    ]
)

analysis_panel(
    PANEL_X[2], GREEN, GREEN_BG, GREEN_BOX,
    "C", "AIMNet2 dynamics",
    [
        ("One MD-derived N16", "finite non-periodic\nstarting structure"),
        ("Short diagnostic trajectory", "0.25 ps NVT →\n0.50 ps NVE"),
        ("Diagnostic outputs", "energy behaviour · hydration\nensemble disagreement"),
    ]
)

fig.subplots_adjust(left=0.008, right=0.992, top=0.992, bottom=0.008)

fig.savefig(SVG_OUT, format="svg", bbox_inches="tight", pad_inches=0.02, facecolor="white")
fig.savefig(PNG_OUT, format="png", dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")


print("Created:")
print(SVG_OUT)
print(PNG_OUT)