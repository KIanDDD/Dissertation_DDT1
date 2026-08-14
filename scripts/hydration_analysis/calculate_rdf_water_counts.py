from pathlib import Path
from collections import Counter
import csv
import math
import numpy as np

from schrodinger.application.desmond.packages import traj

BASE = Path("../00_raw_production_archive/CHE701P2_TAC_prod_NPT_10ns_1ps_001")
TRJ = BASE / "CHE701P2_TAC_prod_NPT_10ns_1ps_001_trj"

RDF_OUT = Path(".")
WATER_OUT = Path("../04_water_count_outputs")
WATER_OUT.mkdir(parents=True, exist_ok=True)

# Atom mapping from inspection:
# Schrodinger atom index 1-30 = tacrine
# atom index 31 = chloride
# atom index 32 onward = SPC water atoms in repeating O,H,H order
#
# Python coordinate array is 0-based:
# atom 1 -> pos[0]
# atom 31 -> pos[30]
# atom 32 -> pos[31]

tacrine_heavy = np.arange(0, 15)      # atoms 1-15: N,N,C...C
chloride = 30                         # atom 31
water_oxygen = np.arange(31, 1846, 3) # atoms 32,35,38,...

assert len(tacrine_heavy) == 15
assert len(water_oxygen) == 605
assert water_oxygen[0] == 31
assert water_oxygen[-1] == 1843

r_max = 10.0
bin_width = 0.05
edges = np.arange(0.0, r_max + bin_width, bin_width)
centers = 0.5 * (edges[:-1] + edges[1:])
shell_volumes = (4.0 / 3.0) * math.pi * (edges[1:]**3 - edges[:-1]**3)

def get_box_lengths(frame):
    box = np.array(frame.box, dtype=float)
    if box.shape == (3,):
        lengths = box
    else:
        box = box.reshape(3, 3)
        lengths = np.array([
            np.linalg.norm(box[0]),
            np.linalg.norm(box[1]),
            np.linalg.norm(box[2])
        ])
    if np.any(lengths <= 0):
        raise ValueError(f"Invalid box lengths: {lengths}")
    return lengths

def minimum_image_differences(a, b, box_lengths):
    # returns differences between each row of a and each row of b
    # shape = (len(a), len(b), 3)
    diff = a[:, None, :] - b[None, :, :]
    diff -= box_lengths * np.round(diff / box_lengths)
    return diff

def smooth(y, window=7):
    if window < 3:
        return y.copy()
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")

print("Reading trajectory:", TRJ)
tr = traj.read_traj(str(TRJ))
print("Frames:", len(tr))

pair_hist_total = np.zeros(len(centers), dtype=float)
rdf_accum = np.zeros(len(centers), dtype=float)
min_dist_hist_total = np.zeros(len(centers), dtype=float)

times = []
min_dists_by_frame = []

n_tacrine_heavy = len(tacrine_heavy)
n_water_o = len(water_oxygen)

for frame_i, fr in enumerate(tr):
    pos = fr.pos()
    box_lengths = get_box_lengths(fr)
    volume = float(np.prod(box_lengths))
    water_number_density = n_water_o / volume

    tac_pos = pos[tacrine_heavy]
    wat_pos = pos[water_oxygen]

    diff = minimum_image_differences(wat_pos, tac_pos, box_lengths)
    distances = np.sqrt(np.sum(diff * diff, axis=2))  # water O x tacrine heavy atoms

    pair_distances = distances.ravel()
    pair_hist, _ = np.histogram(pair_distances, bins=edges)
    pair_hist_total += pair_hist

    # Per-frame RDF normalisation, then average over frames.
    rdf_frame = pair_hist / (n_tacrine_heavy * water_number_density * shell_volumes)
    rdf_accum += rdf_frame

    min_dists = np.min(distances, axis=1)
    min_hist, _ = np.histogram(min_dists, bins=edges)
    min_dist_hist_total += min_hist

    min_dists_by_frame.append(min_dists.astype(np.float32))
    times.append(float(fr.time))

    if (frame_i + 1) % 1000 == 0:
        print(f"Processed {frame_i + 1} / {len(tr)} frames")

rdf = rdf_accum / len(tr)
min_dist_probability = min_dist_hist_total / np.sum(min_dist_hist_total)

# Smooth RDF only for locating peak/minimum. Raw RDF is still saved.
rdf_smooth = smooth(rdf, window=9)

# Locate first major peak in a chemically sensible water-contact region.
peak_region = np.where((centers >= 1.8) & (centers <= 4.5))[0]
if len(peak_region) == 0:
    raise RuntimeError("No RDF peak search region found.")
peak_idx = peak_region[np.argmax(rdf_smooth[peak_region])]
peak_r = centers[peak_idx]

# Locate first local minimum after the first peak.
candidate_minima = []
for i in range(peak_idx + 5, len(centers) - 1):
    if centers[i] > 6.5:
        break
    if centers[i] <= peak_r + 0.25:
        continue
    if rdf_smooth[i] <= rdf_smooth[i - 1] and rdf_smooth[i] <= rdf_smooth[i + 1]:
        candidate_minima.append(i)

if candidate_minima:
    min_idx = candidate_minima[0]
else:
    fallback_region = np.where((centers >= peak_r + 0.5) & (centers <= 6.5))[0]
    min_idx = fallback_region[np.argmin(rdf_smooth[fallback_region])]

cutoff = float(centers[min_idx])

# Count first-shell waters in each frame using the RDF-derived cutoff.
water_counts = []
for frame_i, min_dists in enumerate(min_dists_by_frame):
    water_counts.append(int(np.sum(min_dists <= cutoff)))

count_hist = Counter(water_counts)
dominant_count, dominant_frequency = count_hist.most_common(1)[0]

# Save RDF CSV.
rdf_csv = RDF_OUT / "tacrineHeavy_waterO_pair_RDF.csv"
with rdf_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["r_A", "g_r", "g_r_smoothed", "raw_pair_count"])
    for r, g, gs, c in zip(centers, rdf, rdf_smooth, pair_hist_total):
        writer.writerow([f"{r:.4f}", f"{g:.8f}", f"{gs:.8f}", int(c)])

# Save minimum-distance distribution.
min_csv = RDF_OUT / "waterO_min_distance_to_tacrineHeavy_distribution.csv"
with min_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["r_A", "probability", "raw_count"])
    for r, p, c in zip(centers, min_dist_probability, min_dist_hist_total):
        writer.writerow([f"{r:.4f}", f"{p:.10f}", int(c)])

# Save water count by frame.
count_csv = WATER_OUT / "water_count_by_frame.csv"
with count_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_index", "time_ps", "first_shell_water_count"])
    for i, (t, c) in enumerate(zip(times, water_counts)):
        writer.writerow([i, f"{t:.6f}", c])

# Save water count histogram.
hist_csv = WATER_OUT / "water_count_histogram.csv"
with hist_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["first_shell_water_count", "n_frames"])
    for count in sorted(count_hist):
        writer.writerow([count, count_hist[count]])

# Save decision text.
decision_txt = RDF_OUT / "rdf_cutoff_decision.txt"
with decision_txt.open("w") as f:
    f.write("Tacrine-water RDF and hydration-shell cutoff decision\n")
    f.write("====================================================\n\n")
    f.write("Trajectory: CHE701P2_TAC_prod_NPT_10ns_1ps_001\n")
    f.write(f"Frames analysed: {len(tr)}\n")
    f.write("System: tacrine + 1 Cl- + 605 SPC waters\n")
    f.write("Tacrine heavy atoms used: atoms 1-15\n")
    f.write("Water oxygen atoms used: atoms 32,35,38,...,1844\n\n")
    f.write(f"First RDF peak position: {peak_r:.3f} A\n")
    f.write(f"First RDF minimum / proposed hydration cutoff: {cutoff:.3f} A\n")
    f.write(f"Dominant first-shell water count: {dominant_count}\n")
    f.write(f"Frames with dominant water count: {dominant_frequency} / {len(tr)}\n\n")
    f.write("Important note:\n")
    f.write("The cutoff was estimated from the first minimum after the first tacrine-heavy-atom/water-oxygen RDF peak.\n")
    f.write("This cutoff must be visually checked from the RDF plot before final cluster extraction.\n")

# Try to save plots.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(centers, rdf, label="raw RDF")
    plt.plot(centers, rdf_smooth, label="smoothed RDF")
    plt.axvline(cutoff, linestyle="--", label=f"cutoff {cutoff:.2f} A")
    plt.xlabel("r / A")
    plt.ylabel("g(r)")
    plt.title("Tacrine heavy atoms - water oxygen RDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RDF_OUT / "tacrineHeavy_waterO_pair_RDF.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(centers, min_dist_probability)
    plt.axvline(cutoff, linestyle="--", label=f"cutoff {cutoff:.2f} A")
    plt.xlabel("minimum distance to tacrine heavy atom / A")
    plt.ylabel("probability")
    plt.title("Water oxygen minimum-distance distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RDF_OUT / "waterO_min_distance_distribution.png", dpi=300)
    plt.close()

    plt.figure()
    xs = sorted(count_hist)
    ys = [count_hist[x] for x in xs]
    plt.bar(xs, ys)
    plt.axvline(dominant_count, linestyle="--", label=f"dominant count {dominant_count}")
    plt.xlabel("first-shell water count")
    plt.ylabel("number of frames")
    plt.title("First-shell water-count histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(WATER_OUT / "water_count_histogram.png", dpi=300)
    plt.close()

    print("Plots saved successfully.")
except Exception as e:
    print("Plotting failed, but CSV files were saved.")
    print("Plotting error:", repr(e))

print()
print("RDF/WATER COUNT ANALYSIS COMPLETE")
print("Frames analysed:", len(tr))
print("First RDF peak A:", f"{peak_r:.3f}")
print("Proposed cutoff A:", f"{cutoff:.3f}")
print("Dominant water count:", dominant_count)
print("Dominant count frames:", dominant_frequency)
print("Saved:", rdf_csv)
print("Saved:", min_csv)
print("Saved:", count_csv)
print("Saved:", hist_csv)
print("Saved:", decision_txt)
