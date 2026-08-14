from pathlib import Path
import csv
import math
import numpy as np

from schrodinger.application.desmond.packages import traj

BASE = Path("../00_raw_production_archive/CHE701P2_TAC_prod_NPT_10ns_1ps_001")
TRJ = BASE / "CHE701P2_TAC_prod_NPT_10ns_1ps_001_trj"

OUT = Path(".")
PLOT_OUT = Path(".")
SELECT_OUT = Path("../05_selected_frames")
SELECT_OUT.mkdir(parents=True, exist_ok=True)

# Mapping confirmed by atom inspection:
# 1-based atom IDs:
# atoms 1-30 = tacrine
# atom 31 = chloride
# atoms 32,35,38,... = water oxygens
#
# Python 0-based indices:
# atoms 0-29 = tacrine
# atom 30 = chloride
# water oxygens = 31,34,37,...,1843

tacrine_all = np.arange(0, 30)
tacrine_heavy = np.arange(0, 15)
chloride = 30
water_oxygen = np.arange(31, 1846, 3)

candidate_N = [8, 12, 16, 20]
full_shell_cutoff = 5.325  # Angstrom, from previous RDF/min-distance analysis

assert len(tacrine_all) == 30
assert len(tacrine_heavy) == 15
assert chloride == 30
assert len(water_oxygen) == 605

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
    diff = a[:, None, :] - b[None, :, :]
    diff -= box_lengths * np.round(diff / box_lengths)
    return diff

print("Reading trajectory:", TRJ)
tr = traj.read_traj(str(TRJ))
print("Frames:", len(tr))
print("Water oxygens:", len(water_oxygen))
print("Candidate N values:", candidate_N)
print("Full-shell cutoff A:", full_shell_cutoff)

summary_rows = []
per_frame_rows = []

# Store selected-frame candidates for each N.
candidate_frame_records = {N: [] for N in candidate_N}

for frame_i, fr in enumerate(tr):
    pos = fr.pos()
    box_lengths = get_box_lengths(fr)

    tac_pos = pos[tacrine_heavy]
    wat_pos = pos[water_oxygen]

    diff = minimum_image_differences(wat_pos, tac_pos, box_lengths)
    distances = np.sqrt(np.sum(diff * diff, axis=2))  # water O x tacrine heavy
    min_dists = np.min(distances, axis=1)

    order = np.argsort(min_dists)
    sorted_dists = min_dists[order]

    full_shell_count = int(np.sum(min_dists <= full_shell_cutoff))

    row = {
        "frame_index": frame_i,
        "time_ps": float(fr.time),
        "full_shell_count_5p325A": full_shell_count,
    }

    for N in candidate_N:
        nth_distance = float(sorted_dists[N - 1])
        mean_selected_distance = float(np.mean(sorted_dists[:N]))
        max_selected_distance = float(np.max(sorted_dists[:N]))
        selected_inside_full_shell = int(np.sum(sorted_dists[:N] <= full_shell_cutoff))

        row[f"N{N}_nth_nearest_distance_A"] = nth_distance
        row[f"N{N}_mean_selected_distance_A"] = mean_selected_distance
        row[f"N{N}_max_selected_distance_A"] = max_selected_distance
        row[f"N{N}_selected_inside_full_shell"] = selected_inside_full_shell

        # candidate frame is clean if all selected waters are inside the full-shell cutoff
        # and the full shell has at least N waters.
        clean = selected_inside_full_shell == N and full_shell_count >= N
        if clean:
            candidate_frame_records[N].append((frame_i, float(fr.time), nth_distance, full_shell_count))

    per_frame_rows.append(row)

    if (frame_i + 1) % 1000 == 0:
        print(f"Processed {frame_i + 1} / {len(tr)} frames")

# Save per-frame candidate diagnostics.
per_frame_csv = OUT / "microhydration_candidate_distances_by_frame.csv"
fieldnames = list(per_frame_rows[0].keys())
with per_frame_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(per_frame_rows)

# Summary statistics.
summary_csv = OUT / "microhydration_candidate_summary.csv"
with summary_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "N_waters",
        "cluster_atoms_tacrine_plus_Nwaters",
        "clean_candidate_frames",
        "nth_distance_mean_A",
        "nth_distance_median_A",
        "nth_distance_p95_A",
        "nth_distance_max_A",
        "mean_selected_distance_A",
        "max_selected_distance_median_A",
        "interpretation"
    ])

    for N in candidate_N:
        nth = np.array([r[f"N{N}_nth_nearest_distance_A"] for r in per_frame_rows])
        mean_sel = np.array([r[f"N{N}_mean_selected_distance_A"] for r in per_frame_rows])
        max_sel = np.array([r[f"N{N}_max_selected_distance_A"] for r in per_frame_rows])
        clean_count = len(candidate_frame_records[N])
        cluster_atoms = 30 + 3 * N

        if N <= 12:
            interpretation = "small; likely DFT-feasible; may miss some second-contact waters"
        elif N <= 16:
            interpretation = "balanced; likely good microhydration candidate"
        else:
            interpretation = "larger; more complete hydration but higher DFT cost"

        writer.writerow([
            N,
            cluster_atoms,
            clean_count,
            f"{np.mean(nth):.4f}",
            f"{np.median(nth):.4f}",
            f"{np.percentile(nth, 95):.4f}",
            f"{np.max(nth):.4f}",
            f"{np.mean(mean_sel):.4f}",
            f"{np.median(max_sel):.4f}",
            interpretation
        ])

# Select 100 non-adjacent candidate frames for each N.
# Spacing rule: avoid adjacent/similar frames by spreading across time.
# This does not export clusters yet; it only makes frame lists.
for N in candidate_N:
    records = candidate_frame_records[N]
    selected_csv = SELECT_OUT / f"selected_100_frame_candidates_N{N}.csv"

    if len(records) >= 100:
        indices = np.linspace(0, len(records) - 1, 100, dtype=int)
        selected = [records[i] for i in indices]
    else:
        selected = records

    with selected_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["selection_rank", "frame_index", "time_ps", "Nth_nearest_distance_A", "full_shell_count_5p325A"])
        for rank, rec in enumerate(selected, start=1):
            frame_i, time_ps, nth_distance, full_shell_count = rec
            writer.writerow([rank, frame_i, f"{time_ps:.6f}", f"{nth_distance:.6f}", full_shell_count])

# Save decision note.
decision_txt = OUT / "microhydration_candidate_decision_note.txt"
with decision_txt.open("w") as f:
    f.write("Microhydration candidate analysis\n")
    f.write("=================================\n\n")
    f.write("Purpose:\n")
    f.write("The full RDF/minimum-distance analysis gave a whole-solute hydration-envelope cutoff of 5.325 A and dominant occupancy of 48 waters.\n")
    f.write("This script evaluates smaller fixed-composition nearest-water subsets for DFT/MACE feasibility.\n\n")
    f.write("Candidate N values tested: 8, 12, 16, 20 waters.\n")
    f.write("Cluster atom counts:\n")
    for N in candidate_N:
        f.write(f"  N={N}: 30 + 3*{N} = {30 + 3*N} atoms\n")
    f.write("\nRecommendation rule:\n")
    f.write("Prefer the largest N that remains DFT-feasible while keeping selected waters well inside the RDF-derived 5.325 A hydration envelope.\n")
    f.write("Likely final choice should be N=12 or N=16, pending supervisor/DFT feasibility.\n")

# Try plots.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for N in candidate_N:
        nth = np.array([r[f"N{N}_nth_nearest_distance_A"] for r in per_frame_rows])

        plt.figure()
        plt.hist(nth, bins=50)
        plt.axvline(full_shell_cutoff, linestyle="--", label="full-shell cutoff 5.325 A")
        plt.xlabel(f"{N}th nearest water oxygen distance to tacrine / A")
        plt.ylabel("number of frames")
        plt.title(f"N={N} microhydration candidate distance distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_OUT / f"N{N}_nth_nearest_distance_histogram.png", dpi=300)
        plt.close()

    plt.figure()
    labels = []
    data = []
    for N in candidate_N:
        labels.append(str(N))
        data.append([r[f"N{N}_nth_nearest_distance_A"] for r in per_frame_rows])
    plt.boxplot(data, labels=labels)
    plt.axhline(full_shell_cutoff, linestyle="--", label="full-shell cutoff 5.325 A")
    plt.xlabel("N nearest waters")
    plt.ylabel("Nth nearest distance / A")
    plt.title("Microhydration candidate distance comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_OUT / "microhydration_N_comparison_boxplot.png", dpi=300)
    plt.close()

    print("Plots saved successfully.")
except Exception as e:
    print("Plotting failed, but CSV files were saved.")
    print("Plotting error:", repr(e))

print()
print("MICROHYDRATION CANDIDATE ANALYSIS COMPLETE")
print("Frames analysed:", len(tr))
print("Full-shell cutoff A:", full_shell_cutoff)

for N in candidate_N:
    nth = np.array([r[f"N{N}_nth_nearest_distance_A"] for r in per_frame_rows])
    print()
    print(f"N = {N}")
    print("Cluster atoms:", 30 + 3 * N)
    print("Clean candidate frames:", len(candidate_frame_records[N]))
    print("Nth distance mean A:", f"{np.mean(nth):.3f}")
    print("Nth distance median A:", f"{np.median(nth):.3f}")
    print("Nth distance 95th percentile A:", f"{np.percentile(nth, 95):.3f}")
    print("Nth distance max A:", f"{np.max(nth):.3f}")
    print("Selected-frame list:", SELECT_OUT / f"selected_100_frame_candidates_N{N}.csv")

print()
print("Saved:", per_frame_csv)
print("Saved:", summary_csv)
print("Saved:", decision_txt)
