from pathlib import Path
import csv
import shutil
import numpy as np

from schrodinger.application.desmond.packages import traj, topo

BASE = Path("../00_raw_production_archive/CHE701P2_TAC_prod_NPT_10ns_1ps_001")
CMS = BASE / "CHE701P2_TAC_prod_NPT_10ns_1ps_001-out.cms"
TRJ = BASE / "CHE701P2_TAC_prod_NPT_10ns_1ps_001_trj"

SELECTED = Path("../05_selected_frames/selected_100_frame_candidates_N16.csv")

OUTDIR = Path("N16_xyz")
OUTDIR.mkdir(parents=True, exist_ok=True)

MANIFEST = Path("cluster_manifest_N16.csv")
QC = Path("cluster_extraction_QC_N16.txt")

N_WATERS = 16
EXPECTED_ATOMS = 30 + 3 * N_WATERS

# Confirmed atom mapping from previous inspection:
# 1-based atom IDs:
#   atoms 1-30 = tacrine
#   atom 31 = chloride
#   atoms 32,35,38,... = water oxygens
#
# Python 0-based indices:
#   atoms 0-29 = tacrine
#   atom 30 = chloride
#   water oxygens = 31,34,37,...,1843

tacrine_all = np.arange(0, 30)
tacrine_heavy = np.arange(0, 15)
chloride = 30
water_oxygen = np.arange(31, 1846, 3)
water_h1 = water_oxygen + 1
water_h2 = water_oxygen + 2

assert len(tacrine_all) == 30
assert len(tacrine_heavy) == 15
assert chloride == 30
assert len(water_oxygen) == 605
assert water_oxygen[0] == 31
assert water_oxygen[-1] == 1843

def safe(atom, attr):
    try:
        value = getattr(atom, attr)
        return "" if value is None else value
    except Exception:
        return ""

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

def minimum_image_vector(diff, box_lengths):
    return diff - box_lengths * np.round(diff / box_lengths)

def unwrap_tacrine(pos, box_lengths):
    # Keep tacrine as a single molecule under periodic boundary conditions.
    ref = pos[0].copy()
    coords = []
    for idx in tacrine_all:
        diff = minimum_image_vector(pos[idx] - ref, box_lengths)
        coords.append(ref + diff)
    return np.array(coords)

def unwrap_water_near_tacrine(pos, water_i, tac_heavy_coords, box_lengths):
    # water_i is the water number in the water_oxygen array.
    o_idx = water_oxygen[water_i]
    h1_idx = water_h1[water_i]
    h2_idx = water_h2[water_i]

    o_raw = pos[o_idx]
    h1_raw = pos[h1_idx]
    h2_raw = pos[h2_idx]

    # Place water oxygen in the periodic image closest to any tacrine heavy atom.
    diffs = o_raw[None, :] - tac_heavy_coords
    diffs_mi = minimum_image_vector(diffs, box_lengths)
    dists = np.sqrt(np.sum(diffs_mi * diffs_mi, axis=1))

    nearest_tac_idx = int(np.argmin(dists))
    min_dist = float(dists[nearest_tac_idx])
    o_unwrapped = tac_heavy_coords[nearest_tac_idx] + diffs_mi[nearest_tac_idx]

    # Keep the water molecule intact by unwrapping hydrogens relative to oxygen.
    h1_unwrapped = o_unwrapped + minimum_image_vector(h1_raw - o_raw, box_lengths)
    h2_unwrapped = o_unwrapped + minimum_image_vector(h2_raw - o_raw, box_lengths)

    return min_dist, np.array([o_unwrapped, h1_unwrapped, h2_unwrapped])

def write_xyz(path, elements, coords, comment):
    with path.open("w") as f:
        f.write(f"{len(elements)}\n")
        f.write(comment + "\n")
        for elem, xyz in zip(elements, coords):
            f.write(f"{elem:2s} {xyz[0]: .8f} {xyz[1]: .8f} {xyz[2]: .8f}\n")

print("CMS:", CMS)
print("TRJ:", TRJ)
print("Selected frames CSV:", SELECTED)

if not CMS.exists():
    raise FileNotFoundError(CMS)
if not TRJ.exists():
    raise FileNotFoundError(TRJ)
if not SELECTED.exists():
    raise FileNotFoundError(SELECTED)

print("Reading CMS atom information...")
msys_model, cms_model = topo.read_cms(str(CMS))
fsys = cms_model.fsys_ct
atoms = [atom for atom in fsys.atom]
elements_all = [str(safe(atom, "element")) for atom in atoms]

if len(elements_all) != 1846:
    raise RuntimeError(f"Expected 1846 atoms from CMS, got {len(elements_all)}")

tacrine_elements = [elements_all[i] for i in tacrine_all]
water_triplet_elements = ["O", "H", "H"]

print("Reading trajectory...")
tr = traj.read_traj(str(TRJ))
print("Trajectory frames:", len(tr))

selected_rows = []
with SELECTED.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        selected_rows.append(row)

if len(selected_rows) == 0:
    raise RuntimeError("No selected frames found.")

print("Selected frames:", len(selected_rows))

# Copy selected-frame list into export folder for provenance.
shutil.copy2(SELECTED, Path("selected_100_frame_candidates_N16_used_for_extraction.csv"))

manifest_rows = []
all_atom_counts_ok = True
all_selected_water_counts_ok = True
max_distance_seen = 0.0

for rank, row in enumerate(selected_rows, start=1):
    frame_index = int(row["frame_index"])
    time_ps = float(row["time_ps"])

    if frame_index < 0 or frame_index >= len(tr):
        raise IndexError(f"Frame index out of range: {frame_index}")

    fr = tr[frame_index]
    pos = fr.pos()
    box_lengths = get_box_lengths(fr)

    tac_coords = unwrap_tacrine(pos, box_lengths)
    tac_heavy_coords = tac_coords[:15]

    # Compute nearest waters for this frame.
    water_records = []
    for wi in range(len(water_oxygen)):
        min_dist, water_coords = unwrap_water_near_tacrine(pos, wi, tac_heavy_coords, box_lengths)
        water_records.append((min_dist, wi, water_coords))

    water_records.sort(key=lambda x: x[0])
    selected_waters = water_records[:N_WATERS]

    selected_water_indices = [wi for _, wi, _ in selected_waters]
    selected_distances = [float(d) for d, _, _ in selected_waters]
    max_selected_distance = max(selected_distances)
    max_distance_seen = max(max_distance_seen, max_selected_distance)

    coords = []
    elements = []

    # Add tacrine.
    for elem, coord in zip(tacrine_elements, tac_coords):
        elements.append(elem)
        coords.append(coord)

    # Add selected waters.
    for dist, wi, water_coords in selected_waters:
        for elem, coord in zip(water_triplet_elements, water_coords):
            elements.append(elem)
            coords.append(coord)

    coords = np.array(coords, dtype=float)

    if len(elements) != EXPECTED_ATOMS:
        all_atom_counts_ok = False
        raise RuntimeError(f"Wrong atom count in frame {frame_index}: {len(elements)}")

    if len(selected_waters) != N_WATERS:
        all_selected_water_counts_ok = False
        raise RuntimeError(f"Wrong selected water count in frame {frame_index}: {len(selected_waters)}")

    # Center the cluster on tacrine heavy-atom centroid.
    center = np.mean(coords[:15], axis=0)
    coords_centered = coords - center

    xyz_name = f"cluster_N16_{rank:03d}_frame{frame_index:05d}.xyz"
    xyz_path = OUTDIR / xyz_name

    comment = (
        f"N16 tacrine microhydration cluster; "
        f"source_frame={frame_index}; time_ps={time_ps:.6f}; "
        f"n_waters=16; atoms=78; "
        f"max_selected_waterO_min_distance_A={max_selected_distance:.6f}; "
        f"selection=nearest_16_water_oxygens_to_tacrine_heavy_atoms"
    )

    write_xyz(xyz_path, elements, coords_centered, comment)

    manifest_rows.append({
        "rank": rank,
        "frame_index": frame_index,
        "time_ps": f"{time_ps:.6f}",
        "xyz_file": str(xyz_path),
        "n_atoms": len(elements),
        "n_waters": N_WATERS,
        "max_selected_waterO_min_distance_A": f"{max_selected_distance:.6f}",
        "selected_water_indices_zero_based": " ".join(str(x) for x in selected_water_indices),
        "selected_water_oxygen_atom_ids_one_based": " ".join(str(int(water_oxygen[x] + 1)) for x in selected_water_indices),
    })

# Save manifest.
with MANIFEST.open("w", newline="") as f:
    fieldnames = [
        "rank",
        "frame_index",
        "time_ps",
        "xyz_file",
        "n_atoms",
        "n_waters",
        "max_selected_waterO_min_distance_A",
        "selected_water_indices_zero_based",
        "selected_water_oxygen_atom_ids_one_based",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(manifest_rows)

# Additional QC from written XYZ files.
xyz_files = sorted(OUTDIR.glob("cluster_N16_*.xyz"))
atom_counts = []
element_count_summary = []

for xyz in xyz_files:
    with xyz.open() as f:
        lines = f.readlines()
    atom_counts.append(int(lines[0].strip()))

    elems = []
    for line in lines[2:]:
        parts = line.split()
        if parts:
            elems.append(parts[0])

    counts = {elem: elems.count(elem) for elem in sorted(set(elems))}
    element_count_summary.append(counts)

unique_atom_counts = sorted(set(atom_counts))
unique_element_counts = []
for counts in element_count_summary:
    if counts not in unique_element_counts:
        unique_element_counts.append(counts)

with QC.open("w") as f:
    f.write("N16 cluster extraction QC\n")
    f.write("=========================\n\n")
    f.write(f"Input trajectory: {TRJ}\n")
    f.write(f"Selected frame CSV: {SELECTED}\n")
    f.write(f"Number of selected frames read: {len(selected_rows)}\n")
    f.write(f"Number of XYZ files written: {len(xyz_files)}\n")
    f.write(f"Expected atoms per cluster: {EXPECTED_ATOMS}\n")
    f.write(f"Unique atom counts in XYZ files: {unique_atom_counts}\n")
    f.write(f"Unique element-count dictionaries: {unique_element_counts}\n")
    f.write(f"All atom counts OK: {all_atom_counts_ok}\n")
    f.write(f"All selected water counts OK: {all_selected_water_counts_ok}\n")
    f.write(f"Maximum selected water-O minimum distance seen: {max_distance_seen:.6f} A\n\n")
    f.write("Interpretation:\n")
    f.write("Each exported XYZ contains protonated tacrine plus the 16 nearest waters.\n")
    f.write("Clusters were unwrapped using minimum-image convention so selected waters remain close to tacrine.\n")
    f.write("Clusters were centered on the tacrine heavy-atom centroid for cleaner nonperiodic DFT preparation.\n")

print()
print("N16 CLUSTER EXTRACTION COMPLETE")
print("Selected frames read:", len(selected_rows))
print("XYZ files written:", len(xyz_files))
print("Expected atoms per XYZ:", EXPECTED_ATOMS)
print("Unique atom counts:", unique_atom_counts)
print("Unique element counts:", unique_element_counts)
print("Max selected water-O distance A:", f"{max_distance_seen:.6f}")
print("Saved manifest:", MANIFEST)
print("Saved QC:", QC)
print("XYZ directory:", OUTDIR)
