from pathlib import Path
from collections import Counter
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Check 100 canonical N16 XYZ files.")
parser.add_argument("--input-dir", type=Path, default=Path("N16_xyz"))
XYZ_DIR = parser.parse_args().input_dir
xyz_files = sorted(XYZ_DIR.glob("cluster_N16_*.xyz"))

expected_n_files = 100
expected_n_atoms = 78
expected_formula = {"C": 13, "H": 47, "N": 2, "O": 16}

problems = []
min_pair_distances = []
max_abs_coords = []

def read_xyz(path):
    with path.open() as f:
        lines = [line.rstrip() for line in f]
    n = int(lines[0])
    comment = lines[1]
    elems = []
    coords = []
    for line in lines[2:]:
        parts = line.split()
        elems.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return n, comment, elems, np.array(coords, dtype=float)

if len(xyz_files) != expected_n_files:
    raise SystemExit(f"Expected {expected_n_files} XYZ files, found {len(xyz_files)}")

for path in xyz_files:
    n, comment, elems, coords = read_xyz(path)

    if n != expected_n_atoms:
        problems.append(f"{path.name}: expected {expected_n_atoms} atoms, got {n}")

    if len(elems) != expected_n_atoms:
        problems.append(f"{path.name}: atom line count mismatch")

    formula = dict(Counter(elems))
    if formula != expected_formula:
        problems.append(f"{path.name}: formula mismatch {formula}")

    if not np.isfinite(coords).all():
        problems.append(f"{path.name}: NaN or infinite coordinate")

    max_abs = float(np.max(np.abs(coords)))
    max_abs_coords.append(max_abs)
    if max_abs > 30.0:
        problems.append(f"{path.name}: very large coordinate magnitude {max_abs:.3f} Å")

    # Minimum interatomic distance, excluding self.
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = float(np.min(dist))
    min_pair_distances.append(min_dist)

    if min_dist < 0.60:
        problems.append(f"{path.name}: suspiciously short interatomic distance {min_dist:.3f} Å")

print("N16 XYZ STRICT QC")
print("=================")
print("XYZ files:", len(xyz_files))
print("Expected files:", expected_n_files)
print("Expected atoms per file:", expected_n_atoms)
print("Expected formula:", expected_formula)
print("Minimum pair distance across dataset:", f"{min(min_pair_distances):.4f} Å")
print("Median minimum pair distance:", f"{np.median(min_pair_distances):.4f} Å")
print("Maximum absolute coordinate:", f"{max(max_abs_coords):.4f} Å")
print("Problems found:", len(problems))

if problems:
    print()
    print("PROBLEMS:")
    for p in problems[:50]:
        print("-", p)
    raise SystemExit("QC FAILED")
else:
    print("QC PASSED")
