"""Verify the dissertation's path-local zero-force comparator from retained forces."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

EXPECTED_SHA256 = "8f6c01d0d3bdd9eb91d8637e16bda711c26c54ba3d10deb124ca3cc2cc84637a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected", type=Path, default=Path(__file__).resolve().parents[2] / "data/b3lyp_benchmark/zero_force_baseline_derived.json")
    args = parser.parse_args()
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise SystemExit("Input SHA256 does not match the retained corrected Gaussian force table.")
    data = pd.read_csv(args.input)
    if len(data) != 244 * 78 or data.groupby("gaussian_step_1based").size().to_dict() != dict.fromkeys(range(1, 245), 78):
        raise SystemExit("Expected 244 path steps with 78 atoms each.")
    if data.duplicated(["gaussian_step_1based", "atom_index_1based"]).any():
        raise SystemExit("Duplicate step/atom records.")
    force = data[[f"F{axis}_corrected_standard_eV_per_A" for axis in "xyz"]].to_numpy(float)
    if not np.isfinite(force).all():
        raise SystemExit("Non-finite reference force.")
    values = {"component_MAE": float(np.mean(np.abs(force))), "component_RMSE": float(np.sqrt(np.mean(force ** 2))), "vector_RMS": float(np.sqrt(np.mean(np.sum(force ** 2, axis=1))))}
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    if not all(np.isclose(v, expected[k], rtol=1e-12, atol=1e-12) for k, v in values.items()):
        raise SystemExit("Derived metrics do not match the recorded comparator.")
    print(json.dumps({"status": "PASS", "units": "eV/angstrom", **values}, indent=2))


if __name__ == "__main__":
    main()
