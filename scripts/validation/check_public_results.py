"""Check frozen dissertation results from public data; no model inference or private inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[2]


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(actual, expected, message, tolerance=1e-9):
    check(np.allclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=False), message)


def main():
    rank = pd.read_csv(ROOT / "data/b3lyp_md100/md100_b3lyp_mlff_structure_ranking.csv").set_index("structure_id")
    check(len(rank) == 100 and set(rank.index) == set(range(1, 101)), "MD100 IDs must be exactly 1..100.")
    reference = rank.scf_energy_hartree
    check(np.isfinite(rank.select_dtypes(include="number")).all().all(), "Non-finite MD100 values.")
    close(reference.rank(), rank.b3lyp_rank, "B3LYP ranks disagree with absolute energies.")
    check(reference.idxmin() == 57 and rank.loc[57, "expected_frame_from_manifest"] == 5645, "B3LYP minimum/frame mismatch.")
    close((reference.loc[13] - reference.loc[57]) * 27.211386245988, 0.08745685116837759, "B3LYP 13-57 difference mismatch.")
    specifications = [
        ("mace_energy_eV", "mace_rank", .9122952295229522, .7535353535353536, [5, 8, 16]),
        ("aimnet_energy_eV", "aimnet2_rank", .9585598559855985, .8359595959595961, [4, 10, 18]),
        ("uma_energy_eV", "uma_omol_rank", .9573957395739572, .8327272727272729, [5, 9, 18]),
    ]
    for energy, ranking, rho, tau, overlaps in specifications:
        predicted = rank[energy]
        close(predicted.rank(), rank[ranking], f"{ranking} disagrees with absolute energies.")
        close(spearmanr(reference, predicted).statistic, rho, f"{ranking} Spearman mismatch.")
        close(kendalltau(reference, predicted, variant="b").statistic, tau, f"{ranking} Kendall mismatch.")
        check(predicted.idxmin() == 13 and rank.loc[13, "expected_frame_from_manifest"] == 1209, "MLFF minimum/frame mismatch.")
        check(rank.loc[57, ranking] == 2 and rank.loc[13, "b3lyp_rank"] == 2, "Reciprocal second ranks mismatch.")
        for k, expected in zip([5, 10, 20], overlaps):
            check(len(set(reference.nsmallest(k).index) & set(predicted.nsmallest(k).index)) == expected, f"{ranking} top-{k} mismatch.")
    for row in rank.itertuples():
        path = ROOT / "data/n16_xyz" / row.filename
        check(hashlib.sha256(path.read_bytes()).hexdigest() == row.canonical_xyz_sha256, f"XYZ hash mismatch: {row.filename}")
    qc = pd.read_csv(ROOT / "data/b3lyp_md100/gaussian_md100_qc.csv").set_index("structure_id")
    check(len(qc) == 100 and qc.index.is_unique and set(qc.index) == set(rank.index), "QC IDs mismatch.")
    close(qc.loc[rank.index, "scf_energy_hartree"], reference, "QC and ranking energies disagree.")
    for name, value in [("charge", 1), ("multiplicity", 1), ("atom_count", 78), ("force_row_count", 78), ("normal_termination_count", 1)]:
        check(qc[name].eq(value).all(), f"QC {name} mismatch.")
    for name in ["frame_match", "route_ok", "charge_mult_ok", "natoms_ok", "energy_ok", "force_ok", "termination_ok", "geometry_ok"]:
        check(qc[name].eq(True).all(), f"QC {name} failure.")
    histogram = pd.read_csv(ROOT / "data/hydration/water_count_histogram.csv")
    check(histogram.n_frames.sum() == 9982, "Hydration frame count mismatch.")
    check(histogram.loc[histogram.n_frames.idxmax(), "first_shell_water_count"] == 48, "Hydration occupancy mode mismatch.")
    check(histogram.loc[histogram.first_shell_water_count.eq(48), "n_frames"].iloc[0] == 1549, "Modal occupancy frequency mismatch.")
    path = pd.read_csv(ROOT / "data/b3lyp_benchmark/energy_predictions_and_errors.csv").set_index("gaussian_step_1based")
    check(len(path) == 244 and set(path.index) == set(range(1, 245)), "Path IDs must be exactly 1..244.")
    check(np.isfinite(path.select_dtypes(include="number")).all().all(), "Non-finite path values.")
    check(path.scf_energy_eV.idxmin() == 227, "Lowest sampled B3LYP step mismatch.")
    ref = path.scf_energy_eV - path.loc[227, "scf_energy_eV"]
    close(ref, path.b3lyp_relative_energy_eV, "B3LYP path reference mismatch.")
    metrics = pd.read_csv(ROOT / "data/b3lyp_benchmark/energy_metrics.csv").set_index("model")
    for model, column, prefix in [("MACE-OFF23", "mace_energy_eV", "mace"), ("AIMNet2 ensemble", "aimnet2_ensemble_energy_eV", "aimnet2"), ("UMA/OMol", "uma_energy_eV", "uma")]:
        relative = path[column] - path.loc[227, column]
        error = relative - ref
        close(relative, path[prefix + "_relative_energy_eV"], f"{model} path anchor mismatch.")
        close(error, path[prefix + "_relative_energy_error_eV"], f"{model} path errors mismatch.")
        row = metrics.loc[model]
        close(np.mean(abs(error)), row.relative_energy_mae_eV, f"{model} path MAE mismatch.")
        close(np.sqrt(np.mean(error ** 2)), row.relative_energy_rmse_eV, f"{model} path RMSE mismatch.")
        close(spearmanr(ref, relative).statistic, row.spearman_rho, f"{model} path Spearman mismatch.")
        check(path[column].idxmin() == row.predicted_minimum_step, f"{model} sampled minimum mismatch.")
    forces = pd.read_csv(ROOT / "data/b3lyp_benchmark/force_overall_metrics.csv")
    check(forces.n_atomic_vectors.eq(244 * 78).all(), "Force vector count mismatch.")
    check(forces.n_cartesian_components.eq(244 * 78 * 3).all(), "Force component count mismatch.")
    close(forces.rms_vector_error_eV_per_A, np.sqrt(3) * forces.component_rmse_eV_per_A, "Force RMS identity mismatch.")
    recovery = pd.read_csv(ROOT / "data/b3lyp_benchmark/disagreement_recovery.csv")
    close(recovery.recovery_fraction, recovery.overlap_count / recovery.n_selected, "Recovery fraction mismatch.")
    close(recovery.random_expected_fraction, recovery.selection_fraction, "Historical nominal random fraction changed.")
    close(recovery.enrichment_over_random, recovery.recovery_fraction / recovery.random_expected_fraction, "Historical recovery enrichment mismatch.")
    check(np.array_equal(recovery.n_selected, np.ceil(244 * recovery.selection_fraction)), "Retrospective selected counts mismatch.")
    nvt = pd.read_csv(ROOT / "data/diagnostic_dynamics/aimnet2_nvt_timeseries.csv")
    nve = pd.read_csv(ROOT / "data/diagnostic_dynamics/aimnet2_nve_timeseries.csv")
    check(len(nvt) == 102 and len(nve) == 202, "Historical diagnostic row count changed.")
    check(nvt.step.duplicated().sum() == 1 and nve.step.duplicated().sum() == 1, "Historical initial duplicate changed.")
    close(nvt.temperature_K.mean(), 340.8009360028667, "Historical NVT mean mismatch.")
    close(nve.total_energy_eV.iloc[-1] - nve.total_energy_eV.iloc[0], -0.0004474400848267, "NVE drift mismatch.")
    close(nve.total_energy_eV.max() - nve.total_energy_eV.min(), 0.00336690844415, "NVE energy range mismatch.")
    baseline = json.loads((ROOT / "data/b3lyp_benchmark/zero_force_baseline_derived.json").read_text(encoding="utf8"))
    close(baseline["vector_RMS"], np.sqrt(3) * baseline["component_RMSE"], "Zero-force RMS identity mismatch.")
    print("PASS: public MD100 ranks, minima, overlaps, XYZ hashes, QC, hydration, path energy metrics, aggregate identities and historical diagnostic summaries.")
    print("Boundary: aggregate force/recovery checks do not reconstruct private atomic predictions; no new inference or dynamics were run.")


if __name__ == "__main__":
    main()
