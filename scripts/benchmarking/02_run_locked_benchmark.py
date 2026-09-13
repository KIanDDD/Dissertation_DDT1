#!/usr/bin/env python
"""
CHE701P locked B3LYP-referenced numerical benchmark
====================================================

Reads the aligned and frozen B3LYP, MACE-OFF23, AIMNet2 and UMA/OMol outputs,
then calculates path-local relative-energy, atomic-force, disagreement,
prioritisation and timing results.

Scientific scope
----------------
- Reference method: B3LYP/6-31G(d), corrected Standard-orientation forces.
- Energy reference geometry: Gaussian step 227 for every method.
- The 244 structures are sequentially correlated geometries from one
  unconverged Gaussian optimisation path.
- Correlation coefficients are descriptive; inferential p-values are not used.
- AIMNet2 spread is called ensemble disagreement, not calibrated uncertainty.
- MACE-OFF23 received no explicit molecular charge.
- AIMNet2 and UMA/OMol received total charge +1; UMA also received singlet
  multiplicity 1.
- No new electronic-structure or MLFF calculations are performed.

Usage
-----
conda activate che701p-gaussian-analysis
python 02_run_locked_benchmark.py "%STAGE%"

Outputs
-------
05_b3lyp_referenced_benchmark/
    03_tables/
    04_figures/
    05_reports/
    06_manifests/

The script refuses to replace derived results unless --overwrite is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


EXPECTED_STEPS = 244
EXPECTED_ATOMS = 78
REFERENCE_STEP = 227
EPS = 1.0e-12

DEFAULT_STAGE = None

PRIMARY_MODELS = ["MACE-OFF23", "AIMNet2 ensemble", "UMA/OMol"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the locked path-local B3LYP-referenced benchmark after "
            "alignment QC has passed."
        )
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default=DEFAULT_STAGE,
        help="Path to 12_gaussian_path_benchmark.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing derived benchmark outputs deliberately.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV contains no rows: {path}")
    return frame


def safe_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numerical value: {value}")
    return result


def mae(error: np.ndarray) -> float:
    return float(np.mean(np.abs(error)))


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def safe_correlations(x: Iterable[float], y: Iterable[float]) -> dict[str, float]:
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)

    if x_array.shape != y_array.shape:
        raise ValueError(
            f"Correlation arrays differ in shape: {x_array.shape} vs {y_array.shape}"
        )
    if len(x_array) < 3:
        return {"pearson_r": float("nan"), "spearman_rho": float("nan")}
    if np.std(x_array) <= EPS or np.std(y_array) <= EPS:
        return {"pearson_r": float("nan"), "spearman_rho": float("nan")}

    pearson_value = float(pearsonr(x_array, y_array).statistic)
    spearman_value = float(spearmanr(x_array, y_array).statistic)
    return {"pearson_r": pearson_value, "spearman_rho": spearman_value}


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def ensure_outputs_absent(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Derived benchmark outputs already exist. Archive them or rerun "
            "deliberately with --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )


def relative_to_reference_step(
    frame: pd.DataFrame,
    energy_column: str,
    output_column: str,
    step_column: str = "gaussian_step_1based",
) -> pd.DataFrame:
    result = frame.copy()
    reference_rows = result.loc[
        pd.to_numeric(result[step_column]).astype(int) == REFERENCE_STEP,
        energy_column,
    ]
    if len(reference_rows) != 1:
        raise ValueError(
            f"Expected exactly one energy at reference step {REFERENCE_STEP}; "
            f"found {len(reference_rows)}."
        )
    reference_energy = safe_float(reference_rows.iloc[0])
    result[output_column] = pd.to_numeric(
        result[energy_column], errors="raise"
    ).astype(float) - reference_energy
    return result


def energy_metric_row(
    model: str,
    reference_relative: np.ndarray,
    predicted_relative: np.ndarray,
    steps: np.ndarray,
) -> dict[str, Any]:
    error = predicted_relative - reference_relative
    correlations = safe_correlations(reference_relative, predicted_relative)
    predicted_minimum_step = int(steps[np.argmin(predicted_relative)])
    reference_minimum_step = int(steps[np.argmin(reference_relative)])

    return {
        "model": model,
        "n_structures": len(error),
        "reference_step_used": REFERENCE_STEP,
        "reference_minimum_step_in_path": reference_minimum_step,
        "predicted_minimum_step": predicted_minimum_step,
        "predicted_minimum_matches_b3lyp": (
            predicted_minimum_step == reference_minimum_step
        ),
        "relative_energy_mae_eV": mae(error),
        "relative_energy_rmse_eV": rmse(error),
        "relative_energy_max_abs_error_eV": float(np.max(np.abs(error))),
        "relative_energy_mean_signed_error_eV": float(np.mean(error)),
        "pearson_r": correlations["pearson_r"],
        "spearman_rho": correlations["spearman_rho"],
    }


def ranking_overlap_rows(
    model: str,
    reference_relative: np.ndarray,
    predicted_relative: np.ndarray,
    steps: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in (5, 10, 20, 25, 50):
        ref_steps = set(steps[np.argsort(reference_relative)[:k]].astype(int))
        pred_steps = set(steps[np.argsort(predicted_relative)[:k]].astype(int))
        overlap = len(ref_steps.intersection(pred_steps))
        union = len(ref_steps.union(pred_steps))
        rows.append(
            {
                "model": model,
                "top_k": k,
                "overlap_count": overlap,
                "recall_fraction": overlap / k,
                "jaccard_fraction": overlap / union if union else float("nan"),
            }
        )
    return rows


def build_force_error_table(
    model_name: str,
    reference: pd.DataFrame,
    prediction: pd.DataFrame,
    prediction_force_columns: tuple[str, str, str],
) -> pd.DataFrame:
    key_columns = [
        "gaussian_step_1based",
        "atom_index_1based",
        "structure_sha256",
        "element",
        "molecular_group",
        "atom_role",
    ]
    pred_hash_column = "source_structure_sha256"

    ref_columns = key_columns + [
        "Fx_corrected_standard_eV_per_A",
        "Fy_corrected_standard_eV_per_A",
        "Fz_corrected_standard_eV_per_A",
    ]
    pred_columns = [
        "gaussian_step_1based",
        "atom_index_1based",
        pred_hash_column,
        "element",
        "molecular_group",
        "atom_role",
        *prediction_force_columns,
    ]

    ref = reference[ref_columns].copy()
    pred = prediction[pred_columns].copy().rename(
        columns={pred_hash_column: "structure_sha256"}
    )

    merged = ref.merge(
        pred,
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )
    expected_rows = EXPECTED_STEPS * EXPECTED_ATOMS
    if len(merged) != expected_rows:
        raise ValueError(
            f"{model_name}: force merge produced {len(merged)} rows; "
            f"expected {expected_rows}."
        )

    ref_xyz = merged[
        [
            "Fx_corrected_standard_eV_per_A",
            "Fy_corrected_standard_eV_per_A",
            "Fz_corrected_standard_eV_per_A",
        ]
    ].to_numpy(dtype=float)
    pred_xyz = merged[list(prediction_force_columns)].to_numpy(dtype=float)

    delta = pred_xyz - ref_xyz
    ref_magnitude = np.linalg.norm(ref_xyz, axis=1)
    pred_magnitude = np.linalg.norm(pred_xyz, axis=1)
    vector_error = np.linalg.norm(delta, axis=1)

    dot = np.sum(ref_xyz * pred_xyz, axis=1)
    denominator = ref_magnitude * pred_magnitude
    cosine = np.full(len(merged), np.nan, dtype=float)
    valid = denominator > EPS
    cosine[valid] = np.clip(dot[valid] / denominator[valid], -1.0, 1.0)

    angle_degrees = np.full(len(merged), np.nan, dtype=float)
    angle_degrees[valid] = np.degrees(np.arccos(cosine[valid]))

    output = merged[key_columns].copy()
    output.insert(0, "model", model_name)
    output["b3lyp_force_x_eV_per_A"] = ref_xyz[:, 0]
    output["b3lyp_force_y_eV_per_A"] = ref_xyz[:, 1]
    output["b3lyp_force_z_eV_per_A"] = ref_xyz[:, 2]
    output["predicted_force_x_eV_per_A"] = pred_xyz[:, 0]
    output["predicted_force_y_eV_per_A"] = pred_xyz[:, 1]
    output["predicted_force_z_eV_per_A"] = pred_xyz[:, 2]
    output["error_x_eV_per_A"] = delta[:, 0]
    output["error_y_eV_per_A"] = delta[:, 1]
    output["error_z_eV_per_A"] = delta[:, 2]
    output["b3lyp_force_magnitude_eV_per_A"] = ref_magnitude
    output["predicted_force_magnitude_eV_per_A"] = pred_magnitude
    output["force_magnitude_error_eV_per_A"] = pred_magnitude - ref_magnitude
    output["force_vector_error_eV_per_A"] = vector_error
    output["force_direction_cosine"] = cosine
    output["force_direction_angle_degrees"] = angle_degrees
    return output


def force_metrics(
    frame: pd.DataFrame,
    model: str,
    grouping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dx = frame["error_x_eV_per_A"].to_numpy(dtype=float)
    dy = frame["error_y_eV_per_A"].to_numpy(dtype=float)
    dz = frame["error_z_eV_per_A"].to_numpy(dtype=float)
    component_error = np.concatenate([dx, dy, dz])
    vector_error = frame["force_vector_error_eV_per_A"].to_numpy(dtype=float)
    magnitude_error = frame["force_magnitude_error_eV_per_A"].to_numpy(dtype=float)
    cosines = frame["force_direction_cosine"].to_numpy(dtype=float)
    angles = frame["force_direction_angle_degrees"].to_numpy(dtype=float)

    valid_cosines = cosines[np.isfinite(cosines)]
    valid_angles = angles[np.isfinite(angles)]
    component_reference = frame[
        [
            "b3lyp_force_x_eV_per_A",
            "b3lyp_force_y_eV_per_A",
            "b3lyp_force_z_eV_per_A",
        ]
    ].to_numpy(dtype=float).ravel()
    component_prediction = frame[
        [
            "predicted_force_x_eV_per_A",
            "predicted_force_y_eV_per_A",
            "predicted_force_z_eV_per_A",
        ]
    ].to_numpy(dtype=float).ravel()
    correlations = safe_correlations(component_reference, component_prediction)

    row: dict[str, Any] = {
        "model": model,
        "n_atomic_vectors": len(frame),
        "n_cartesian_components": len(component_error),
        "component_mae_eV_per_A": mae(component_error),
        "component_rmse_eV_per_A": rmse(component_error),
        "component_max_abs_error_eV_per_A": float(
            np.max(np.abs(component_error))
        ),
        "mean_vector_error_eV_per_A": float(np.mean(vector_error)),
        "rms_vector_error_eV_per_A": rmse(vector_error),
        "maximum_vector_error_eV_per_A": float(np.max(vector_error)),
        "force_magnitude_mae_eV_per_A": mae(magnitude_error),
        "force_magnitude_rmse_eV_per_A": rmse(magnitude_error),
        "mean_direction_cosine": (
            float(np.mean(valid_cosines)) if len(valid_cosines) else float("nan")
        ),
        "median_direction_angle_degrees": (
            float(np.median(valid_angles)) if len(valid_angles) else float("nan")
        ),
        "component_pearson_r": correlations["pearson_r"],
        "component_spearman_rho": correlations["spearman_rho"],
    }
    if grouping:
        row.update(grouping)
    return row


def per_structure_force_metrics(force_errors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, step), group in force_errors.groupby(
        ["model", "gaussian_step_1based"], sort=True
    ):
        dx = group["error_x_eV_per_A"].to_numpy(dtype=float)
        dy = group["error_y_eV_per_A"].to_numpy(dtype=float)
        dz = group["error_z_eV_per_A"].to_numpy(dtype=float)
        component_error = np.concatenate([dx, dy, dz])
        vector_error = group["force_vector_error_eV_per_A"].to_numpy(dtype=float)

        rows.append(
            {
                "model": model,
                "gaussian_step_1based": int(step),
                "n_atoms": len(group),
                "force_component_mae_eV_per_A": mae(component_error),
                "force_component_rmse_eV_per_A": rmse(component_error),
                "force_vector_error_mean_eV_per_A": float(np.mean(vector_error)),
                "force_vector_error_rms_eV_per_A": rmse(vector_error),
                "force_vector_error_max_eV_per_A": float(np.max(vector_error)),
            }
        )
    return pd.DataFrame(rows)


def correlation_row(
    indicator_name: str,
    target_name: str,
    indicator: Iterable[float],
    target: Iterable[float],
    scope: str,
) -> dict[str, Any]:
    indicator_array = np.asarray(list(indicator), dtype=float)
    target_array = np.asarray(list(target), dtype=float)
    correlations = safe_correlations(indicator_array, target_array)
    return {
        "scope": scope,
        "indicator": indicator_name,
        "target_error": target_name,
        "n_structures": len(indicator_array),
        "pearson_r": correlations["pearson_r"],
        "spearman_rho": correlations["spearman_rho"],
    }


def recovery_rows(
    indicator_name: str,
    target_name: str,
    indicator: np.ndarray,
    target: np.ndarray,
    steps: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_total = len(steps)

    for fraction in (0.05, 0.10, 0.20):
        n_selected = max(1, int(math.ceil(n_total * fraction)))
        indicator_top = set(
            steps[np.argsort(indicator)[::-1][:n_selected]].astype(int)
        )
        target_top = set(steps[np.argsort(target)[::-1][:n_selected]].astype(int))
        overlap = len(indicator_top.intersection(target_top))
        recall = overlap / n_selected
        rows.append(
            {
                "indicator": indicator_name,
                "target_error": target_name,
                "selection_fraction": fraction,
                "n_selected": n_selected,
                "overlap_count": overlap,
                "recovery_fraction": recall,
                "random_expected_fraction": fraction,
                "enrichment_over_random": recall / fraction,
            }
        )
    return rows


def percentile_rank_high(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True, ascending=True)


def main() -> int:
    args = parse_args()
    if args.stage is None:
        raise SystemExit(
            "A private 12_gaussian_path_benchmark stage path is required. "
            "Pass it explicitly as the first argument; raw Gaussian/project files are not redistributed."
        )
    stage = Path(args.stage).expanduser().resolve()
    benchmark = stage / "05_b3lyp_referenced_benchmark"
    table_dir = benchmark / "03_tables"
    figure_dir = benchmark / "04_figures"
    report_dir = benchmark / "05_reports"
    manifest_dir = benchmark / "06_manifests"

    for directory in (table_dir, figure_dir, report_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)

    qc_report = require_file(benchmark / "02_qc" / "benchmark_alignment_qc.txt")
    if "Status: PASS" not in qc_report.read_text(encoding="utf-8"):
        raise RuntimeError("Alignment QC is not PASS. Benchmarking is prohibited.")

    input_paths = {
        "b3lyp_structures": require_file(
            stage
            / "02_reference_extraction"
            / "02_outputs"
            / "gaussian_path_structures.csv"
        ),
        "b3lyp_forces": require_file(
            stage
            / "02_reference_extraction"
            / "02_outputs"
            / "gaussian_path_atomic_forces.csv"
        ),
        "mace_structures": require_file(
            stage
            / "04_mlff_path_predictions"
            / "01_mace"
            / "02_full_path"
            / "mace_gaussian_full_structure_results.csv"
        ),
        "mace_forces": require_file(
            stage
            / "04_mlff_path_predictions"
            / "01_mace"
            / "02_full_path"
            / "mace_gaussian_full_atomic_forces.csv"
        ),
        "mace_metadata": require_file(
            stage
            / "04_mlff_path_predictions"
            / "01_mace"
            / "02_full_path"
            / "mace_gaussian_full_metadata.json"
        ),
        "aim_member_structures": require_file(
            stage
            / "04_mlff_path_predictions"
            / "02_aimnet2"
            / "02_full_path"
            / "aimnet2_gaussian_full_member_structure_results.csv"
        ),
        "aim_member_forces": require_file(
            stage
            / "04_mlff_path_predictions"
            / "02_aimnet2"
            / "02_full_path"
            / "aimnet2_gaussian_full_member_atomic_forces.csv"
        ),
        "aim_ensemble_structures": require_file(
            stage
            / "04_mlff_path_predictions"
            / "02_aimnet2"
            / "02_full_path"
            / "aimnet2_gaussian_full_ensemble_structure_summary.csv"
        ),
        "aim_ensemble_atoms": require_file(
            stage
            / "04_mlff_path_predictions"
            / "02_aimnet2"
            / "02_full_path"
            / "aimnet2_gaussian_full_ensemble_atomic_summary.csv"
        ),
        "aim_metadata": require_file(
            stage
            / "04_mlff_path_predictions"
            / "02_aimnet2"
            / "02_full_path"
            / "aimnet2_gaussian_full_metadata.json"
        ),
        "uma_structures": require_file(
            stage
            / "04_mlff_path_predictions"
            / "03_uma"
            / "02_full_path"
            / "uma_gaussian_full_structure_results.csv"
        ),
        "uma_forces": require_file(
            stage
            / "04_mlff_path_predictions"
            / "03_uma"
            / "02_full_path"
            / "uma_gaussian_full_atomic_forces.csv"
        ),
        "uma_metadata": require_file(
            stage
            / "04_mlff_path_predictions"
            / "03_uma"
            / "02_full_path"
            / "uma_gaussian_full_metadata.json"
        ),
    }

    outputs = {
        "energy_predictions": table_dir / "energy_predictions_and_errors.csv",
        "energy_metrics": table_dir / "energy_metrics.csv",
        "energy_ranking": table_dir / "energy_ranking_overlap.csv",
        "force_errors": table_dir / "force_predictions_and_errors.csv",
        "force_overall": table_dir / "force_overall_metrics.csv",
        "force_element": table_dir / "force_element_metrics.csv",
        "force_group": table_dir / "force_molecular_group_metrics.csv",
        "force_role": table_dir / "force_atom_role_metrics.csv",
        "force_structure": table_dir / "per_structure_force_errors.csv",
        "aim_member_energy": table_dir / "aimnet2_member_energy_metrics.csv",
        "aim_member_force": table_dir / "aimnet2_member_force_metrics.csv",
        "cross_model": table_dir / "cross_model_structure_disagreement.csv",
        "correlations": table_dir / "disagreement_error_correlations.csv",
        "recovery": table_dir / "disagreement_recovery.csv",
        "candidates": table_dir / "prioritised_quantum_candidates.csv",
        "timing": table_dir / "timing_summary.csv",
        "summary": report_dir / "locked_benchmark_summary.txt",
        "metadata": report_dir / "locked_benchmark_metadata.json",
        "manifest": manifest_dir / "locked_benchmark_outputs_SHA256.csv",
    }

    figure_paths = {
        "energy_path_png": figure_dir / "relative_energy_path.png",
        "energy_path_pdf": figure_dir / "relative_energy_path.pdf",
        "energy_parity_png": figure_dir / "relative_energy_parity.png",
        "energy_parity_pdf": figure_dir / "relative_energy_parity.pdf",
        "force_model_png": figure_dir / "force_error_by_model.png",
        "force_model_pdf": figure_dir / "force_error_by_model.pdf",
        "force_element_png": figure_dir / "force_rmse_by_element.png",
        "force_element_pdf": figure_dir / "force_rmse_by_element.pdf",
        "aim_energy_png": figure_dir / "aimnet2_energy_disagreement_vs_error.png",
        "aim_energy_pdf": figure_dir / "aimnet2_energy_disagreement_vs_error.pdf",
        "aim_force_png": figure_dir / "aimnet2_force_disagreement_vs_error.png",
        "aim_force_pdf": figure_dir / "aimnet2_force_disagreement_vs_error.pdf",
        "recovery_png": figure_dir / "disagreement_recovery.png",
        "recovery_pdf": figure_dir / "disagreement_recovery.pdf",
        "timing_png": figure_dir / "observed_cpu_timing.png",
        "timing_pdf": figure_dir / "observed_cpu_timing.pdf",
    }

    ensure_outputs_absent(
        [*outputs.values(), *figure_paths.values()],
        overwrite=args.overwrite,
    )

    b3s = read_csv(input_paths["b3lyp_structures"])
    b3f = read_csv(input_paths["b3lyp_forces"])
    ms = read_csv(input_paths["mace_structures"])
    mf = read_csv(input_paths["mace_forces"])
    ais = read_csv(input_paths["aim_member_structures"])
    aif = read_csv(input_paths["aim_member_forces"])
    aes = read_csv(input_paths["aim_ensemble_structures"])
    aea = read_csv(input_paths["aim_ensemble_atoms"])
    us = read_csv(input_paths["uma_structures"])
    uf = read_csv(input_paths["uma_forces"])

    require_columns(
        b3s,
        {
            "gaussian_step_1based",
            "structure_sha256",
            "scf_energy_eV",
        },
        "B3LYP structures",
    )

    # -----------------------
    # Relative-energy tables
    # -----------------------
    b3_energy = b3s[
        ["gaussian_step_1based", "structure_sha256", "scf_energy_eV"]
    ].copy()
    b3_energy = relative_to_reference_step(
        b3_energy,
        "scf_energy_eV",
        "b3lyp_relative_energy_eV",
    )

    mace_energy = ms[
        [
            "gaussian_step_1based",
            "source_structure_sha256",
            "total_energy_eV",
        ]
    ].rename(
        columns={
            "source_structure_sha256": "structure_sha256",
            "total_energy_eV": "mace_energy_eV",
        }
    )
    mace_energy = relative_to_reference_step(
        mace_energy,
        "mace_energy_eV",
        "mace_relative_energy_eV",
    )

    aim_energy = aes[
        [
            "gaussian_step_1based",
            "source_structure_sha256",
            "energy_mean_eV",
            "energy_std_population_eV",
            "energy_range_eV",
        ]
    ].rename(
        columns={
            "source_structure_sha256": "structure_sha256",
            "energy_mean_eV": "aimnet2_ensemble_energy_eV",
            "energy_std_population_eV": (
                "aimnet2_energy_disagreement_std_eV"
            ),
            "energy_range_eV": "aimnet2_energy_disagreement_range_eV",
        }
    )
    aim_energy = relative_to_reference_step(
        aim_energy,
        "aimnet2_ensemble_energy_eV",
        "aimnet2_relative_energy_eV",
    )

    uma_energy = us[
        [
            "gaussian_step_1based",
            "source_structure_sha256",
            "total_energy_eV",
        ]
    ].rename(
        columns={
            "source_structure_sha256": "structure_sha256",
            "total_energy_eV": "uma_energy_eV",
        }
    )
    uma_energy = relative_to_reference_step(
        uma_energy,
        "uma_energy_eV",
        "uma_relative_energy_eV",
    )

    energy = (
        b3_energy.merge(
            mace_energy,
            on=["gaussian_step_1based", "structure_sha256"],
            validate="one_to_one",
        )
        .merge(
            aim_energy,
            on=["gaussian_step_1based", "structure_sha256"],
            validate="one_to_one",
        )
        .merge(
            uma_energy,
            on=["gaussian_step_1based", "structure_sha256"],
            validate="one_to_one",
        )
        .sort_values("gaussian_step_1based")
        .reset_index(drop=True)
    )
    if len(energy) != EXPECTED_STEPS:
        raise ValueError(f"Energy merge produced {len(energy)} rows.")

    energy["mace_relative_energy_error_eV"] = (
        energy["mace_relative_energy_eV"]
        - energy["b3lyp_relative_energy_eV"]
    )
    energy["aimnet2_relative_energy_error_eV"] = (
        energy["aimnet2_relative_energy_eV"]
        - energy["b3lyp_relative_energy_eV"]
    )
    energy["uma_relative_energy_error_eV"] = (
        energy["uma_relative_energy_eV"]
        - energy["b3lyp_relative_energy_eV"]
    )

    model_energy_columns = {
        "MACE-OFF23": "mace_relative_energy_eV",
        "AIMNet2 ensemble": "aimnet2_relative_energy_eV",
        "UMA/OMol": "uma_relative_energy_eV",
    }
    reference_relative = energy["b3lyp_relative_energy_eV"].to_numpy(dtype=float)
    steps = energy["gaussian_step_1based"].to_numpy(dtype=int)

    energy_metric_rows = []
    energy_ranking_rows = []
    for model, column in model_energy_columns.items():
        predicted = energy[column].to_numpy(dtype=float)
        energy_metric_rows.append(
            energy_metric_row(model, reference_relative, predicted, steps)
        )
        energy_ranking_rows.extend(
            ranking_overlap_rows(model, reference_relative, predicted, steps)
        )

    energy_metrics = pd.DataFrame(energy_metric_rows)
    energy_ranking = pd.DataFrame(energy_ranking_rows)

    # AIMNet2 individual member energy metrics.
    aim_member_energy_rows: list[dict[str, Any]] = []
    for model_alias, group in ais.groupby("model_alias", sort=True):
        member = group[
            [
                "gaussian_step_1based",
                "source_structure_sha256",
                "energy_eV",
            ]
        ].rename(
            columns={
                "source_structure_sha256": "structure_sha256",
                "energy_eV": "member_energy_eV",
            }
        )
        member = relative_to_reference_step(
            member,
            "member_energy_eV",
            "member_relative_energy_eV",
        )
        aligned = b3_energy.merge(
            member,
            on=["gaussian_step_1based", "structure_sha256"],
            validate="one_to_one",
        ).sort_values("gaussian_step_1based")

        row = energy_metric_row(
            str(model_alias),
            aligned["b3lyp_relative_energy_eV"].to_numpy(dtype=float),
            aligned["member_relative_energy_eV"].to_numpy(dtype=float),
            aligned["gaussian_step_1based"].to_numpy(dtype=int),
        )
        aim_member_energy_rows.append(row)

    aim_member_energy_metrics = pd.DataFrame(aim_member_energy_rows)

    # -------------------
    # Atomic-force tables
    # -------------------
    mace_force_errors = build_force_error_table(
        "MACE-OFF23",
        b3f,
        mf,
        (
            "force_x_eV_per_A",
            "force_y_eV_per_A",
            "force_z_eV_per_A",
        ),
    )
    aim_force_errors = build_force_error_table(
        "AIMNet2 ensemble",
        b3f,
        aea,
        (
            "force_mean_x_eV_per_A",
            "force_mean_y_eV_per_A",
            "force_mean_z_eV_per_A",
        ),
    )
    uma_force_errors = build_force_error_table(
        "UMA/OMol",
        b3f,
        uf,
        (
            "force_x_eV_per_A",
            "force_y_eV_per_A",
            "force_z_eV_per_A",
        ),
    )

    force_errors = pd.concat(
        [mace_force_errors, aim_force_errors, uma_force_errors],
        ignore_index=True,
    )

    overall_rows = [
        force_metrics(
            force_errors.loc[force_errors["model"] == model],
            model,
        )
        for model in PRIMARY_MODELS
    ]
    force_overall = pd.DataFrame(overall_rows)

    element_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []

    for model in PRIMARY_MODELS:
        model_frame = force_errors.loc[force_errors["model"] == model]

        for element, group in model_frame.groupby("element", sort=True):
            element_rows.append(
                force_metrics(
                    group,
                    model,
                    {"element": str(element)},
                )
            )

        for molecular_group, group in model_frame.groupby(
            "molecular_group", sort=True
        ):
            group_rows.append(
                force_metrics(
                    group,
                    model,
                    {"molecular_group": str(molecular_group)},
                )
            )

        for atom_role, group in model_frame.groupby("atom_role", sort=True):
            role_rows.append(
                force_metrics(
                    group,
                    model,
                    {"atom_role": str(atom_role)},
                )
            )

    force_element = pd.DataFrame(element_rows)
    force_group = pd.DataFrame(group_rows)
    force_role = pd.DataFrame(role_rows)
    force_structure = per_structure_force_metrics(force_errors)

    # AIMNet2 member force metrics.
    aim_member_force_rows: list[dict[str, Any]] = []
    for model_alias, group in aif.groupby("model_alias", sort=True):
        member_errors = build_force_error_table(
            str(model_alias),
            b3f,
            group,
            (
                "force_x_eV_per_A",
                "force_y_eV_per_A",
                "force_z_eV_per_A",
            ),
        )
        aim_member_force_rows.append(
            force_metrics(member_errors, str(model_alias))
        )
    aim_member_force_metrics = pd.DataFrame(aim_member_force_rows)

    # ---------------------------------------
    # Disagreement and error-prioritisation QC
    # ---------------------------------------
    aim_force_structure = force_structure.loc[
        force_structure["model"] == "AIMNet2 ensemble"
    ].copy()

    aim_structure_disagreement = aes[
        [
            "gaussian_step_1based",
            "energy_std_population_eV",
            "energy_range_eV",
            "per_atom_vector_disagreement_mean_eV_per_A",
            "per_atom_vector_disagreement_rms_eV_per_A",
            "per_atom_vector_disagreement_max_eV_per_A",
        ]
    ].copy()

    aim_eval = (
        energy[
            [
                "gaussian_step_1based",
                "aimnet2_energy_disagreement_std_eV",
                "aimnet2_energy_disagreement_range_eV",
                "aimnet2_relative_energy_error_eV",
            ]
        ]
        .merge(
            aim_force_structure,
            on="gaussian_step_1based",
            validate="one_to_one",
        )
        .merge(
            aim_structure_disagreement,
            on="gaussian_step_1based",
            validate="one_to_one",
            suffixes=("", "_raw"),
        )
    )
    aim_eval["aimnet2_relative_energy_abs_error_eV"] = np.abs(
        aim_eval["aimnet2_relative_energy_error_eV"]
    )

    # Cross-model energy disagreement after common step-227 referencing.
    relative_matrix = energy[
        [
            "mace_relative_energy_eV",
            "aimnet2_relative_energy_eV",
            "uma_relative_energy_eV",
        ]
    ].to_numpy(dtype=float)
    energy_error_matrix = np.column_stack(
        [
            np.abs(energy["mace_relative_energy_error_eV"].to_numpy(dtype=float)),
            np.abs(
                energy["aimnet2_relative_energy_error_eV"].to_numpy(dtype=float)
            ),
            np.abs(energy["uma_relative_energy_error_eV"].to_numpy(dtype=float)),
        ]
    )
    cross_energy_std = np.std(relative_matrix, axis=1, ddof=0)
    cross_energy_range = np.ptp(relative_matrix, axis=1)

    # Cross-model force disagreement from the three aligned force-vector sets.
    force_by_model = {
        model: (
            force_errors.loc[force_errors["model"] == model]
            .sort_values(["gaussian_step_1based", "atom_index_1based"])
            .reset_index(drop=True)
        )
        for model in PRIMARY_MODELS
    }
    force_stack = np.stack(
        [
            force_by_model[model][
                [
                    "predicted_force_x_eV_per_A",
                    "predicted_force_y_eV_per_A",
                    "predicted_force_z_eV_per_A",
                ]
            ].to_numpy(dtype=float)
            for model in PRIMARY_MODELS
        ],
        axis=0,
    )
    model_mean_force = np.mean(force_stack, axis=0)
    model_deviation_norm = np.linalg.norm(
        force_stack - model_mean_force[None, :, :],
        axis=2,
    )
    atomic_cross_model_disagreement = np.sqrt(
        np.mean(np.square(model_deviation_norm), axis=0)
    )

    atomic_keys = force_by_model["MACE-OFF23"][
        ["gaussian_step_1based", "atom_index_1based"]
    ].copy()
    atomic_keys["cross_model_force_disagreement_eV_per_A"] = (
        atomic_cross_model_disagreement
    )

    cross_force_rows: list[dict[str, Any]] = []
    for step, group in atomic_keys.groupby("gaussian_step_1based", sort=True):
        values = group["cross_model_force_disagreement_eV_per_A"].to_numpy(
            dtype=float
        )
        cross_force_rows.append(
            {
                "gaussian_step_1based": int(step),
                "cross_model_force_disagreement_mean_eV_per_A": float(
                    np.mean(values)
                ),
                "cross_model_force_disagreement_rms_eV_per_A": rmse(values),
                "cross_model_force_disagreement_max_eV_per_A": float(
                    np.max(values)
                ),
            }
        )
    cross_force = pd.DataFrame(cross_force_rows)

    force_error_pivot = force_structure.pivot(
        index="gaussian_step_1based",
        columns="model",
        values="force_vector_error_rms_eV_per_A",
    ).reset_index()
    force_error_max_pivot = force_structure.pivot(
        index="gaussian_step_1based",
        columns="model",
        values="force_vector_error_max_eV_per_A",
    ).reset_index()

    cross_model = pd.DataFrame(
        {
            "gaussian_step_1based": steps,
            "cross_model_relative_energy_std_eV": cross_energy_std,
            "cross_model_relative_energy_range_eV": cross_energy_range,
            "mean_model_relative_energy_abs_error_eV": np.mean(
                energy_error_matrix, axis=1
            ),
            "maximum_model_relative_energy_abs_error_eV": np.max(
                energy_error_matrix, axis=1
            ),
        }
    )
    cross_model = (
        cross_model.merge(
            cross_force,
            on="gaussian_step_1based",
            validate="one_to_one",
        )
        .merge(
            force_error_pivot,
            on="gaussian_step_1based",
            validate="one_to_one",
        )
        .merge(
            force_error_max_pivot,
            on="gaussian_step_1based",
            validate="one_to_one",
            suffixes=("_rms", "_max"),
        )
    )

    cross_model["mean_model_force_vector_rmse_eV_per_A"] = cross_model[
        [f"{model}_rms" for model in PRIMARY_MODELS]
    ].mean(axis=1)
    cross_model["maximum_model_force_vector_rmse_eV_per_A"] = cross_model[
        [f"{model}_rms" for model in PRIMARY_MODELS]
    ].max(axis=1)
    cross_model["maximum_model_atomic_vector_error_eV_per_A"] = cross_model[
        [f"{model}_max" for model in PRIMARY_MODELS]
    ].max(axis=1)

    correlation_rows = [
        correlation_row(
            "AIMNet2 energy ensemble std",
            "AIMNet2 absolute relative-energy error",
            aim_eval["aimnet2_energy_disagreement_std_eV"],
            aim_eval["aimnet2_relative_energy_abs_error_eV"],
            "AIMNet2 ensemble",
        ),
        correlation_row(
            "AIMNet2 force-vector disagreement RMS",
            "AIMNet2 force-vector RMSE",
            aim_eval["per_atom_vector_disagreement_rms_eV_per_A"],
            aim_eval["force_vector_error_rms_eV_per_A"],
            "AIMNet2 ensemble",
        ),
        correlation_row(
            "AIMNet2 maximum atomic force disagreement",
            "AIMNet2 maximum atomic force-vector error",
            aim_eval["per_atom_vector_disagreement_max_eV_per_A"],
            aim_eval["force_vector_error_max_eV_per_A"],
            "AIMNet2 ensemble",
        ),
        correlation_row(
            "Cross-model relative-energy std",
            "Mean absolute relative-energy error across models",
            cross_model["cross_model_relative_energy_std_eV"],
            cross_model["mean_model_relative_energy_abs_error_eV"],
            "MACE-AIMNet2-UMA",
        ),
        correlation_row(
            "Cross-model force disagreement RMS",
            "Mean force-vector RMSE across models",
            cross_model["cross_model_force_disagreement_rms_eV_per_A"],
            cross_model["mean_model_force_vector_rmse_eV_per_A"],
            "MACE-AIMNet2-UMA",
        ),
        correlation_row(
            "Cross-model maximum atomic force disagreement",
            "Maximum atomic force error across models",
            cross_model["cross_model_force_disagreement_max_eV_per_A"],
            cross_model["maximum_model_atomic_vector_error_eV_per_A"],
            "MACE-AIMNet2-UMA",
        ),
    ]
    correlations = pd.DataFrame(correlation_rows)

    recovery_rows_all: list[dict[str, Any]] = []
    recovery_rows_all.extend(
        recovery_rows(
            "AIMNet2 energy ensemble std",
            "AIMNet2 absolute relative-energy error",
            aim_eval["aimnet2_energy_disagreement_std_eV"].to_numpy(dtype=float),
            aim_eval["aimnet2_relative_energy_abs_error_eV"].to_numpy(dtype=float),
            aim_eval["gaussian_step_1based"].to_numpy(dtype=int),
        )
    )
    recovery_rows_all.extend(
        recovery_rows(
            "AIMNet2 force disagreement RMS",
            "AIMNet2 force-vector RMSE",
            aim_eval[
                "per_atom_vector_disagreement_rms_eV_per_A"
            ].to_numpy(dtype=float),
            aim_eval["force_vector_error_rms_eV_per_A"].to_numpy(dtype=float),
            aim_eval["gaussian_step_1based"].to_numpy(dtype=int),
        )
    )
    recovery_rows_all.extend(
        recovery_rows(
            "Cross-model force disagreement RMS",
            "Mean model force-vector RMSE",
            cross_model[
                "cross_model_force_disagreement_rms_eV_per_A"
            ].to_numpy(dtype=float),
            cross_model[
                "mean_model_force_vector_rmse_eV_per_A"
            ].to_numpy(dtype=float),
            cross_model["gaussian_step_1based"].to_numpy(dtype=int),
        )
    )
    recovery = pd.DataFrame(recovery_rows_all)

    # Retrospective prioritisation score based only on disagreement indicators.
    candidate = (
        aim_eval[
            [
                "gaussian_step_1based",
                "aimnet2_energy_disagreement_std_eV",
                "per_atom_vector_disagreement_rms_eV_per_A",
                "per_atom_vector_disagreement_max_eV_per_A",
                "aimnet2_relative_energy_abs_error_eV",
                "force_vector_error_rms_eV_per_A",
                "force_vector_error_max_eV_per_A",
            ]
        ]
        .merge(
            cross_model[
                [
                    "gaussian_step_1based",
                    "cross_model_relative_energy_std_eV",
                    "cross_model_force_disagreement_rms_eV_per_A",
                    "cross_model_force_disagreement_max_eV_per_A",
                    "mean_model_relative_energy_abs_error_eV",
                    "mean_model_force_vector_rmse_eV_per_A",
                ]
            ],
            on="gaussian_step_1based",
            validate="one_to_one",
        )
    )

    indicator_columns = [
        "aimnet2_energy_disagreement_std_eV",
        "per_atom_vector_disagreement_rms_eV_per_A",
        "per_atom_vector_disagreement_max_eV_per_A",
        "cross_model_relative_energy_std_eV",
        "cross_model_force_disagreement_rms_eV_per_A",
        "cross_model_force_disagreement_max_eV_per_A",
    ]
    rank_columns = []
    for column in indicator_columns:
        rank_column = f"{column}_percentile"
        candidate[rank_column] = percentile_rank_high(candidate[column])
        rank_columns.append(rank_column)

    candidate["disagreement_priority_score"] = candidate[rank_columns].mean(axis=1)
    candidate = candidate.sort_values(
        ["disagreement_priority_score", "gaussian_step_1based"],
        ascending=[False, True],
    ).reset_index(drop=True)
    candidate.insert(0, "priority_rank", np.arange(1, len(candidate) + 1))
    candidate["recommended_for_targeted_qm_review"] = (
        candidate["priority_rank"] <= 20
    )
    prioritised_candidates = candidate.head(50).copy()

    # -------
    # Timing
    # -------
    mace_metadata = json.loads(
        input_paths["mace_metadata"].read_text(encoding="utf-8")
    )
    aim_metadata = json.loads(
        input_paths["aim_metadata"].read_text(encoding="utf-8")
    )
    uma_metadata = json.loads(
        input_paths["uma_metadata"].read_text(encoding="utf-8")
    )

    timing_rows = [
        {
            "workflow": "MACE-OFF23-medium",
            "device": "CPU",
            "geometry_count": 244,
            "member_evaluation_count": 244,
            "model_load_seconds": mace_metadata.get("model_load_seconds"),
            "calculation_or_wall_seconds": mace_metadata.get(
                "total_calculation_seconds"
            ),
            "seconds_per_geometry": (
                float(mace_metadata["total_calculation_seconds"]) / 244
            ),
            "geometries_per_second": (
                244 / float(mace_metadata["total_calculation_seconds"])
            ),
            "notes": "Single MACE model; no explicit molecular charge input.",
        },
        {
            "workflow": "AIMNet2 four-member ensemble",
            "device": "CPU, eager execution",
            "geometry_count": 244,
            "member_evaluation_count": 976,
            "model_load_seconds": sum(
                float(value)
                for value in aim_metadata.get("model_load_seconds", {}).values()
            ),
            "calculation_or_wall_seconds": aim_metadata.get(
                "total_wall_time_seconds"
            ),
            "seconds_per_geometry": (
                float(aim_metadata["total_wall_time_seconds"]) / 244
            ),
            "geometries_per_second": (
                244 / float(aim_metadata["total_wall_time_seconds"])
            ),
            "notes": (
                "Four member predictions per geometry; total charge +1; "
                "TORCH_COMPILE_DISABLE=1."
            ),
        },
        {
            "workflow": "UMA-S/OMol",
            "device": "CPU",
            "geometry_count": 244,
            "member_evaluation_count": 244,
            "model_load_seconds": uma_metadata.get("model_load_seconds"),
            "calculation_or_wall_seconds": uma_metadata.get(
                "total_calculation_seconds"
            ),
            "seconds_per_geometry": (
                float(uma_metadata["total_calculation_seconds"]) / 244
            ),
            "geometries_per_second": (
                244 / float(uma_metadata["total_calculation_seconds"])
            ),
            "notes": "Charge +1 and singlet multiplicity 1.",
        },
    ]
    timing = pd.DataFrame(timing_rows)

    # Write tables before figures.
    write_csv(outputs["energy_predictions"], energy)
    write_csv(outputs["energy_metrics"], energy_metrics)
    write_csv(outputs["energy_ranking"], energy_ranking)
    write_csv(outputs["force_errors"], force_errors)
    write_csv(outputs["force_overall"], force_overall)
    write_csv(outputs["force_element"], force_element)
    write_csv(outputs["force_group"], force_group)
    write_csv(outputs["force_role"], force_role)
    write_csv(outputs["force_structure"], force_structure)
    write_csv(outputs["aim_member_energy"], aim_member_energy_metrics)
    write_csv(outputs["aim_member_force"], aim_member_force_metrics)
    write_csv(outputs["cross_model"], cross_model)
    write_csv(outputs["correlations"], correlations)
    write_csv(outputs["recovery"], recovery)
    write_csv(outputs["candidates"], prioritised_candidates)
    write_csv(outputs["timing"], timing)

    # -------
    # Figures
    # -------
    plt.figure(figsize=(8.0, 5.0))
    plt.plot(steps, reference_relative, label="B3LYP/6-31G(d)")
    for model, column in model_energy_columns.items():
        plt.plot(steps, energy[column], label=model)
    plt.axvline(REFERENCE_STEP, linestyle="--", linewidth=1)
    plt.xlabel("Gaussian optimisation step")
    plt.ylabel("Relative energy referenced to step 227 (eV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_paths["energy_path_png"], dpi=300)
    plt.savefig(figure_paths["energy_path_pdf"])
    plt.close()

    plt.figure(figsize=(6.2, 6.0))
    for model, column in model_energy_columns.items():
        plt.scatter(
            reference_relative,
            energy[column].to_numpy(dtype=float),
            s=18,
            alpha=0.65,
            label=model,
        )
    combined = np.concatenate(
        [
            reference_relative,
            *[
                energy[column].to_numpy(dtype=float)
                for column in model_energy_columns.values()
            ],
        ]
    )
    minimum = float(np.min(combined))
    maximum = float(np.max(combined))
    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    plt.xlabel("B3LYP relative energy (eV)")
    plt.ylabel("MLFF relative energy (eV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_paths["energy_parity_png"], dpi=300)
    plt.savefig(figure_paths["energy_parity_pdf"])
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    model_order = PRIMARY_MODELS
    values = [
        force_structure.loc[
            force_structure["model"] == model,
            "force_vector_error_rms_eV_per_A",
        ].to_numpy(dtype=float)
        for model in model_order
    ]
    plt.boxplot(values, tick_labels=model_order, showfliers=False)
    plt.ylabel("Per-structure force-vector RMSE (eV Å$^{-1}$)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(figure_paths["force_model_png"], dpi=300)
    plt.savefig(figure_paths["force_model_pdf"])
    plt.close()

    element_pivot = force_element.pivot(
        index="element",
        columns="model",
        values="rms_vector_error_eV_per_A",
    )
    element_pivot = element_pivot.reindex(columns=PRIMARY_MODELS)
    plt.figure(figsize=(7.2, 4.8))
    element_pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("RMS atomic force-vector error (eV Å$^{-1}$)")
    plt.xlabel("Element")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figure_paths["force_element_png"], dpi=300)
    plt.savefig(figure_paths["force_element_pdf"])
    plt.close()

    plt.figure(figsize=(6.2, 5.0))
    plt.scatter(
        aim_eval["aimnet2_energy_disagreement_std_eV"],
        aim_eval["aimnet2_relative_energy_abs_error_eV"],
        s=20,
        alpha=0.7,
    )
    plt.xlabel("AIMNet2 ensemble energy disagreement, std (eV)")
    plt.ylabel("Absolute relative-energy error vs B3LYP (eV)")
    plt.tight_layout()
    plt.savefig(figure_paths["aim_energy_png"], dpi=300)
    plt.savefig(figure_paths["aim_energy_pdf"])
    plt.close()

    plt.figure(figsize=(6.2, 5.0))
    plt.scatter(
        aim_eval["per_atom_vector_disagreement_rms_eV_per_A"],
        aim_eval["force_vector_error_rms_eV_per_A"],
        s=20,
        alpha=0.7,
    )
    plt.xlabel("AIMNet2 force disagreement RMS (eV Å$^{-1}$)")
    plt.ylabel("AIMNet2 force-vector RMSE vs B3LYP (eV Å$^{-1}$)")
    plt.tight_layout()
    plt.savefig(figure_paths["aim_force_png"], dpi=300)
    plt.savefig(figure_paths["aim_force_pdf"])
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    for indicator, group in recovery.groupby("indicator", sort=False):
        plt.plot(
            100.0 * group["selection_fraction"],
            100.0 * group["recovery_fraction"],
            marker="o",
            label=indicator,
        )
    fractions = np.array([5.0, 10.0, 20.0])
    plt.plot(fractions, fractions, linestyle="--", label="Random expectation")
    plt.xlabel("Fraction prioritised (%)")
    plt.ylabel("High-error configurations recovered (%)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_paths["recovery_png"], dpi=300)
    plt.savefig(figure_paths["recovery_pdf"])
    plt.close()

    plt.figure(figsize=(7.0, 4.8))
    plt.bar(
        timing["workflow"],
        timing["calculation_or_wall_seconds"],
    )
    plt.ylabel("Observed CPU calculation/wall time (s)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(figure_paths["timing_png"], dpi=300)
    plt.savefig(figure_paths["timing_pdf"])
    plt.close()

    # ----------------
    # Locked summaries
    # ----------------
    best_energy_model = energy_metrics.sort_values(
        "relative_energy_rmse_eV"
    ).iloc[0]
    best_force_model = force_overall.sort_values(
        "rms_vector_error_eV_per_A"
    ).iloc[0]

    aim_energy_corr = correlations.loc[
        correlations["indicator"] == "AIMNet2 energy ensemble std"
    ].iloc[0]
    aim_force_corr = correlations.loc[
        correlations["indicator"] == "AIMNet2 force-vector disagreement RMS"
    ].iloc[0]

    summary_lines = [
        "CHE701P locked B3LYP-referenced benchmark",
        "=" * 45,
        "Status: PASS",
        f"Reference energy geometry: Gaussian step {REFERENCE_STEP}",
        f"Structures: {EXPECTED_STEPS}",
        f"Atoms per structure: {EXPECTED_ATOMS}",
        "",
        "Scope",
        "-----",
        (
            "Path-local comparison against B3LYP/6-31G(d) over 244 sequentially "
            "correlated geometries from one unconverged optimisation path."
        ),
        "No claim of global transferability or independent-sample validation.",
        "",
        "Energy",
        "------",
        (
            f"Lowest relative-energy RMSE: {best_energy_model['model']} "
            f"({best_energy_model['relative_energy_rmse_eV']:.6g} eV)."
        ),
        "",
    ]
    for row in energy_metrics.to_dict(orient="records"):
        summary_lines.append(
            f"{row['model']}: MAE={row['relative_energy_mae_eV']:.6g} eV; "
            f"RMSE={row['relative_energy_rmse_eV']:.6g} eV; "
            f"Pearson r={row['pearson_r']:.6g}; "
            f"Spearman rho={row['spearman_rho']:.6g}; "
            f"predicted minimum step={row['predicted_minimum_step']}."
        )

    summary_lines.extend(
        [
            "",
            "Forces",
            "------",
            (
                f"Lowest RMS atomic force-vector error: "
                f"{best_force_model['model']} "
                f"({best_force_model['rms_vector_error_eV_per_A']:.6g} "
                "eV/A)."
            ),
        ]
    )
    for row in force_overall.to_dict(orient="records"):
        summary_lines.append(
            f"{row['model']}: component MAE="
            f"{row['component_mae_eV_per_A']:.6g} eV/A; "
            f"component RMSE={row['component_rmse_eV_per_A']:.6g} eV/A; "
            f"RMS vector error={row['rms_vector_error_eV_per_A']:.6g} eV/A."
        )

    summary_lines.extend(
        [
            "",
            "AIMNet2 disagreement",
            "--------------------",
            (
                "Energy disagreement vs absolute relative-energy error: "
                f"Pearson r={aim_energy_corr['pearson_r']:.6g}; "
                f"Spearman rho={aim_energy_corr['spearman_rho']:.6g}."
            ),
            (
                "Force disagreement vs force-vector RMSE: "
                f"Pearson r={aim_force_corr['pearson_r']:.6g}; "
                f"Spearman rho={aim_force_corr['spearman_rho']:.6g}."
            ),
            (
                "These coefficients assess retrospective ranking utility; "
                "they do not establish calibrated uncertainty."
            ),
            "",
            "Files",
            "-----",
            f"Tables: {table_dir}",
            f"Figures: {figure_dir}",
            f"Reports: {report_dir}",
            "",
        ]
    )
    outputs["summary"].write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    metadata = {
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference_method": "B3LYP/6-31G(d)",
        "reference_energy_step": REFERENCE_STEP,
        "structures": EXPECTED_STEPS,
        "atoms_per_structure": EXPECTED_ATOMS,
        "primary_models": PRIMARY_MODELS,
        "correlation_p_values_reported": False,
        "reason_p_values_omitted": (
            "The 244 path geometries are sequentially correlated rather than "
            "independent observations."
        ),
        "source_files": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in input_paths.items()
        },
        "scientific_limits": [
            "Path-local benchmark only.",
            "Sequentially correlated geometries from one unconverged path.",
            "MACE-OFF23 received no explicit charge.",
            "AIMNet2 disagreement is not calibrated uncertainty.",
            "B3LYP/6-31G(d) is the local computational reference, not experiment.",
        ],
    }
    outputs["metadata"].write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest_targets = [
        path
        for path in [*outputs.values(), *figure_paths.values()]
        if path != outputs["manifest"]
    ]
    manifest_rows = [
        {
            "filename": path.name,
            "relative_path": str(path.relative_to(benchmark)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in manifest_targets
    ]
    pd.DataFrame(manifest_rows).to_csv(outputs["manifest"], index=False)

    print(outputs["summary"].read_text(encoding="utf-8"))
    print(f"Energy metrics: {outputs['energy_metrics']}")
    print(f"Force metrics: {outputs['force_overall']}")
    print(f"Correlations: {outputs['correlations']}")
    print(f"Candidates: {outputs['candidates']}")
    print(f"Manifest: {outputs['manifest']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        raise SystemExit(1)

