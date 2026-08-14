#!/usr/bin/env python
"""
CHE701P benchmark alignment and integrity QC
=============================================

Validates exact alignment among:
- corrected B3LYP/6-31G(d) Gaussian reference data;
- full-path geometry index;
- MACE-OFF23 predictions;
- four-member AIMNet2 predictions and ensemble summaries;
- UMA/OMol predictions.

This script performs no benchmarking and creates no scientific performance
claims. It establishes that every later energy/force comparison joins the same
Gaussian step, authoritative structure hash, atom index, element and molecular
group.

Usage
-----
conda activate che701p-gaussian-analysis

python 01_validate_benchmark_alignment.py "%STAGE%"

Outputs
-------
05_b3lyp_referenced_benchmark/02_qc/
    benchmark_alignment_qc.txt
    benchmark_alignment_summary.json
    benchmark_alignment_counts.csv

05_b3lyp_referenced_benchmark/06_manifests/
    benchmark_alignment_qc_outputs_SHA256.csv
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
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_STEPS = 244
EXPECTED_ATOMS = 78
EXPECTED_ATOMIC_ROWS = EXPECTED_STEPS * EXPECTED_ATOMS
EXPECTED_AIMNET_MEMBERS = 4
EXPECTED_AIMNET_MEMBER_STRUCTURE_ROWS = EXPECTED_STEPS * EXPECTED_AIMNET_MEMBERS
EXPECTED_AIMNET_MEMBER_ATOMIC_ROWS = (
    EXPECTED_STEPS * EXPECTED_AIMNET_MEMBERS * EXPECTED_ATOMS
)

AIMNET_MODELS = {
    "aimnet2",
    "aimnet2-wb97m-d3_1",
    "aimnet2-wb97m-d3_2",
    "aimnet2-wb97m-d3_3",
}

DEFAULT_STAGE = None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate exact alignment of all B3LYP and MLFF benchmark tables."
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
        help="Replace previous derived QC outputs.",
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
        raise ValueError(f"CSV contains no data rows: {path}")
    return frame


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def all_finite(frame: pd.DataFrame, columns: list[str]) -> bool:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(values).all())


def unique_key(frame: pd.DataFrame, columns: list[str]) -> bool:
    return not bool(frame.duplicated(columns).any())


def sequential_steps(values: pd.Series) -> bool:
    steps = sorted(pd.to_numeric(values, errors="raise").astype(int).unique().tolist())
    return steps == list(range(1, EXPECTED_STEPS + 1))


def exact_string_mapping(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    left_value: str,
    right_value: str,
) -> bool:
    left_map = (
        left[[key, left_value]]
        .drop_duplicates()
        .assign(**{key: lambda x: pd.to_numeric(x[key]).astype(int)})
        .sort_values(key)
        .reset_index(drop=True)
    )
    right_map = (
        right[[key, right_value]]
        .drop_duplicates()
        .assign(**{key: lambda x: pd.to_numeric(x[key]).astype(int)})
        .sort_values(key)
        .reset_index(drop=True)
    )
    if len(left_map) != len(right_map):
        return False
    return bool(
        np.array_equal(left_map[key].to_numpy(), right_map[key].to_numpy())
        and np.array_equal(
            left_map[left_value].astype(str).to_numpy(),
            right_map[right_value].astype(str).to_numpy(),
        )
    )


def atom_alignment_matches(
    reference: pd.DataFrame,
    prediction: pd.DataFrame,
    prediction_hash_column: str,
) -> bool:
    reference_columns = [
        "gaussian_step_1based",
        "atom_index_1based",
        "structure_sha256",
        "element",
        "molecular_group",
        "atom_role",
    ]
    prediction_columns = [
        "gaussian_step_1based",
        "atom_index_1based",
        prediction_hash_column,
        "element",
        "molecular_group",
        "atom_role",
    ]

    ref = reference[reference_columns].copy()
    pred = prediction[prediction_columns].copy()

    ref["gaussian_step_1based"] = pd.to_numeric(
        ref["gaussian_step_1based"]
    ).astype(int)
    pred["gaussian_step_1based"] = pd.to_numeric(
        pred["gaussian_step_1based"]
    ).astype(int)
    ref["atom_index_1based"] = pd.to_numeric(ref["atom_index_1based"]).astype(int)
    pred["atom_index_1based"] = pd.to_numeric(pred["atom_index_1based"]).astype(int)

    ref = ref.sort_values(
        ["gaussian_step_1based", "atom_index_1based"]
    ).reset_index(drop=True)
    pred = pred.sort_values(
        ["gaussian_step_1based", "atom_index_1based"]
    ).reset_index(drop=True)

    if len(ref) != len(pred):
        return False

    comparisons = [
        np.array_equal(
            ref["gaussian_step_1based"].to_numpy(),
            pred["gaussian_step_1based"].to_numpy(),
        ),
        np.array_equal(
            ref["atom_index_1based"].to_numpy(),
            pred["atom_index_1based"].to_numpy(),
        ),
        np.array_equal(
            ref["structure_sha256"].astype(str).to_numpy(),
            pred[prediction_hash_column].astype(str).to_numpy(),
        ),
        np.array_equal(
            ref["element"].astype(str).to_numpy(),
            pred["element"].astype(str).to_numpy(),
        ),
        np.array_equal(
            ref["molecular_group"].astype(str).to_numpy(),
            pred["molecular_group"].astype(str).to_numpy(),
        ),
        np.array_equal(
            ref["atom_role"].astype(str).to_numpy(),
            pred["atom_role"].astype(str).to_numpy(),
        ),
    ]
    return all(comparisons)


def ensure_outputs_absent(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Derived QC outputs already exist. Archive them or rerun with "
            "--overwrite:\n" + "\n".join(str(path) for path in existing)
        )


def main() -> int:
    args = parse_args()
    if args.stage is None:
        raise SystemExit(
            "A private 12_gaussian_path_benchmark stage path is required. "
            "Pass it explicitly as the first argument; raw Gaussian/project files are not redistributed."
        )
    stage = Path(args.stage).expanduser().resolve()

    benchmark = stage / "05_b3lyp_referenced_benchmark"
    qc_dir = benchmark / "02_qc"
    manifest_dir = benchmark / "06_manifests"
    qc_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    report_path = qc_dir / "benchmark_alignment_qc.txt"
    summary_path = qc_dir / "benchmark_alignment_summary.json"
    counts_path = qc_dir / "benchmark_alignment_counts.csv"
    manifest_path = manifest_dir / "benchmark_alignment_qc_outputs_SHA256.csv"

    ensure_outputs_absent(
        [report_path, summary_path, counts_path, manifest_path],
        overwrite=args.overwrite,
    )

    paths = {
        "input_index": require_file(
            stage
            / "04_mlff_path_predictions"
            / "00_full_path_inputs"
            / "full_path_input_index.csv"
        ),
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
        "b3lyp_qc": require_file(
            stage
            / "02_reference_extraction"
            / "02_outputs"
            / "gaussian_path_extraction_qc.txt"
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
        "mace_summary": require_file(
            stage
            / "04_mlff_path_predictions"
            / "01_mace"
            / "02_full_path"
            / "mace_gaussian_full_summary.txt"
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
        "aim_summary": require_file(
            stage
            / "04_mlff_path_predictions"
            / "02_aimnet2"
            / "02_full_path"
            / "aimnet2_gaussian_full_summary.txt"
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
        "uma_summary": require_file(
            stage
            / "04_mlff_path_predictions"
            / "03_uma"
            / "02_full_path"
            / "uma_gaussian_full_summary.txt"
        ),
    }

    source_hashes = {name: sha256_file(path) for name, path in paths.items()}

    if "Status: PASS" not in paths["b3lyp_qc"].read_text(encoding="utf-8"):
        raise RuntimeError("B3LYP extraction QC is not PASS.")
    if "Status: PASS" not in paths["mace_summary"].read_text(encoding="utf-8"):
        raise RuntimeError("MACE full-path summary is not PASS.")
    if "Status: PASS" not in paths["aim_summary"].read_text(encoding="utf-8"):
        raise RuntimeError("AIMNet2 full-path summary is not PASS.")
    if "Status: PASS" not in paths["uma_summary"].read_text(encoding="utf-8"):
        raise RuntimeError("UMA full-path summary is not PASS.")

    index = read_csv(paths["input_index"])
    b3s = read_csv(paths["b3lyp_structures"])
    b3f = read_csv(paths["b3lyp_forces"])
    ms = read_csv(paths["mace_structures"])
    mf = read_csv(paths["mace_forces"])
    ais = read_csv(paths["aim_member_structures"])
    aif = read_csv(paths["aim_member_forces"])
    aes = read_csv(paths["aim_ensemble_structures"])
    aea = read_csv(paths["aim_ensemble_atoms"])
    us = read_csv(paths["uma_structures"])
    uf = read_csv(paths["uma_forces"])

    require_columns(
        index,
        {
            "frame_index_0based",
            "gaussian_step_1based",
            "input_filename",
            "input_file_sha256",
            "source_structure_sha256",
            "atom_count",
        },
        "full-path input index",
    )
    require_columns(
        b3s,
        {
            "frame_index_0based",
            "gaussian_step_1based",
            "structure_sha256",
            "scf_energy_eV",
        },
        "B3LYP structure table",
    )
    require_columns(
        b3f,
        {
            "frame_index_0based",
            "gaussian_step_1based",
            "structure_sha256",
            "atom_index_1based",
            "element",
            "molecular_group",
            "atom_role",
            "Fx_corrected_standard_eV_per_A",
            "Fy_corrected_standard_eV_per_A",
            "Fz_corrected_standard_eV_per_A",
        },
        "B3LYP force table",
    )
    require_columns(
        ms,
        {
            "gaussian_step_1based",
            "source_structure_sha256",
            "total_energy_eV",
            "status",
            "intended_system_charge_e",
            "explicit_charge_input",
        },
        "MACE structure table",
    )
    require_columns(
        mf,
        {
            "gaussian_step_1based",
            "source_structure_sha256",
            "atom_index_1based",
            "element",
            "molecular_group",
            "atom_role",
            "force_x_eV_per_A",
            "force_y_eV_per_A",
            "force_z_eV_per_A",
        },
        "MACE force table",
    )
    require_columns(
        ais,
        {
            "gaussian_step_1based",
            "source_structure_sha256",
            "member_index_0based",
            "model_alias",
            "charge_e",
            "energy_eV",
            "status",
        },
        "AIMNet2 member structure table",
    )
    require_columns(
        aif,
        {
            "gaussian_step_1based",
            "source_structure_sha256",
            "member_index_0based",
            "model_alias",
            "charge_e",
            "atom_index_1based",
            "element",
            "molecular_group",
            "atom_role",
            "force_x_eV_per_A",
            "force_y_eV_per_A",
            "force_z_eV_per_A",
        },
        "AIMNet2 member force table",
    )
    require_columns(
        aes,
        {
            "gaussian_step_1based",
            "source_structure_sha256",
            "member_count",
            "charge_e",
            "energy_mean_eV",
            "energy_std_population_eV",
        },
        "AIMNet2 ensemble structure table",
    )
    require_columns(
        aea,
        {
            "gaussian_step_1based",
            "source_structure_sha256",
            "member_count",
            "charge_e",
            "atom_index_1based",
            "element",
            "molecular_group",
            "atom_role",
            "force_mean_x_eV_per_A",
            "force_mean_y_eV_per_A",
            "force_mean_z_eV_per_A",
        },
        "AIMNet2 ensemble atomic table",
    )
    require_columns(
        us,
        {
            "gaussian_step_1based",
            "source_structure_sha256",
            "charge_e",
            "spin_multiplicity",
            "total_energy_eV",
            "status",
        },
        "UMA structure table",
    )
    require_columns(
        uf,
        {
            "gaussian_step_1based",
            "source_structure_sha256",
            "charge_e",
            "spin_multiplicity",
            "atom_index_1based",
            "element",
            "molecular_group",
            "atom_role",
            "force_x_eV_per_A",
            "force_y_eV_per_A",
            "force_z_eV_per_A",
        },
        "UMA force table",
    )

    checks: dict[str, bool] = {}

    checks["input_index_rows_244"] = len(index) == EXPECTED_STEPS
    checks["b3lyp_structure_rows_244"] = len(b3s) == EXPECTED_STEPS
    checks["b3lyp_force_rows_19032"] = len(b3f) == EXPECTED_ATOMIC_ROWS
    checks["mace_structure_rows_244"] = len(ms) == EXPECTED_STEPS
    checks["mace_force_rows_19032"] = len(mf) == EXPECTED_ATOMIC_ROWS
    checks["aim_member_structure_rows_976"] = (
        len(ais) == EXPECTED_AIMNET_MEMBER_STRUCTURE_ROWS
    )
    checks["aim_member_force_rows_76128"] = (
        len(aif) == EXPECTED_AIMNET_MEMBER_ATOMIC_ROWS
    )
    checks["aim_ensemble_structure_rows_244"] = len(aes) == EXPECTED_STEPS
    checks["aim_ensemble_atomic_rows_19032"] = len(aea) == EXPECTED_ATOMIC_ROWS
    checks["uma_structure_rows_244"] = len(us) == EXPECTED_STEPS
    checks["uma_force_rows_19032"] = len(uf) == EXPECTED_ATOMIC_ROWS

    for name, frame in {
        "input_index": index,
        "b3lyp_structures": b3s,
        "b3lyp_forces": b3f,
        "mace_structures": ms,
        "mace_forces": mf,
        "aim_member_structures": ais,
        "aim_member_forces": aif,
        "aim_ensemble_structures": aes,
        "aim_ensemble_atoms": aea,
        "uma_structures": us,
        "uma_forces": uf,
    }.items():
        checks[f"{name}_steps_1_to_244"] = sequential_steps(
            frame["gaussian_step_1based"]
        )

    checks["b3lyp_unique_step_hash_mapping"] = unique_key(
        b3s, ["gaussian_step_1based", "structure_sha256"]
    )
    checks["b3lyp_unique_step_atom_keys"] = unique_key(
        b3f, ["gaussian_step_1based", "atom_index_1based"]
    )
    checks["mace_unique_step_atom_keys"] = unique_key(
        mf, ["gaussian_step_1based", "atom_index_1based"]
    )
    checks["aim_member_unique_structure_keys"] = unique_key(
        ais, ["model_alias", "gaussian_step_1based"]
    )
    checks["aim_member_unique_atomic_keys"] = unique_key(
        aif, ["model_alias", "gaussian_step_1based", "atom_index_1based"]
    )
    checks["aim_ensemble_unique_step_atom_keys"] = unique_key(
        aea, ["gaussian_step_1based", "atom_index_1based"]
    )
    checks["uma_unique_step_atom_keys"] = unique_key(
        uf, ["gaussian_step_1based", "atom_index_1based"]
    )

    checks["index_hashes_match_b3lyp"] = exact_string_mapping(
        index,
        b3s,
        "gaussian_step_1based",
        "source_structure_sha256",
        "structure_sha256",
    )
    checks["mace_hashes_match_b3lyp"] = exact_string_mapping(
        ms,
        b3s,
        "gaussian_step_1based",
        "source_structure_sha256",
        "structure_sha256",
    )
    checks["aim_member_hashes_match_b3lyp"] = exact_string_mapping(
        ais,
        b3s,
        "gaussian_step_1based",
        "source_structure_sha256",
        "structure_sha256",
    )
    checks["aim_ensemble_hashes_match_b3lyp"] = exact_string_mapping(
        aes,
        b3s,
        "gaussian_step_1based",
        "source_structure_sha256",
        "structure_sha256",
    )
    checks["uma_hashes_match_b3lyp"] = exact_string_mapping(
        us,
        b3s,
        "gaussian_step_1based",
        "source_structure_sha256",
        "structure_sha256",
    )

    checks["mace_atomic_alignment_matches_b3lyp"] = atom_alignment_matches(
        b3f, mf, "source_structure_sha256"
    )

    # Each member must independently align with B3LYP.
    aim_member_alignments = {}
    for model_alias, group in aif.groupby("model_alias", sort=True):
        aim_member_alignments[str(model_alias)] = atom_alignment_matches(
            b3f, group, "source_structure_sha256"
        )
    checks["all_aimnet_member_atomic_alignments_match_b3lyp"] = (
        set(aim_member_alignments) == AIMNET_MODELS
        and all(aim_member_alignments.values())
    )

    checks["aim_ensemble_atomic_alignment_matches_b3lyp"] = atom_alignment_matches(
        b3f, aea, "source_structure_sha256"
    )
    checks["uma_atomic_alignment_matches_b3lyp"] = atom_alignment_matches(
        b3f, uf, "source_structure_sha256"
    )

    checks["all_mace_structures_pass"] = set(ms["status"].astype(str)) == {"PASS"}
    checks["all_aimnet_member_structures_pass"] = (
        set(ais["status"].astype(str)) == {"PASS"}
    )
    checks["all_uma_structures_pass"] = set(us["status"].astype(str)) == {"PASS"}

    checks["aimnet_model_set_exact"] = set(ais["model_alias"].astype(str)) == AIMNET_MODELS
    checks["aimnet_four_members_per_step"] = bool(
        (ais.groupby("gaussian_step_1based")["model_alias"].nunique() == 4).all()
    )
    checks["aimnet_78_atoms_per_member_step"] = bool(
        (
            aif.groupby(["model_alias", "gaussian_step_1based"])[
                "atom_index_1based"
            ].nunique()
            == EXPECTED_ATOMS
        ).all()
    )
    checks["aimnet_ensemble_member_count_is_4"] = (
        set(pd.to_numeric(aes["member_count"]).astype(int)) == {4}
        and set(pd.to_numeric(aea["member_count"]).astype(int)) == {4}
    )

    checks["mace_no_explicit_charge"] = set(
        ms["explicit_charge_input"].astype(str).str.lower()
    ) == {"none"}
    checks["mace_intended_charge_plus_1"] = set(
        pd.to_numeric(ms["intended_system_charge_e"]).astype(int)
    ) == {1}
    checks["aimnet_charge_plus_1"] = (
        set(pd.to_numeric(ais["charge_e"]).astype(int)) == {1}
        and set(pd.to_numeric(aif["charge_e"]).astype(int)) == {1}
        and set(pd.to_numeric(aes["charge_e"]).astype(int)) == {1}
        and set(pd.to_numeric(aea["charge_e"]).astype(int)) == {1}
    )
    checks["uma_charge_plus_1_multiplicity_1"] = (
        set(pd.to_numeric(us["charge_e"]).astype(int)) == {1}
        and set(pd.to_numeric(us["spin_multiplicity"]).astype(int)) == {1}
        and set(pd.to_numeric(uf["charge_e"]).astype(int)) == {1}
        and set(pd.to_numeric(uf["spin_multiplicity"]).astype(int)) == {1}
    )

    checks["b3lyp_energies_finite"] = all_finite(b3s, ["scf_energy_eV"])
    checks["b3lyp_forces_finite"] = all_finite(
        b3f,
        [
            "Fx_corrected_standard_eV_per_A",
            "Fy_corrected_standard_eV_per_A",
            "Fz_corrected_standard_eV_per_A",
        ],
    )
    checks["mace_energies_finite"] = all_finite(ms, ["total_energy_eV"])
    checks["mace_forces_finite"] = all_finite(
        mf, ["force_x_eV_per_A", "force_y_eV_per_A", "force_z_eV_per_A"]
    )
    checks["aimnet_member_energies_finite"] = all_finite(ais, ["energy_eV"])
    checks["aimnet_member_forces_finite"] = all_finite(
        aif, ["force_x_eV_per_A", "force_y_eV_per_A", "force_z_eV_per_A"]
    )
    checks["aimnet_ensemble_energies_finite"] = all_finite(
        aes, ["energy_mean_eV", "energy_std_population_eV"]
    )
    checks["aimnet_ensemble_forces_finite"] = all_finite(
        aea,
        [
            "force_mean_x_eV_per_A",
            "force_mean_y_eV_per_A",
            "force_mean_z_eV_per_A",
        ],
    )
    checks["uma_energies_finite"] = all_finite(us, ["total_energy_eV"])
    checks["uma_forces_finite"] = all_finite(
        uf, ["force_x_eV_per_A", "force_y_eV_per_A", "force_z_eV_per_A"]
    )

    overall_pass = all(checks.values())

    counts_rows = [
        {"dataset": "full_path_input_index", "rows": len(index), "expected": 244},
        {"dataset": "b3lyp_structures", "rows": len(b3s), "expected": 244},
        {"dataset": "b3lyp_atomic_forces", "rows": len(b3f), "expected": 19032},
        {"dataset": "mace_structures", "rows": len(ms), "expected": 244},
        {"dataset": "mace_atomic_forces", "rows": len(mf), "expected": 19032},
        {"dataset": "aimnet_member_structures", "rows": len(ais), "expected": 976},
        {"dataset": "aimnet_member_atomic_forces", "rows": len(aif), "expected": 76128},
        {"dataset": "aimnet_ensemble_structures", "rows": len(aes), "expected": 244},
        {"dataset": "aimnet_ensemble_atomic", "rows": len(aea), "expected": 19032},
        {"dataset": "uma_structures", "rows": len(us), "expected": 244},
        {"dataset": "uma_atomic_forces", "rows": len(uf), "expected": 19032},
    ]
    pd.DataFrame(counts_rows).to_csv(counts_path, index=False)

    summary = {
        "status": "PASS" if overall_pass else "FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": str(stage),
        "purpose": (
            "Exact step/hash/atom alignment validation before B3LYP-referenced "
            "energy and force benchmarking."
        ),
        "expected": {
            "steps": EXPECTED_STEPS,
            "atoms_per_structure": EXPECTED_ATOMS,
            "atomic_rows_per_single_model": EXPECTED_ATOMIC_ROWS,
            "aimnet_members": EXPECTED_AIMNET_MEMBERS,
            "aimnet_member_structure_rows": EXPECTED_AIMNET_MEMBER_STRUCTURE_ROWS,
            "aimnet_member_atomic_rows": EXPECTED_AIMNET_MEMBER_ATOMIC_ROWS,
        },
        "checks": checks,
        "aimnet_member_atomic_alignment": aim_member_alignments,
        "source_files": {
            name: {
                "path": str(path),
                "sha256": source_hashes[name],
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "scientific_scope": [
            "This QC establishes exact data alignment; it does not establish model accuracy.",
            "The 244 Gaussian geometries are sequentially correlated points from one unconverged path.",
            "MACE-OFF23 received no explicit total charge.",
            "AIMNet2 received total charge +1; ensemble spread is disagreement, not calibrated uncertainty.",
            "UMA/OMol received charge +1 and singlet multiplicity 1.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "CHE701P B3LYP-referenced benchmark alignment QC",
        "=" * 51,
        f"Status: {'PASS' if overall_pass else 'FAIL'}",
        "",
        "Purpose",
        "-------",
        "Confirm exact Gaussian-step, structure-hash, atom-index, element and",
        "molecular-group alignment before calculating any model error metric.",
        "",
        "Dataset counts",
        "--------------",
    ]
    for row in counts_rows:
        report_lines.append(
            f"{row['dataset']}: {row['rows']} (expected {row['expected']})"
        )

    report_lines.extend(["", "Mandatory checks", "----------------"])
    for name, value in checks.items():
        report_lines.append(f"{name}: {value}")

    report_lines.extend(
        [
            "",
            "AIMNet2 member atomic alignment",
            "-------------------------------",
        ]
    )
    for model_alias, value in sorted(aim_member_alignments.items()):
        report_lines.append(f"{model_alias}: {value}")

    report_lines.extend(
        [
            "",
            "Interpretation",
            "--------------",
        ]
    )
    if overall_pass:
        report_lines.extend(
            [
                "All reference and prediction tables are aligned exactly.",
                "It is now valid to calculate path-local B3LYP-referenced relative-energy",
                "and force deviations using Gaussian step and atom index as join keys.",
                "No accuracy conclusion is made by this QC stage.",
            ]
        )
    else:
        failed = [name for name, value in checks.items() if not value]
        report_lines.extend(
            [
                "Benchmarking must not proceed.",
                "Failed checks:",
                *[f"- {name}" for name in failed],
            ]
        )

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest_rows = []
    for output in (report_path, summary_path, counts_path):
        manifest_rows.append(
            {
                "filename": output.name,
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            }
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["filename", "bytes", "sha256"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(report_path.read_text(encoding="utf-8"))
    print(f"Summary JSON: {summary_path}")
    print(f"Counts CSV: {counts_path}")
    print(f"Manifest: {manifest_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        raise SystemExit(1)

