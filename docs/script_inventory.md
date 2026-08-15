# Script inventory

Only retained/final workflow scripts relevant to the dissertation are included; superseded exploratory copies are intentionally omitted.

| Workflow | Script | Purpose |
|---|---|---|
| Hydration analysis | `scripts/hydration_analysis/calculate_rdf_water_counts.py` | Calculate hydration-distance/RDF and water-count records |
| Hydration analysis | `scripts/hydration_analysis/analyse_microhydration_candidates.py` | Compare candidate fixed water counts used in the N16 decision |
| N16 construction | `scripts/n16_extraction/extract_N16_clusters.py` | Extract the 16 nearest waters for selected trajectory frames |
| N16 QC | `scripts/n16_extraction/qc_N16_xyz.py` | Check composition, atom count and exported N16 structures |
| MACE 100-set | `scripts/mace/run_mace_off_batch_100.py` | Fixed-geometry MACE-OFF23 inference on 100 N16 structures |
| AIMNet2 100-set | `scripts/aimnet2/run_aimnet2_batch_100.py` | AIMNet2 fixed-geometry inference on 100 N16 structures |
| AIMNet2 ensemble | `scripts/aimnet2/analyse_aimnet2_ensemble.py` | Combine/analyse member-level predictions |
| AIMNet2 dynamics | `scripts/aimnet2/run_aimnet2_diagnostic_md.py` | Short finite-cluster NVT/NVE diagnostic |
| UMA 100-set | `scripts/uma/uma_batch_100.py` | UMA/OMol fixed-geometry inference on 100 N16 structures |
| Three-model comparison | `scripts/uma/compare_mace_aimnet2_uma.py` | Construct aligned three-model comparison records |
| Gaussian path extraction | `scripts/gaussian_analysis/02_extract_gaussian_path_corrected_v3.py` | Extract/orientation-correct the Gaussian reference path; private raw input required |
| Gaussian path inputs | `scripts/gaussian_analysis/04_prepare_full_gaussian_path_inputs.py` | Prepare geometry-only inputs for 244 fixed-geometry evaluations |
| MACE path | `scripts/mace/run_mace_gaussian_full_244.py` | MACE inference on corrected 244-geometry path |
| AIMNet2 path | `scripts/aimnet2/run_aimnet2_gaussian_full_244_ensemble.py` | Four-member AIMNet2 inference on the 244 geometries |
| UMA path | `scripts/uma/run_uma_gaussian_full_244.py` | UMA/OMol inference on the 244 geometries |
| Benchmark alignment QC | `scripts/benchmarking/01_validate_benchmark_alignment.py` | Verify step/structure/atom alignment |
| Locked benchmark | `scripts/benchmarking/02_run_locked_benchmark.py` | Generate the frozen path-local benchmark outputs |
| Chemical force analysis | `scripts/benchmarking/analyse_chemical_force_groups.py` | Chemically resolved force disagreement/error analysis |
| Final MD100 Figure 7 | `scripts/figures/make_Figure7_MD100_B3LYP_vs_MLFF_ranks.py` | Reproduce the final B3LYP-versus-MLFF energetic-rank figure from public derived data and verify the locked rank statistics |
| Secondary inter-model figures | `scripts/figures/make_FINAL_Figures7_8_intermodel_repo.py` | Regenerate the secondary inter-model energy/force comparison; historical script filename retained |
| Figures 9-10 | `scripts/figures/make_FINAL_Section3_2_Figures_9_10_PERFECT.py` | Regenerate B3LYP path-local energy/force Results figures from public derived tables |
| Figure 11 | `scripts/figures/make_FINAL_Figure11_retrospective_recovery_repo.py` | Regenerate retrospective recovery figure from public derived table |

Additional self-contained/structural dissertation plotting scripts are retained under `scripts/figures/` where available, but their presence is not required to reproduce the central quantitative benchmark conclusions.

The raw B3LYP100 parsing/QC script is retained in the private project because it consumes the supervisor-supplied Gaussian archive. Public derived QC, ranking and summary records are provided under `data/b3lyp_md100/`.
