# Script inventory

The submitted PDF and `docs/figure_provenance.md` define final figure numbering. Historical filenames retain their provenance. The table records script roles and interfaces. Original-stage model/trajectory commands require their documented inputs and environments.

| Script | Role, interface and limitations |
|---|---|
| scripts/aimnet2/analyse_aimnet2_ensemble.py | Four-member MD100 analysis; sample energy SD (ddof=1), population RMS force spread. Inputs are private model predictions. This is not calibrated uncertainty. |
| scripts/aimnet2/run_aimnet2_batch_100.py | Single-member inference; run once per declared member with charge +1. Third-party model access is required. No full inference rerun was performed in the audit. |
| scripts/aimnet2/run_aimnet2_diagnostic_md.py | Historical Langevin NVT diagnostic only. The seed fixes initial velocities, not an explicitly supplied thermostat random stream. Historical duplicate step 0 rows and summaries are retained. |
| scripts/aimnet2/run_aimnet2_gaussian_full_244_ensemble.py | Four-member path inference with charge +1, ordered aliases, hash/shape checks and population spread. Per-member immutable checkpoint hashes are not all established. |
| scripts/aimnet2/run_aimnet2_nve_energy_test.py | Unchanged retained NVE runner. Continues final NVT coordinates and momenta using Velocity-Verlet; finite-cluster numerical diagnostic only. |
| scripts/benchmarking/01_validate_benchmark_alignment.py | Private 244-path alignment QC. Checks the retained input alignment; the recorded IDs are integral. Verification in a copied analysis folder passed. |
| scripts/benchmarking/02_run_locked_benchmark.py | Private locked path analysis; all ten public tables reproduced in audit. Unit strings are repaired. Frozen recovery uses nominal budget fractions. The zero-force comparator is a separate derived record. |
| scripts/benchmarking/analyse_chemical_force_groups.py | Earlier two-model MD100 inter-model force analysis; requires its private aligned schema. It is not a path-local B3LYP force-error calculation. |
| scripts/benchmarking/derive_zero_force_baseline.py | Verifies the private corrected-force source hash and recomputes the path-local zero-force baseline. Public derived record is separately labelled. |
| scripts/figures/make_FINAL_Figure11_retrospective_recovery_repo.py | Public numerical variant for final Figure11; actual-count random line. Palette, legend and layout differ from archived submitted artwork. |
| scripts/figures/make_FINAL_Figures7_8_intermodel_repo.py | Historical parity output labelled Figure7 is not final Figure7. Its one-panel Figure8 is a public variant; submitted Figure8 has element and nitrogen-site panels. |
| scripts/figures/make_FINAL_Section3_2_Figures_9_10_PERFECT.py | Public Figure9/10 numerical reproduction with repaired Å labels. Embedded submitted aspect ratio/typography remain separately archived. |
| scripts/figures/make_Figure7_MD100_B3LYP_vs_MLFF_ranks.py | Public final-Figure7 rank data reproduction with strict integral ranks and minima assertions. Plot styling differs from the submitted asset. |
| scripts/figures/make_figure3_FINAL_NO_OVERFLOW.py | Earlier workflow schematic with draft numbering, superseded by submitted Figure4 and its MD100 B3LYP branch. The archived submitted image contains the final workflow. |
| scripts/figures/make_intro_figure1_exact.py | Milestones schematic candidate; its name does not establish exact final typography. Archived SVG and labelled PDF clip preserve the submitted evidence. |
| scripts/figures/make_intro_figure2_tacrine_tacrinium_FINAL.py | Earlier molecular alternative, not the proved generator of submitted Figure3 EMF. It imports CairoSVG, whose historical installation/version is unresolved. Not an advertised final reproduction command. |
| scripts/gaussian_analysis/02_extract_gaussian_path_corrected_v3.py | Final private Gaussian extraction and force rotation; copied-stage replay passed. Two message strings repaired. Scientific extraction/alignment logic is retained. |
| scripts/gaussian_analysis/04_prepare_full_gaussian_path_inputs.py | Final private 244-geometry input preparation with source/hash/shape checks. Repaired unit message. Use a fresh copied stage; --overwrite must not target original frozen outputs. |
| scripts/hydration_analysis/analyse_microhydration_candidates.py | Original-stage candidate-N analysis with private relative-layout assumptions. Early decision prose is historical; final N16 remains a controlled subset. |
| scripts/hydration_analysis/calculate_rdf_water_counts.py | Original proprietary trajectory reader; orthorhombic minimum-image convention matches recorded system. Earlier peak search reports 4.475 Å; final plot peak 4.875 Å, with the same 5.325 Å operational cutoff. |
| scripts/mace/run_mace_gaussian_full_244.py | Final path inference with input hash/shape checks; no explicit molecular-charge input. Retain MACE neutral-system applicability caveat. |
| scripts/mace/run_mace_off_batch_100.py | Original-layout MD100 inference runner. Relocated relative paths are not a repository-root reproduction interface. Use retained original stage; no CLI refactor is claimed. |
| scripts/n16_extraction/extract_N16_clusters.py | Original proprietary trajectory extraction. Preserves nearest-water selection, atom order and finite-cluster construction; original working-directory layout required. |
| scripts/n16_extraction/qc_N16_xyz.py | Public --input-dir option; requires 100 structures; excludes only self-distances and reports missing/coincident inputs. Actual 100 canonical structures pass. |
| scripts/uma/compare_mace_aimnet2_uma.py | Three-model MD100 comparison with one-to-one alignment and hash/order checks. Energy-only N/A force fields are meaningful. |
| scripts/uma/run_uma_gaussian_full_244.py | Final path inference; uma-s-1p2, omol, CPU, seed 701, charge +1 and spin 1. Input hash/order and shape checks retained. |
| scripts/uma/uma_batch_100.py | MD100 inference. Use fresh output directories. Historical --resume does not fully validate hash/model/charge/spin and is not recommended for reproduction. |
| scripts/validation/check_public_results.py | Read-only public numerical assertions from absolute energies, ranks, hashes and derived tables. It cannot reconstruct private atomic predictions from aggregate tables. |
| scripts/validation/verify_release.py | Verifies file coverage, sizes and exact checksums. --write-from-index generates the two current manifests from staged Git files. |
