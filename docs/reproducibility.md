# Reproduction guide

## Public inputs and tests

Run the commands in the root README using Python 3.11.15 and `environments/public_analysis/requirements.txt`. The four advertised plotting commands, compilation, QC and public numerical assertions passed in a fresh isolated Windows installation on 2026-09-13. They regenerate repository variants from public tables. Exact recovered submitted artwork is separately identified in `figures/final_dissertation/FIGURE_MANIFEST.csv`.

MD100 ranks, minima and top-k overlap can be recalculated directly from public absolute energies. Hydration and inter-model plots use public derived tables. The public path force/recovery aggregates are insufficient to independently reconstruct every underlying atomic prediction or ensemble statistic. A full benchmark rerun requires the private corrected Gaussian force table and member/model predictions.

## Full retained workflow

Preparation and production MD used Schrödinger 2023-4 (Maestro, LigPrep, Epik, Desmond), OPLS4 and SPC water. Hydration analysis of 9982 readable frames preceded deterministic N16 extraction. Model inference used 100 fixed configurations. Dr Devis Di Tommaso supplied the Gaussian B3LYP/6-31G(d) calculations on QMUL Apocrita. Kian performed downstream processing, QC/alignment, benchmarking and interpretation.

The separate 244-path Gaussian extraction and benchmark scripts require an explicit copied private stage. They were rerun during the audit: extraction, force-frame rotation, alignment and all ten public benchmark tables passed. No new electronic-structure calculations or full pretrained-model inference were performed. Avoid --overwrite on frozen originals. MACE's MD100 runner and the raw trajectory scripts retain original-stage path assumptions; they are not repository-root commands.

NVT/NVE diagnostic continuity was checked using retained trajectory coordinates and momenta. The time series contain duplicate initial diagnostic rows, and the reported 340.8 K NVT mean includes them. The thermostat random stream is not fully controlled by the velocity-initialisation seed. Preserve the historical trajectory and summaries; a new stochastic trajectory is not evidence of an exact rerun.

## Environment and rights boundaries

Retained package records differ between workflows. The convenience analysis environment is newly assembled from recorded versions; its fresh Windows installation passed the recorded public tests. RDKit environments were detected, but exact originating figure environments and a CairoSVG historical version were not established. The final molecular schematic is an embedded EMF. Psi4 MD40 was exploratory and is not a production reference.

Raw/proprietary inputs, checkpoint files, model weights/caches, credentials and private archives remain excluded. No open-source licence is added. Third-party figures retain separate rights. Preserve v1.1.0 and its repository URL; regenerate current-tree manifests only at the end of approved maintenance.


`python scripts/validation/check_public_results.py` independently asserts the public MD100 energy ranks, minima, overlaps and XYZ hashes, QC summaries, hydration mode, path energy metrics, force/recovery identities and retained diagnostic summaries. Full atomic force/recovery recomputation still requires private inputs. `python scripts/validation/verify_release.py` checks exact current bytes after final manifest generation.

Fresh-install results are recorded in `environments/public_analysis/validation_2026-09-13.json`; final clean-clone and integrity results are recorded in `manifests/MAINTENANCE_AUDIT_2026-09-13.txt`. This is subsequent maintenance of the repository, without a new numbered release. `CITATION.cff` identifies the unchanged dissertation-cited v1.1.0 release as the preferred citation, using its verified publication date, 2026-08-15. The maintained main branch and the fixed release are distinct snapshots.

**UNRESOLVED — EVIDENCE REQUIRED:** exact historical figure environments require original asset-linked package/export records. The recovered document artwork remains independently identified and hashed.
