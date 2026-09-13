# Tacrine microhydration: dissertation reproducibility

Code, selected derived data and figure-source records supporting Kian Davarpanah's MSc dissertation, *Accelerating Molecular Simulations with MLFFs* (CHE701P, MSc Artificial Intelligence for Drug Discovery, Queen Mary University of London, 2026).

## Evidence and authorship

The primary benchmark compares electronic-energy ordering across 100 identical, fixed-geometry MD-derived N16 configurations using B3LYP/6-31G(d), MACE-OFF23-medium, a four-member AIMNet2 ensemble and UMA/OMol. N16 is the finite non-periodic +1 cluster [tacrineH]+(H2O)16: 78 atoms, with chloride omitted. The 16 nearest waters are a controlled subset, not the complete 5.325 Å operational hydration envelope, whose modal occupancy is 48 waters.

The secondary benchmark uses 244 sequential geometries from one unconverged Gaussian optimisation path. Step 227 is its lowest sampled B3LYP geometry, not a confirmed optimised minimum. Path-local force agreement is much weaker than energetic-ordering agreement. B3LYP is a computational reference, not experimental or exact truth.

Dr Devis Di Tommaso generated/supplied the Gaussian B3LYP calculations on QMUL Apocrita. Kian performed structure processing, pretrained-model workflows, QC/alignment, benchmarking, statistical analysis, interpretation and reproducibility work.

MACE was evaluated without explicit molecular charge/multiplicity input, outside its stated neutral-system applicability boundary. AIMNet2 received charge +1 for all four members. UMA used uma-s-1p2 with task omol, charge +1 and multiplicity 1. The short AIMNet2 NVT/NVE trajectory is a finite-cluster diagnostic. Disagreement-based recovery is retrospective, not calibrated uncertainty or prospective active learning. Timings have no matched B3LYP denominator and do not establish a universal acceleration factor.

## Contents

- `data/n16_xyz/`: 100 canonical MD-derived configurations.
- `data/hydration/`: operational-envelope, occupancy and candidate-N tables.
- `data/inter_model/`: MD100 model-model comparisons.
- `data/b3lyp_md100/`: primary energetic-ordering benchmark and derived Gaussian QC.
- `data/b3lyp_benchmark/`: secondary path-local benchmark tables.
- `data/diagnostic_dynamics/`: retained NVT/NVE time series and provenance notes.
- `figures/final_dissertation/`: recovered submitted artwork and figure-source manifest. Figure 2 is third-party material and is linked, not redistributed.
- `scripts/`: original-stage workflows and public derived-data plot reproductions. Some retained filenames use draft figure numbers; the authoritative mapping is in `docs/script_inventory.md`.
- `environments/`: retained package/model records plus a separately labelled convenience analysis environment.
- `manifests/`: current public-byte integrity records and clearly distinguished historical private-stage evidence.

## Public numerical reproduction

Use the recorded Gaussian-analysis environment, or the separately labelled convenience environment under `environments/public_analysis/`. From the repository root:

```bash
python -m compileall -q scripts
python scripts/validation/check_public_results.py
python scripts/n16_extraction/qc_N16_xyz.py --input-dir data/n16_xyz
python scripts/figures/make_Figure7_MD100_B3LYP_vs_MLFF_ranks.py --outdir reproduced_figures
python scripts/figures/make_FINAL_Figures7_8_intermodel_repo.py
python scripts/figures/make_FINAL_Section3_2_Figures_9_10_PERFECT.py
python scripts/figures/make_FINAL_Figure11_retrospective_recovery_repo.py
```

These regenerate plots from frozen public numerical inputs. They do not all reproduce the submitted typography, panel arrangement or artwork exactly. Use the recovered assets and provenance map for exact dissertation correspondence. The first rank command writes to `reproduced_figures/`; the other figure commands write ignored output files beside their scripts.

## Reproduction boundaries

Public tables support numerical inspection and selected plot regeneration. Recomputing path-local force errors, ensemble disagreement and recovery from atomic predictions requires retained private-stage inputs; aggregate tables alone do not independently reconstruct every statistic. Full inference also requires third-party model access and compatible environments. Proprietary Schrödinger/Desmond material, raw Gaussian files/checkpoints, weights/caches and complete private archives are excluded. See `docs/reproducibility.md`, `docs/software_versions.md` and `docs/figure_provenance.md`.

## Historical release and integrity

The dissertation cites `v1.1.0`. Its tag and release are preserved unchanged. The repository URL remains https://github.com/KIanDDD/Dissertation_DDT1. Maintenance changes concern reproducibility and documentation; they do not replace the submitted scientific record. The v1.1.0 release checksum list describes Windows working-tree bytes, including two mixed-line-ending files; see `manifests/README.md` for the exact distinction from current canonical bytes.

Verify the current checkout with `python scripts/validation/verify_release.py`. Fresh Windows installation results are recorded in `environments/public_analysis/validation_2026-09-13.json`; final clean-clone results are in `manifests/PUBLIC_RELEASE_AUDIT_v1.1.1.txt`. This maintenance branch is awaiting review. Its citation publication-date placeholder must be resolved and checksums regenerated before a release is approved.

## Citation and rights

Use `CITATION.cff`. No open-source licence is granted; all rights are reserved unless stated otherwise. Third-party software, models and literature figures retain their own rights and terms.
