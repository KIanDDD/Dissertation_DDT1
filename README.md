# Tacrine microhydration MLFF reproducibility code

Author-developed analysis and reproducibility code supporting the MSc dissertation **Accelerating Molecular Simulations with MLFFs** (CHE701P, MSc Artificial Intelligence for Drug Discovery, Queen Mary University of London, 2026).

## Scope

This repository documents the author-developed computational workflow used to evaluate pretrained machine-learned force fields for protonated tacrine microhydration. It contains selected derived data, fixed-composition N16 structures, analysis/inference scripts, software-environment records, integrity manifests and repository-adapted plotting scripts.

It is **not** a redistribution of the complete private computational project archive.

## Scientific scope and evidence boundaries

- N16 denotes the finite non-periodic `[tacrineH]+(H2O)16` cluster: 78 atoms, net charge +1. The chloride counterion from the original periodic molecular-dynamics system is omitted from these finite clusters.
- The 100 MD-derived N16 structures have no common quantum-mechanical labels. Results on this set quantify inter-model agreement/disagreement, not quantum-mechanical accuracy.
- The B3LYP benchmark is based on 244 sequential geometries extracted from one unconverged B3LYP/6-31G(d) Gaussian optimisation path. It is a path-local comparison, not an independent validation set.
- The short AIMNet2 trajectory is a finite-cluster diagnostic, not evidence for bulk aqueous behaviour or validated long-timescale dynamics.
- Recorded workflow timings used unmatched implementations/hardware conditions and are not a universal MLFF-to-DFT speed-up factor.
- Disagreement-based recovery was evaluated retrospectively using already-known B3LYP-referenced errors; it is not calibrated predictive uncertainty or prospective active learning.

## Repository structure

- `data/n16_xyz/` — 100 fixed-composition N16 XYZ structures used for the MD-derived comparison.
- `data/hydration/` — public derived hydration/N16-selection tables.
- `data/inter_model/` — aligned three-model derived comparison tables for the 100-structure dataset.
- `data/b3lyp_benchmark/` — selected frozen path-local B3LYP-referenced benchmark tables used in the dissertation.
- `scripts/hydration_analysis/` — hydration-envelope and candidate-N analysis.
- `scripts/n16_extraction/` — deterministic N16 extraction and structural QC.
- `scripts/mace/`, `scripts/aimnet2/`, `scripts/uma/` — pretrained-model inference/analysis scripts.
- `scripts/gaussian_analysis/` — Gaussian-path extraction/preparation scripts; these require the non-redistributed private stage path as an explicit argument.
- `scripts/benchmarking/` — alignment QC and locked benchmark scripts; these require the private benchmark stage for a full rerun.
- `scripts/figures/` — selected final/repository-adapted dissertation plotting scripts, including the quantitative Results figures reproducible from public derived tables.
- `environments/` — recorded Conda/Python/package information from the environments used during the project.
- `manifests/` — integrity records and final public-release SHA256 manifests.
- `docs/script_inventory.md` — script-to-purpose map.
- `docs/reproducibility.md` — reproduction guide and public/private input boundary.

## Quick reproduction from public derived data

The quantitative Results plots can be regenerated without the proprietary/private raw files once the relevant Python packages are installed:

```bash
python scripts/figures/make_FINAL_Figures7_8_intermodel_repo.py
python scripts/figures/make_FINAL_Section3_2_Figures_9_10_PERFECT.py
python scripts/figures/make_FINAL_Figure11_retrospective_recovery_repo.py
```

These scripts include checks on structure counts, step numbering and locked numerical values before plotting.

## Full workflow reproduction

A complete from-raw-data rerun additionally requires materials that cannot be redistributed here, including the Schrödinger/Desmond trajectory/project files, pretrained third-party model checkpoints/caches and the supervisor-supplied Gaussian files. The retained raw-stage scripts therefore require those inputs to be supplied separately. See `docs/reproducibility.md`.

## Computational environments

Separate local environments were used for MACE-OFF23 (`che701p-mace`), AIMNet2 (`che701p-aimnet`), UMA/OMol (`che701p-uma`) and Gaussian-path analysis (`che701p-gaussian-analysis`). Recorded package inventories are retained under `environments/`. Local machine paths have been removed from the public copies.

## Integrity

SHA256 hashes were used throughout the project for file/structure identity and frozen benchmark provenance. The final public package contains a path-portable `manifests/file_manifest.csv` and `manifests/RELEASE_SHA256SUMS.txt`, generated only after the public-release audit passes.

## Data and redistribution restrictions

This repository does not redistribute proprietary Schrödinger files, the complete Desmond trajectory, raw supervisor-supplied Gaussian files/checkpoints, pretrained model weights, Hugging Face caches/account records, credentials, or full private project archives.

## Citation

Citation metadata are provided in `CITATION.cff`.

## Dissertation

Kian Davarpanah, *Accelerating Molecular Simulations with MLFFs*, MSc Artificial Intelligence for Drug Discovery, Queen Mary University of London, 2026.

## Licence

No open-source licence is granted in this dissertation release. All rights are reserved unless stated otherwise. Third-party software and models remain subject to their own licences and terms.
