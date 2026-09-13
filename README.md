# Tacrine microhydration: dissertation reproducibility

Code, data and figures for Kian Davarpanah's MSc dissertation, *Accelerating Molecular Simulations with MLFFs* (CHE701P, MSc Artificial Intelligence for Drug Discovery, Queen Mary University of London, 2026).

The dissertation cites [release v1.1.0](https://github.com/KIanDDD/Dissertation_DDT1/releases/tag/v1.1.0). Its [source snapshot](https://github.com/KIanDDD/Dissertation_DDT1/tree/v1.1.0) is unchanged. The main branch includes later documentation, figure archives and reproducibility fixes.

## Study

The primary benchmark compares B3LYP/6-31G(d) energy rankings with MACE-OFF23-medium, a four-member AIMNet2 ensemble and UMA/OMol for 100 fixed configurations of protonated tacrine with 16 water molecules. A separate benchmark examines energies and forces along a 244-geometry, unconverged Gaussian optimisation path.

Dr Devis Di Tommaso supplied the Gaussian calculations run on QMUL Apocrita. Kian carried out the structure processing, MLFF calculations, analysis and interpretation. [Methods and limitations](docs/reproducibility.md) describe the model settings and scope of the comparisons.

## Repository contents

| Folder | Contents |
|---|---|
| [data](data/) | 100 N16 structures, hydration analysis, benchmark tables and diagnostic time series |
| [scripts](scripts/) | Data processing, model inference and plotting scripts |
| [figures/final_dissertation](figures/final_dissertation/) | Archived dissertation figures and their sources |
| [environments](environments/) | Recorded software environments and installation requirements for public analysis |
| [docs](docs/) | Reproduction guide, software versions and figure/script mapping |
| [manifests](manifests/) | File inventories, checksums and validation records |

## Reproduce the public analysis

Use Python 3.11.15 with the [public analysis requirements](environments/public_analysis/README.md). From the repository root:

```bash
python -m pip install -r environments/public_analysis/requirements.txt
python -m compileall -q scripts
python scripts/validation/check_public_results.py
python scripts/n16_extraction/qc_N16_xyz.py --input-dir data/n16_xyz
python scripts/figures/make_Figure7_MD100_B3LYP_vs_MLFF_ranks.py --outdir reproduced_figures
python scripts/figures/make_FINAL_Figures7_8_intermodel_repo.py
python scripts/figures/make_FINAL_Section3_2_Figures_9_10_PERFECT.py
python scripts/figures/make_FINAL_Figure11_retrospective_recovery_repo.py
```

These commands check the public data and regenerate plots. Plot styling can differ from the submitted figures; the [figure archive](figures/final_dissertation/) contains the recovered document images. The rank plot is written to `reproduced_figures/`; the other plots are written beside their scripts.

Full model calculations and reconstruction of atomic force statistics require additional inputs and model access. Raw Gaussian files, proprietary simulation files and model weights are not included. See the [reproduction guide](docs/reproducibility.md).

## File verification

Run `python scripts/validation/verify_release.py` to check the current files against their recorded checksums. The [manifest guide](manifests/README.md) explains the coverage, historical records and [validation results](manifests/MAINTENANCE_AUDIT_2026-09-13.txt).

## Citation and rights

[CITATION.cff](CITATION.cff) provides the citation for v1.1.0. All rights are reserved unless stated otherwise. Third-party software, models and literature figures retain their own licences and terms.
