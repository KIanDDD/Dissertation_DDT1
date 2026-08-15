# Reproduction guide

## Two levels of reproducibility

### 1. Public derived-data reproduction

The public derived tables are sufficient to reproduce the principal quantitative analyses without access to raw proprietary or supervisor-supplied files.

The final same-configuration B3LYP energetic-rank figure can be regenerated with:

```bash
python scripts/figures/make_Figure7_MD100_B3LYP_vs_MLFF_ranks.py --outdir reproduced_figures
```

Other retained plotting scripts reproduce complementary inter-model, path-local and retrospective-prioritisation analyses:

```bash
python scripts/figures/make_FINAL_Figures7_8_intermodel_repo.py
python scripts/figures/make_FINAL_Section3_2_Figures_9_10_PERFECT.py
python scripts/figures/make_FINAL_Figure11_retrospective_recovery_repo.py
```

The older inter-model plotting script retains its historical filename. It supports the secondary inter-model analysis and does not define the final dissertation Figure 7.

### 2. Full from-raw-data workflow

The complete scientific workflow proceeded in this order:

1. Explicit-water Desmond trajectory generation and processing.
2. Hydration analysis and deterministic fixed-composition N16 construction/QC.
3. Fixed-geometry MACE-OFF23, AIMNet2 and UMA/OMol inference on the 100 MD-derived N16 structures.
4. Fixed-geometry B3LYP/6-31G(d) calculations on the same 100 structures.
5. Gaussian QC, exact structure alignment and B3LYP-versus-MLFF energetic-rank analysis.
6. Extraction and orientation correction of the separate 244-geometry Gaussian optimisation path.
7. Fixed-geometry pretrained-model inference on those 244 geometries.
8. Path-local B3LYP-referenced energy/force benchmarking, chemically resolved analysis and retrospective disagreement prioritisation.
9. Dissertation figure generation from frozen derived tables.

Raw Schrödinger/trajectory material, supervisor-supplied Gaussian files and third-party pretrained-model checkpoints are not redistributed.

The private raw B3LYP100 analysis workflow requires the original supervisor-supplied Gaussian archive. Public reproduction therefore begins from the derived QC and ranking records under `data/b3lyp_md100/`.

## Recorded environments

- `environments/mace/` — retained MACE environment/model verification records.
- `environments/aimnet2/` — Conda list, pip freeze and Python-version records.
- `environments/uma/` — Conda list, pip freeze, Python version, package check and model-provenance records.
- `environments/gaussian_analysis/` — environment YAML, Conda list, pip freeze, Python version and package check.

The records reflect what was retained during the project and are not artificially normalised into identical file sets. Machine-specific local paths have been removed from public copies.

## Scientific limits

Reproducibility does not widen the evidential scope. The 100 MD-derived structures provide a same-configuration fixed-geometry B3LYP energetic-ordering benchmark. This establishes correspondence in electronic-energy ordering within the sampled configuration space, not thermodynamic free energies or universal MLFF accuracy. The separate 244 Gaussian geometries are sequential and unconverged. N16 is a finite +1 cluster without chloride. Diagnostic dynamics do not validate bulk-water or long-timescale dynamics. Disagreement is not calibrated predictive uncertainty, and unmatched timings do not define a universal MLFF-to-DFT speed-up factor.
