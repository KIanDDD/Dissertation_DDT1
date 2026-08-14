# Reproduction guide

## Two levels of reproducibility

### 1. Public derived-data reproduction

The supplied public derived tables are sufficient to reproduce the central quantitative Results figures without access to raw proprietary files:

```bash
python scripts/figures/make_FINAL_Figures7_8_intermodel_repo.py
python scripts/figures/make_FINAL_Section3_2_Figures_9_10_PERFECT.py
python scripts/figures/make_FINAL_Figure11_retrospective_recovery_repo.py
```

The plotting scripts perform numerical/structural QC before creating figures.

### 2. Full from-raw-data workflow

The complete scientific workflow proceeded in this order:

1. Explicit-water Desmond trajectory generation/processing.
2. Hydration analysis and fixed-composition N16 construction/QC.
3. Fixed-geometry MACE-OFF23, AIMNet2 and UMA/OMol inference on the 100 MD-derived N16 structures.
4. Extraction and orientation correction of 244 sequential geometries/forces from the supervisor-supplied Gaussian reference output.
5. Fixed-geometry pretrained-model inference on those 244 geometries.
6. Exact alignment by Gaussian step, structure hash, atom count, element and atom index.
7. Path-local B3LYP-referenced energy/force benchmarking, chemically resolved analysis and retrospective disagreement prioritisation.
8. Dissertation figure generation from frozen derived tables.

Raw Schrödinger/trajectory material, raw supervisor-supplied Gaussian files and third-party model checkpoints are not redistributed. Scripts that require the private Gaussian benchmark stage no longer contain a machine-specific default path; supply the stage explicitly, for example:

```bash
python scripts/gaussian_analysis/02_extract_gaussian_path_corrected_v3.py "PATH_TO/12_gaussian_path_benchmark"
python scripts/gaussian_analysis/04_prepare_full_gaussian_path_inputs.py "PATH_TO/12_gaussian_path_benchmark"
python scripts/benchmarking/01_validate_benchmark_alignment.py "PATH_TO/12_gaussian_path_benchmark"
python scripts/benchmarking/02_run_locked_benchmark.py "PATH_TO/12_gaussian_path_benchmark"
```

## Recorded environments

- `environments/mace/` — retained MACE environment/model verification records.
- `environments/aimnet2/` — Conda list, pip freeze and Python version records.
- `environments/uma/` — Conda list, pip freeze, Python version, package check and model provenance records.
- `environments/gaussian_analysis/` — environment YAML, Conda list, pip freeze, Python version and package check.

The records reflect what was retained during the project and are not artificially normalised into identical file sets. Machine-specific local paths have been removed from their public copies.

## Scientific limits

Reproducibility does not widen the evidential scope: the 100 MD-derived structures provide inter-model evidence only; the 244 Gaussian geometries are sequential and unconverged; N16 is a finite +1 cluster without chloride; diagnostic dynamics do not validate bulk-water dynamics; disagreement is not calibrated uncertainty; and unmatched timings do not define a universal speed-up factor.
