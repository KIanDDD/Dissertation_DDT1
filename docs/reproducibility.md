# Reproduction guide

## Public analysis

The [root README](../README.md) lists the numerical checks and four plotting commands. Their requirements are in [public_analysis](../environments/public_analysis/README.md); all commands passed in a fresh Windows Python 3.11.15 environment on 13 September 2026.

The public absolute energies allow independent recalculation of MD100 ranks, minima and top-k overlap. Hydration and inter-model plots use the included tables. The path force and recovery tables contain aggregate results; reconstruction from atomic predictions requires the private corrected Gaussian force table and member/model outputs.

The [figure archive](../figures/final_dissertation/) contains the recovered dissertation images. Regenerated plots can differ in typography, layout and panel arrangement.

## Calculation workflow

Preparation and molecular dynamics used Schrödinger 2023-4 (Maestro, LigPrep, Epik and Desmond), OPLS4 and SPC water. Hydration analysis covered 9,982 readable frames before selection of the N16 configurations. Each finite, non-periodic cluster contains protonated tacrine and 16 waters: 78 atoms, charge +1, without chloride. The selected waters are a controlled subset of the 5.325 Å operational hydration envelope, whose modal occupancy was 48 waters.

The primary benchmark compares energy ordering for 100 identical fixed configurations. Dr Devis Di Tommaso supplied the B3LYP/6-31G(d) calculations run on QMUL Apocrita. Kian performed the MLFF calculations, processing, alignment and benchmarking.

MACE-OFF23-medium was evaluated without an explicit molecular charge input, outside its stated neutral-system scope. AIMNet2 received charge +1 for each of its four members. UMA used `uma-s-1p2`, task `omol`, charge +1 and multiplicity 1.

The secondary benchmark uses 244 sequential geometries from an unconverged Gaussian optimisation. Step 227 is the lowest sampled B3LYP geometry, rather than a confirmed optimised minimum. Force agreement is substantially weaker than energy-ordering agreement. B3LYP serves as a computational reference. Recovery analysis is retrospective and does not establish calibrated uncertainty or prospective active-learning performance. The timings lack a matched B3LYP measurement and therefore do not establish a general speedup.

## Additional inputs

The Gaussian extraction and path benchmark scripts require the original analysis folder layout and private inputs. Verification in a copied folder reproduced the extraction, force rotation, alignment and all ten public benchmark tables. This did not rerun Gaussian or pretrained-model inference. MACE's MD100 runner and the trajectory readers also retain their original relative paths; see the [script inventory](script_inventory.md).

Raw Gaussian logs and checkpoints, proprietary simulation files and model weights are not distributed here. Model inference requires separate model access and the [recorded software environments](software_versions.md).

## Diagnostic dynamics

The short AIMNet2 NVE run starts from the final NVT coordinates and momenta. Both recorded time series include two initial rows; the reported NVT mean of 340.8 K includes them. The velocity-initialisation seed does not fully determine the thermostat random stream, so a new run can produce a different trajectory. These short finite-cluster runs assess numerical behaviour rather than production-dynamics accuracy. Details are in [diagnostic_dynamics](../data/diagnostic_dynamics/README.md).

## Versions and verification

The PDF cites the fixed [v1.1.0 release](https://github.com/KIanDDD/Dissertation_DDT1/releases/tag/v1.1.0), published on 15 August 2026. The main branch contains subsequent maintenance. [CITATION.cff](../CITATION.cff) points to the cited release.

[Validation results](../manifests/MAINTENANCE_AUDIT_2026-09-13.txt) identify the tested commits. [File verification](../manifests/README.md) checks the current checkout against its manifests. The original software used for every submitted figure is not fully documented; the available source information is in [figure provenance](figure_provenance.md).
