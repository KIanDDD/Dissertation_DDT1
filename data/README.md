# Data availability

This repository contains selected author-generated derived data and the 100 fixed-composition N16 XYZ structures needed to inspect or regenerate the reported public comparison and figure workflows.

## Included

- `n16_xyz/`: 100 MD-derived `[tacrineH]+(H2O)16` structures (78 atoms each).
- `hydration/`: hydration-distance, hydration-occupancy, candidate-N distance, selected-frame and N16 manifest tables.
- `inter_model/`: aligned MACE-OFF23/AIMNet2/UMA comparison tables and observed inference timing records for the 100-structure dataset.
- `b3lyp_md100/`: derived QC, ranking and summary records for the same-configuration fixed-geometry B3LYP/6-31G(d) energetic-ordering benchmark across the 100 MD-derived N16 structures.
- `b3lyp_benchmark/`: selected frozen derived energy, force, disagreement and prioritisation tables for the separate 244-geometry path-local benchmark.

## Not redistributed

- proprietary Schrödinger/Maestro/Desmond project files and the complete trajectory;
- supervisor-supplied Gaussian input, output and checkpoint files;
- third-party pretrained model weights or caches;
- Hugging Face account/cache records, credentials or tokens;
- large frozen private project archives.

The public derived data support reproduction of the reported analyses and quantitative figures. A complete from-raw-data rerun additionally requires authorised access to the corresponding proprietary, private or supervisor-supplied source material.
