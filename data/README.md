# Data availability

This repository contains selected **author-generated derived data** and the 100 fixed-composition N16 XYZ structures needed to inspect or regenerate the reported public comparison/figure workflows.

## Included

- `n16_xyz/`: 100 MD-derived `[tacrineH]+(H2O)16` structures (78 atoms each).
- `hydration/`: RDF/minimum-distance, hydration-occupancy, candidate-N distance, selected-frame and N16 manifest tables.
- `inter_model/`: aligned MACE-OFF23/AIMNet2/UMA comparison tables and observed inference timing summary for the 100-structure dataset.
- `b3lyp_benchmark/`: selected frozen derived energy/force/disagreement/prioritisation tables for the 244-geometry path-local benchmark.

## Not redistributed

- proprietary Schrödinger/Maestro/Desmond project files and the complete trajectory;
- supervisor-supplied Gaussian input/output/checkpoint files;
- third-party pretrained model weights or caches;
- Hugging Face account/cache records, credentials or tokens;
- large frozen private project archives.

The absence of these raw/private inputs means that the repository supports direct reproduction of the public derived-data analyses and figures, while a complete from-raw-data rerun requires authorised access to the corresponding private/proprietary source material.
