# Data

| Folder | Contents |
|---|---|
| [n16_xyz](n16_xyz/) | 100 fixed N16 structures used in the MD100 benchmark |
| [hydration](hydration/) | Hydration-shell counts and candidate cluster sizes |
| [inter_model](inter_model/) | Comparisons between MLFF predictions |
| [b3lyp_md100](b3lyp_md100/) | B3LYP energy rankings and Gaussian quality-control summaries |
| [b3lyp_benchmark](b3lyp_benchmark/) | Energy, force and recovery summaries for the 244-geometry path |
| [diagnostic_dynamics](diagnostic_dynamics/) | Short AIMNet2 NVT/NVE time series |

The MD100 benchmark compares fixed-geometry energy ordering. The separate 244-geometry benchmark follows one correlated, unconverged optimisation path.

The public tables support the numerical checks and plots. Recomputing atomic force errors and ensemble statistics requires the original Gaussian force table and model predictions. Raw Gaussian logs, proprietary trajectories and model weights are not included. Source archive checksums are recorded in [manifests](../manifests/).
