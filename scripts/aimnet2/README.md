# Aimnet2

The MD100 batch runner evaluates one member at a time; the final ensemble used aimnet2 (member 0) and aimnet2-wb97m-d3_1, _2 and _3 with total charge +1. `analyse_aimnet2_ensemble.py` combines the four outputs. The 244-path runner separately validates all 244 geometries and reports population ensemble spread.

`run_aimnet2_diagnostic_md.py` performs Langevin NVT only. The separate retained `run_aimnet2_nve_energy_test.py` continues the last NVT coordinates and momenta using Velocity-Verlet. Original time series contain a duplicate initial diagnostic record; preserve that fact and the reported 340.8 K mean. The NVT seed does not fully fix the Langevin random stream, so exact trajectory regeneration is not promised.

Use the scripts' argument help and fresh output directories. Inference needs the AIMNet2 model files and recorded environment. No diagnostic validates production MD or calibrated uncertainty.
