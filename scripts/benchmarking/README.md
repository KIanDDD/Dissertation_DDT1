# Benchmarking

`01_validate_benchmark_alignment.py STAGE` validates the private 244-path stage. `02_run_locked_benchmark.py STAGE` computes relative-energy, force, disagreement, recovery and observed timing tables after alignment passes. Use a copied stage with empty derived-output folders. The scripts refuse ordinary replacement; do not use --overwrite on frozen originals.

The public aggregate force tables are insufficient to recompute errors from raw vectors. Retained corrected Gaussian forces and model atomic predictions are required. The frozen recovery table stores nominal random fractions; final Figure 11 uses actual selected counts 13/244, 25/244, 49/244. The zero-force comparator is recorded separately.

`analyse_chemical_force_groups.py` is the earlier two-model MD100 force-disagreement analysis and consumes its own aligned atomic schema. It does not calculate B3LYP-referenced path force errors.
