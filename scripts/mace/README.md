# Mace

`run_mace_off_batch_100.py` preserves the original MD100 stage workflow and relative directory assumptions. Its current location in this repository does not reconstruct the original input path. Use an explicitly adapted input/output interface only after validation, or reproduce the original stage layout in a separate working copy.

`run_mace_gaussian_full_244.py` exposes explicit input directory, input index, corrected reference and output arguments for the secondary path. Both workflows use MACE-OFF23-medium at fixed geometry. No total molecular charge or multiplicity was explicitly supplied; application to +1 N16 lies outside the stated neutral-system domain.

Model access and the recorded MACE environment are required. No weights are redistributed. The public plotting commands consume retained numerical outputs and do not run inference.
