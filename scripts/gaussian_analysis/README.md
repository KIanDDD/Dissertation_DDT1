# Gaussian Analysis

These scripts process the separate, unconverged 244-geometry Gaussian optimisation path. They require an explicit private STAGE path; raw supervisor-supplied Gaussian files are excluded.

`02_extract_gaussian_path_corrected_v3.py STAGE` verifies the locked source-log hash, parses energies and Standard orientations, reconstructs the original Cartesian frame, and rotates forces using independently checked rigid-body rotations. It retains 244 evaluated geometries and distinguishes the unmatched 245th orientation.

`04_prepare_full_gaussian_path_inputs.py STAGE` creates geometry-only XYZ inputs with hash/order and read-back checks. Run in a copied stage; --overwrite deliberately replaces generated files. These scripts do not run Gaussian calculations and are not the private MD100 parser.
