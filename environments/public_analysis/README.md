# Public analysis convenience environment

This is a new convenience specification assembled from versions actually retained in `../gaussian_analysis/package_check.txt`. It is not a historical environment snapshot. It contains only the main dependencies needed for the public derived-data plotting commands; it does not run MACE, AIMNet2, UMA, Gaussian or Desmond.

Use Python 3.11.15 and install `requirements.txt` into a separate environment. On 2026-09-13 a fresh isolated Windows environment installed these pins from the package registry without reusing the existing analysis environment. Dependency consistency, compilation, the 100-XYZ QC, public numerical assertions and all four advertised plotting commands passed.


`validation_2026-09-13.json` records the performed checks and their scope. `pip_freeze_2026-09-13.txt` records the freshly installed packages, including PyYAML and jsonschema used for citation validation. These are current validation records, not historical production snapshots. This validation establishes the public derived-data route on Windows; it does not establish full model inference or another operating system.
