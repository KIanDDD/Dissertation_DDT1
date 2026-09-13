# Public analysis environment

These requirements support the public numerical checks and plotting commands. They use analysis versions recorded in [package_check.txt](../gaussian_analysis/package_check.txt).

Use a separate Python 3.11.15 environment. From the repository root:

```bash
python -m pip install -r environments/public_analysis/requirements.txt
```

A fresh Windows installation passed the public checks and all four plotting commands on 13 September 2026. [validation_2026-09-13.json](validation_2026-09-13.json) records the checks; [pip_freeze_2026-09-13.txt](pip_freeze_2026-09-13.txt) lists the installed packages, including PyYAML and jsonschema used for citation validation.

This environment is for analysis of the public data. Model inference, Gaussian and Desmond require the separate software described in [software versions](../../docs/software_versions.md).
