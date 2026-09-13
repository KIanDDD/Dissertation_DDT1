# Software environments

For the public data checks and plots, start with [public_analysis](public_analysis/README.md). Its requirements were tested in a fresh Windows Python 3.11.15 environment on 13 September 2026.

The other folders contain records from the calculation and analysis environments:

| Folder | Purpose |
|---|---|
| [mace](mace/) | MACE-OFF23-medium inference |
| [aimnet2](aimnet2/) | AIMNet2 ensemble inference and diagnostic dynamics |
| [uma](uma/) | UMA/OMol inference |
| [gaussian_analysis](gaussian_analysis/) | Gaussian processing and benchmark analysis |
| [rdkit_figures](rdkit_figures/) | Available environment information for molecular drawing scripts |

The retained records vary by workflow; some include an environment YAML, while others contain package lists and version checks. The software used to produce every original figure could not be identified from the available records.

See [software versions](../docs/software_versions.md) for the full calculation stack, including Schrödinger 2023-4 and Gaussian 16 A.03.
