# Software versions and execution context

| Software | Version evidence | Role | Boundary |
|---|---|---|---|
| Schrödinger Maestro/LigPrep/Epik/Desmond | 2023-4 | Preparation and explicit-water MD | Proprietary; local project/trajectory required |
| Gaussian | 16 Revision A.03 | Supervisor B3LYP MD100 and unconverged path; QMUL Apocrita | Raw logs verified locally; not redistributed |
| ASE | 3.29.0 (main workflow records) | Structures, calculators, dynamics | Retained dependency; RDKit env currently 3.28.0 is separate |
| MACE / PyTorch | mace-torch 0.3.16 / torch 2.13.0+cpu | MACE-OFF23-medium | Model hash/file records retained; no weights |
| AIMNet2 | aimnet 0.2.0; four specified aliases | Charge+1 ensemble and diagnostic dynamics | CPU eager path; TORCH_COMPILE_DISABLE=1 |
| FAIR-Chem / PyTorch | fairchem-core 2.21.0 / torch 2.8.0 | uma-s-1p2;omol;+1;spin1 | Third-party model access required |
| Analysis | Python 3.11.15; NumPy 2.4.6; pandas 3.0.5; SciPy 1.17.1; Matplotlib 3.11.1 | Gaussian/data/plotting | Versions retained and existing environment successfully tested |
| Other analysis | cclib 1.8.1;openpyxl 3.1.5 | Gaussian parse and tabular tooling | Retained package checks |
| RDKit | Current kian-rdkit 2026.03.3;my-rdkit-env 2025.09.2 | Figure candidates; exact historical producer unproved | Current state distinct from original snapshot |
| CairoSVG | Not established | Historical alternative molecular export | No located installation/export; do not invent version |
| Psi4 | 1.11 | Exploratory local MD40 fallback,31 of 40 frozen | Omit from final production stack |
| Microsoft Word | PDF metadata: Word 2019 | Submitted document composition/export | Embedded media and PDF have primary figure provenance |

Package versions identify retained records. The public-analysis pins were also installed and tested in a fresh isolated Windows Python 3.11.15 environment on 2026-09-13; see `environments/public_analysis/validation_2026-09-13.json`. This current test does not recreate the historical production environments. No originating drawing application was proved for the final Figure 3 EMF. Current Conda inventories must not be presented as submission-time snapshots.


**UNRESOLVED — EVIDENCE REQUIRED:** originating figure package/export records and a historical CairoSVG version are not established. Resolve them with original dated provenance, not current package discovery. Model aliases do not substitute for unavailable original per-member checkpoint hashes.
