# Environment records

The original production records are retained under `mace/` (che701p-mace), `aimnet2/` (che701p-aimnet), `uma/` (che701p-uma) and `gaussian_analysis/` (che701p-gaussian-analysis). Their file sets differ because capture differed between workflows. Recorded versions and model provenance are preserved; absent historical exports are not reconstructed and mislabelled as originals.

`public_analysis/` is a new convenience specification assembled from recorded analysis versions. Both the existing Gaussian-analysis environment and a fresh isolated Windows installation passed the public commands. The fresh installation and validation are recorded under `public_analysis/` with date 2026-09-13. This specification is not the single historical environment for all calculations.

`rdkit_figures/` documents the detected kian-rdkit and my-rdkit-env histories and the distinction between present package inventories and original figure-generation provenance. The originating environment for each exact submitted asset and a historical CairoSVG version remain **UNRESOLVED — EVIDENCE REQUIRED**. An original export/package record tied to the asset is needed. No synthetic historical YAML is supplied.

The che701p-psi4 MD40 pilot did not provide the final MD100 Gaussian benchmark and is omitted from the production stack. Schrödinger 2023-4 and Gaussian 16 A.03 are documented in `docs/software_versions.md`. The Conda marker file is not an environment; audit runtimes are not dissertation production environments.
