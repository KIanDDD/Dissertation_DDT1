# Figure-generation scripts

This directory contains selected final and repository-adapted dissertation plotting scripts.

The principal same-configuration energetic-ordering figure is reproduced directly from public derived data using:

```bash
python scripts/figures/make_Figure7_MD100_B3LYP_vs_MLFF_ranks.py --outdir reproduced_figures
```

The script verifies the 100-structure dataset, the B3LYP and shared pretrained-model minima, and the locked Spearman and Kendall tau-b statistics before plotting.

The remaining scripts reproduce complementary inter-model, path-local and retrospective-prioritisation analyses from public derived tables. Some filenames retain historical figure numbering from earlier dissertation stages; their presence is retained as plotting provenance and does not redefine the final dissertation numbering.

Repository-adapted scripts change only public input/output path handling and presentation where necessary. They do not change frozen scientific values or benchmark definitions.

Illustrative or structural figures are included only where an exact retained script was available and no restricted dependency was required.
