# Figure scripts and dissertation correspondence

The four commands in the root README regenerate repository plots from public derived tables and check the locked numerical inputs. They do not all reproduce the submitted artwork exactly. `figures/final_dissertation/` holds recovered artwork and the authoritative Figures 1-11 mapping.

The MD100 rank script corresponds numerically to final Figure 7, with a different public layout. `make_FINAL_Figures7_8_intermodel_repo.py` produces an earlier model-model parity figure and a one-panel force plot; neither its Figure 7 numbering nor its single-panel Figure 8 layout supersedes the dissertation. The final force-disagreement Figure 8 has two panels. The path script maps to Figures 9-10, and the recovery script maps to Figure 11, with documented presentation differences.

`make_intro_figure1_exact.py` creates the milestones schematic but does not reproduce the PDF's narrow typography/placement. `make_intro_figure2_tacrine_tacrinium_FINAL.py` is a historical RDKit/CairoSVG alternative, not the recovered final Figure 3 EMF. `make_figure3_FINAL_NO_OVERFLOW.py` is an earlier workflow schematic, superseded by final Figure 4 with its MD100 B3LYP branch. Historical filenames remain for provenance and compatibility.

The literature Figure 2 is third-party work by Tripathi et al. and has separate CC BY-NC-ND 4.0 conditions. The article is linked in the figure archive; its image is not included.
