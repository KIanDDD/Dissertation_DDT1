# Figure provenance

The submitted 57-page dissertation PDF defines the numbering and appearance. Its SHA256 is `df2939cea3cb46660d2f2224ee8f3c4912a4fc7e61e32f19cd9677801bfe7726`. The archive contains recovered document-source assets and one explicitly labelled PDF extraction. It does not claim every original high-resolution export or generating environment has been recovered.

Figures 4–11 are unchanged media from the retained dissertation Word document. Their pixels match the submitted PDF's embedded images after the crop in `figures/final_dissertation/FIGURE_MANIFEST.csv`. Figure 1's SVG is the actual embedded Word relationship target; its separate PDF clip preserves the submitted typography. Figure 3's EMF is the embedded source and was visually checked against the submitted page. Its drawing application is **UNRESOLVED — EVIDENCE REQUIRED**: an original native drawing file or export record is needed. The different RDKit/CairoSVG alternative is not asserted to be its generator.

The exact original package state for figure production is **UNRESOLVED — EVIDENCE REQUIRED**. Recorded audit replay environments establish that retained candidate scripts run, not that they produced the original embedded assets. A dated package/export record tied to each original asset would resolve that distinction. Current Conda inventories are not represented as historical snapshots.

| Figure | PDF page | Asset | Provenance / public reproduction |
|---|---|---|---|
| 1 | 10 | Figure01_MLFF_evolution.svg | Exact Word relationship/source format; submitted appearance visually checked; NO exact export identity demonstrated; MEANINGFULLY DIFFERENT typography/placement from public script; recover exact PDF clip |
| 2 | 13 | Literature link only | THIRD-PARTY; not redistributed; Not applicable |
| 3 | 15 | Figure03_tacrine_tacrinium.emf | Exact Word relationship/source format; submitted appearance visually checked; NO exact export identity demonstrated; MEANINGFULLY DIFFERENT public RDKit alternative; CairoSVG import fails; recovered EMF visually verified |
| 4 | 19 | Figure04_computational_workflow.png | Proved Word media/PDF embedded pixels after recorded crop; NO exact export identity demonstrated; MEANINGFULLY DIFFERENT public pre-MD100 workflow; local updated candidate still differs in text |
| 5 | 23 | Figure05_hydration_analysis.png | Proved Word media/PDF embedded pixels after recorded crop; NO exact export identity demonstrated; Local original-family generator passes; submitted media is exact, original-export byte identity unproven |
| 6 | 24 | Figure06_N16_snapshots.png | Proved Word media/PDF embedded pixels after recorded crop; NO exact export identity demonstrated; Local generator passes; submitted media is exact, Word crops differ |
| 7 | 33 | Figure07_MD100_ranks.png | Proved Word media/PDF embedded pixels after recorded crop; NO exact export identity demonstrated; MEANINGFULLY DIFFERENT public plot; local OLDSTYLE generator passes |
| 8 | 35 | Figure08_intermodel_force_disagreement.png | Proved Word media/PDF embedded pixels after recorded crop; NO exact export identity demonstrated; MEANINGFULLY DIFFERENT one-panel public plot; local two-panel generator passes |
| 9 | 37 | Figure09_path_energy.png | Proved Word media/PDF embedded pixels after recorded crop; NO exact export identity demonstrated; Public data and panel content agree; exact layout/typography not reproduced |
| 10 | 39 | Figure10_path_force_error.png | Proved Word media/PDF embedded pixels after recorded crop; NO exact export identity demonstrated; MEANINGFULLY DIFFERENT public corrupted unit; local source candidate passes |
| 11 | 41 | Figure11_retrospective_recovery.png | Proved Word media/PDF embedded pixels after recorded crop; NO exact export identity demonstrated; MEANINGFULLY DIFFERENT public colours/ticks/layout; local final generator passes |

Draft numbering is preserved only as historical filenames: the old molecular Figure2 script relates to final Figure3; the old workflow Figure3 script is superseded by final Figure4; model-model parity labelled Figure7 is not the final B3LYP-rank Figure7. The public force-disagreement variant has one panel while submitted Figure8 has two. The figure manifest and script inventory are the authoritative current mapping. Numerical regeneration and exact archived artwork are separate outputs.

Figure 2 is Tripathi et al.'s *Unveiling Zwitterionization of Glycine in the Microhydration Limit*, ACS Omega 2021, 6, 12676–12683, DOI 10.1021/acsomega.1c00869. The [publisher](https://pubs.acs.org/doi/10.1021/acsomega.1c00869) specifies [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). The image is not redistributed here, and the repository rights statement is not applied to it.

## Manifest schema

Each row records figure_number, dissertation_page, description, exact_final_asset, asset_format, sha256, source_document, generator_status, generator_candidate, generator_candidate_sha256, data_inputs, historical_environment, audit_replay_environment, status, crop and notes. Figure 1 has two asset rows; Figure 2 has a provenance-only row with no redistributed asset. Hashes identify exact archived bytes, not regenerated alternatives. Candidate-generator hashes identify retained local scripts, which are not automatically portable public generators.
