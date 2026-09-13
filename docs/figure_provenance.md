# Figure sources

Figure numbering follows the submitted 57-page dissertation PDF (SHA256: `df2939cea3cb46660d2f2224ee8f3c4912a4fc7e61e32f19cd9677801bfe7726`). The files are in [figures/final_dissertation](../figures/final_dissertation/).

Figures 4–11 were recovered from the Word document and matched to the PDF's embedded images using the recorded crops. Figure 1 is available as the embedded SVG and a PDF excerpt. Figure 3 is the embedded EMF, checked visually against the submitted page. These are document-source files; their identity with any earlier export files has not been established.

## Figure mapping

| Figure | PDF page | Archived source and relationship to scripts |
|---|---|---|
| 1 | 10 | Embedded SVG and PDF excerpt. The public milestones script has different typography and placement. |
| 2 | 13 | Literature figure, cited below; no image included. |
| 3 | 15 | Embedded EMF. The earlier RDKit/CairoSVG script produces a different molecular illustration. |
| 4 | 19 | Word image matching the PDF crop. The earlier workflow script omits the final MD100 B3LYP branch. |
| 5 | 23 | Word image matching the PDF crop. A retained local hydration plotting script ran successfully. |
| 6 | 24 | Word image matching the PDF crop. A retained local snapshot script ran successfully; the document applies crops. |
| 7 | 33 | Word image matching the PDF crop. The public rank script reproduces the data with different styling. |
| 8 | 35 | Word image matching the PDF crop. The submitted figure has two panels; the public variant has one. |
| 9 | 37 | Word image matching the PDF crop. Public data and panel content agree; layout and typography differ. |
| 10 | 39 | Word image matching the PDF crop. The current public script has corrected Å labels; aspect ratio and typography differ. |
| 11 | 41 | Word image matching the PDF crop. The public recovery plot uses different colours, ticks and layout. |

Some scripts retain earlier figure numbers in their filenames. The molecular Figure 2 script relates to submitted Figure 3; the workflow Figure 3 script relates to submitted Figure 4. The model-model parity plot labelled Figure 7 is separate from the final B3LYP rank figure. The [script inventory](script_inventory.md) lists current roles and inputs.

## Software provenance

The original drawing application for Figure 3 and the precise software environments used for every submitted figure are unknown. Available environment information and successful script runs do not identify the original exports. [RDKit environment notes](../environments/rdkit_figures/README.md) describe the retained records.

## Figure manifest

[FIGURE_MANIFEST.csv](../figures/final_dissertation/FIGURE_MANIFEST.csv) contains the page number, source document, asset format, SHA256, crops, data inputs and available script/environment information. Figure 1 has two asset rows. Figure 2 has a reference row without an image file.

The asset checksums identify the archived files. Candidate-script checksums identify retained local scripts and do not imply that those scripts are portable repository commands. The comparison statuses record the original v1.1.0 audit; the table above reflects the subsequent unit-label correction.

## Literature figure

Figure 2 is from Tripathi et al., *Unveiling Zwitterionization of Glycine in the Microhydration Limit*, ACS Omega 2021, 6, 12676–12683, [DOI: 10.1021/acsomega.1c00869](https://pubs.acs.org/doi/10.1021/acsomega.1c00869). The article uses [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). The image is not included in this repository.
