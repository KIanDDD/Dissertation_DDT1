# File manifests and checksums

A manifest is a file inventory. A SHA256 checksum identifies the exact bytes of a file and can detect changes or corruption. Matching checksums establish file integrity; they do not establish scientific accuracy.

## Current repository

| File | Purpose |
|---|---|
| [file_manifest.csv](file_manifest.csv) | Path, size in bytes and SHA256 for each covered file |
| [RELEASE_SHA256SUMS.txt](RELEASE_SHA256SUMS.txt) | The same file checksums in standard SHA256 list format |
| [MAINTENANCE_AUDIT_2026-09-13.txt](MAINTENANCE_AUDIT_2026-09-13.txt) | Validation results and the tested commits |

From the repository root:

```bash
python scripts/validation/verify_release.py
```

The verifier checks file coverage, duplicate entries, sizes and exact checksums. The two lists cover 230 tracked files. The lists themselves and the maintenance report are excluded to avoid circular hashes. The report identifies the commits it tested.

## Historical records

The `PUBLIC_RELEASE_AUDIT_v1.0.0.txt` and `PUBLIC_RELEASE_AUDIT_v1.1.0.txt` files describe the earlier releases. The `*_zip_SHA256.txt` records identify five retained private source archives. The Gaussian, alignment and locked-benchmark CSV manifests describe outputs in those private analysis stages; they are not inventories of this GitHub checkout.

The unchanged [v1.1.0 release](https://github.com/KIanDDD/Dissertation_DDT1/releases/tag/v1.1.0) has a historical list of 200 checksums calculated from Windows files. Of these, 198 used CRLF line endings throughout. The supervisor archive hash record and `make_Figure7_MD100_B3LYP_vs_MLFF_ranks.py` used LF internally and CRLF at the final newline. Those differences explain why the historical list does not match an LF checkout. Verification against the original line endings reproduced all 200 hashes.

## Updating the current lists

After staging the final file changes, run:

```bash
python scripts/validation/verify_release.py --write-from-index
```

This reads the staged Git files. Commit the two generated lists with the changes, then run the verifier in a fresh checkout. The repository's `.gitattributes` specifies LF for text files and preserves the archived figure bytes.
