# N16 Extraction

`extract_N16_clusters.py` is the original proprietary-trajectory stage script. It retains tacrinium and 16 nearest water molecules for the selected 100 readable MD frames, omitting chloride and periodic boundaries from each finite cluster. It assumes the original relative project layout and writes stage outputs.

The public canonical XYZ files are in `data/n16_xyz/`. From the repository root, run:

```bash
python scripts/n16_extraction/qc_N16_xyz.py --input-dir data/n16_xyz
```

QC checks 100 files, 78 atoms, composition C13H47N2O16, finite coordinates and minimum interatomic distance. Only diagonal self-distances are excluded, so distinct coincident atoms are detected. Preserve atom order and canonical XYZ byte identity.
