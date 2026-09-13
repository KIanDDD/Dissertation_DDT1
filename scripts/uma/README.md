# Uma

`uma_batch_100.py` performs fixed-geometry MD100 inference. Use uma-s-1p2, task omol, charge +1, multiplicity 1 and fresh output directories. The retained --resume implementation is historical and does not adequately verify checkpoint/settings/input identity; it is not recommended for a new reproduction.

`run_uma_gaussian_full_244.py` locks the documented path settings and validates input hashes, atom order and force shapes. `compare_mace_aimnet2_uma.py` aligns the 100-structure MACE, AIMNet2 ensemble and UMA tables using filename/hash and atomic keys.

These workflows require the FAIR-Chem environment and third-party model access. Model weights, Hugging Face credentials and cache directories are not redistributed. Model-model agreement is distinct from B3LYP-referenced error.
