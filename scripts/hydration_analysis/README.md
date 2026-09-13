# Hydration Analysis

Original-stage hydration analysis precedes N16 construction. `calculate_rdf_water_counts.py` reads the proprietary Schrödinger/Desmond trajectory in the original project layout. `analyse_microhydration_candidates.py` compares fixed-N water subsets.

The system is orthorhombic, so the retained minimum-image calculation applies to the recorded cell. The operational hydration-envelope minimum is 5.325 Å; modal occupancy is 48 waters. The older restricted peak search reports 4.475 Å, while the final plotting analysis identifies the local maximum at 4.875 Å; the cutoff is unchanged. N16 is the 16 nearest waters, not the full envelope.

The scripts use stage-relative paths rather than a repository-root CLI. The public tables in `data/hydration/` support derived-data inspection; the raw trajectory and proprietary runtime are required for full recalculation. Provisional decision text in early output is historical; the dissertation records the final choice.
