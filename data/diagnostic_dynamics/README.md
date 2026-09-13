# Short AIMNet2 diagnostic dynamics

The retained NVT and NVE time series support dissertation Section 3.4. NVT ran 1000 steps at 0.25 fs; NVE continued the final coordinates and momenta for 2000 steps at 0.25 fs. The NVE file reports time relative to its own start; add 0.25 ps for continuous time.

Each original time series contains two initial step 0 rows. Preserve them: the reported NVT mean 340.800936 K includes both. Removing the duplicate gives 340.981431 K and would alter the historical summary. NVE drift and range are unchanged by that duplicate. Preserve original observations; do not silently deduplicate.

These are finite non-periodic +1 N16 diagnostics with no chloride. They are not evidence of production-dynamics accuracy. The seed controls velocity initialisation; the retained script does not explicitly seed Langevin thermostat noise. A new trajectory need not reproduce these observations exactly.


## Source and public-byte provenance

| Public file | Retained original SHA256 | Public LF SHA256 |
|---|---|---|
| aimnet2_nvt_timeseries.csv | 0c7b80e523ac38971c2e28e673069a9785a7284d52864a78aed14fafdea75329 | e127ef931bbb8b6ab1e19becd6b31bfc2ca156006e665abad2ae9f01a9238e51 |
| aimnet2_nve_timeseries.csv | bb8269878a8923af2b778bfe780d4ab13388f176c442109dfd43e0b2330d9c5d | faeefb724697f65ba1d1c12f1508b4acb7f66b1c1aa130cf7a5160eb0cd4c26f |

Public CSV values and row order are unchanged; CRLF source text is represented as LF. Source hashes above refer to original retained bytes.
