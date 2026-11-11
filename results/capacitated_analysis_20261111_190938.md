# Capacitated extension: analysis of `capacitated_decoupled_20261111_175554.csv`

12 distinct ORLIB problems; capacity of [UL] (A.4), at most k customers per facility. Binding (CFLP opt > UFLP opt): k=5 12/12, k=10 8/12. QUBO variables: k=5 864–2700, k=10 880–2750 (uncapacitated 816–2,550). Uncapacitated reference: the same problems, r and seeds 0-4 from `reproduction_20261110_215451.csv`. Averages are over problems.

## Checks

- [PASS] completeness: 576 rows, 576 expected
- [PASS] invariants: repaired >= CFLP opt True, raw feasible >= CFLP opt True, CFLP opt >= UFLP opt True
- [PASS] reproducible: 6 sampled runs re-executed, 0 differ

## Mean repaired gap % (a run with no repairable read scores 100%)

| r | uncapacitated | k = 10 | k = 5 |
|---|---|---|---|
| 0.01 | 25.42 | 12.64 | 68.29 |
| 0.02 | 14.61 | 9.09 | 10.01 |
| 0.05 | 8.31 | 7.65 | 9.37 |
| 0.1 | 9.59 | 8.17 | 9.35 |
| 0.25 | 12.76 | 9.18 | 11.6 |
| 0.5 | 14.86 | 11.24 | 12.88 |
| 0.75 | 15.59 | 11.97 | 13.86 |
| 1 | 16.3 | 12.44 | 14.38 |

Best r: {'uncapacitated': 0.05, 'k = 10': 0.05, 'k = 5': 0.1}

## Raw-feasible reads % (original constraints satisfied)

| r | uncapacitated | k = 10 | k = 5 |
|---|---|---|---|
| 0.01 | 0 | 0 | 0 |
| 0.02 | 0 | 0 | 0 |
| 0.05 | 0 | 0 | 0 |
| 0.1 | 0 | 3.2 | 0.9 |
| 0.25 | 99.6 | 10.7 | 87.2 |
| 0.5 | 100 | 7.7 | 99.8 |
| 0.75 | 100 | 5.8 | 100 |
| 1 | 100 | 5.3 | 100 |

## Reads that are feasible *and* have exact slack (energy = cost), %

| r | 5 | 10 |
|---|---|---|
| 0.01 | 0 | 0 |
| 0.02 | 0 | 0 |
| 0.05 | 0 | 0 |
| 0.1 | 0.9 | 2.3 |
| 0.25 | 87.2 | 6.4 |
| 0.5 | 99.8 | 3.7 |
| 0.75 | 100 | 3.3 |
| 1 | 92.6 | 2.6 |

## Mean raw gap % (best feasible read; – = none)

| r | uncapacitated | k = 10 | k = 5 |
|---|---|---|---|
| 0.01 | – | – | – |
| 0.02 | – | – | – |
| 0.05 | – | – | – |
| 0.1 | – | 43.27 | 58.48 |
| 0.25 | 58.7 | 71.44 | 64.67 |
| 0.5 | 81.9 | 91.66 | 81.45 |
| 0.75 | 94.92 | 101.3 | 90.71 |
| 1 | 100.62 | 109.97 | 97.69 |

## Reads where repair failed (some customer did not fit), total

| r | 5 | 10 |
|---|---|---|
| 0.01 | 1557 | 18 |
| 0.02 | 25 | 0 |
| 0.05 | 0 | 0 |
| 0.1 | 0 | 0 |
| 0.25 | 0 | 0 |
| 0.5 | 0 | 0 |
| 0.75 | 0 | 0 |
| 1 | 0 | 0 |
