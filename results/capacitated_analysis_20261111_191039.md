# Capacitated extension: analysis of `capacitated_20261111_170845.csv`

12 distinct ORLIB problems; capacity of [UL] (A.4), at most k customers per facility. Binding (CFLP opt > UFLP opt): k=5 12/12, k=10 8/12. QUBO variables: k=5 864–2700, k=10 880–2750 (uncapacitated 816–2,550). Uncapacitated reference: the same problems, r and seeds 0-4 from `reproduction_20261110_215451.csv`. Averages are over problems.

## Checks

- [PASS] completeness: 960 rows, 960 expected
- [PASS] invariants: repaired >= CFLP opt True, raw feasible >= CFLP opt True, CFLP opt >= UFLP opt True
- [PASS] reproducible: 6 sampled runs re-executed, 0 differ

## Mean repaired gap % (a run with no repairable read scores 100%)

| r | uncapacitated | k = 10 | k = 5 |
|---|---|---|---|
| 0.01 | 25.53 | 24.42 | 16.43 |
| 0.02 | 14.2 | 25.21 | 19.93 |
| 0.05 | 8.54 | 25.88 | 21.66 |
| 0.1 | 9.83 | 25.64 | 22.13 |
| 0.25 | 12.64 | 25.96 | 22.43 |
| 0.5 | 15.02 | 26.04 | 22.58 |
| 0.75 | 15.77 | 25.91 | 22.6 |
| 1 | 16.29 | 26.18 | 22.48 |

Best r: {'uncapacitated': 0.05, 'k = 10': 0.01, 'k = 5': 0.01}

## Raw-feasible reads % (original constraints satisfied)

| r | uncapacitated | k = 10 | k = 5 |
|---|---|---|---|
| 0.01 | 0 | 0 | 0 |
| 0.02 | 0 | 0 | 0 |
| 0.05 | 0 | 0 | 0 |
| 0.1 | 0 | 1.4 | 1.2 |
| 0.25 | 99.5 | 2.9 | 89.6 |
| 0.5 | 100 | 1.6 | 96.7 |
| 0.75 | 100 | 0.6 | 95.7 |
| 1 | 100 | 0.7 | 95.9 |

## Reads that are feasible *and* have exact slack (energy = cost), %

| r | 5 | 10 |
|---|---|---|
| 0.01 | 0 | 0 |
| 0.02 | 0 | 0 |
| 0.05 | 0 | 0 |
| 0.1 | 1.2 | 1.1 |
| 0.25 | 89.5 | 1.6 |
| 0.5 | 96.5 | 0.7 |
| 0.75 | 95.4 | 0.4 |
| 1 | 92 | 0.3 |

## Mean raw gap % (best feasible read; – = none)

| r | uncapacitated | k = 10 | k = 5 |
|---|---|---|---|
| 0.01 | – | – | – |
| 0.02 | – | – | – |
| 0.05 | – | – | – |
| 0.1 | – | 86.96 | 92.14 |
| 0.25 | 58.52 | 101.94 | 78.31 |
| 0.5 | 82.03 | 149.25 | 93.83 |
| 0.75 | 94.45 | 165.2 | 99.49 |
| 1 | 100.85 | 157.95 | 106.15 |

## Reads where repair failed (some customer did not fit), total

| r | 5 | 10 |
|---|---|---|
| 0.01 | 0 | 0 |
| 0.02 | 0 | 0 |
| 0.05 | 0 | 0 |
| 0.1 | 0 | 0 |
| 0.25 | 0 | 0 |
| 0.5 | 0 | 0 |
| 0.75 | 0 | 0 |
| 1 | 0 | 0 |
