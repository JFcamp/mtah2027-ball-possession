# Report v5

Decisions: `docs/decision_v5.md` (committed before this report). Calibration data: the three external games, degraded to BAR-ATH statistics (`docs/domain_gap.md`). The public window is at the end and was not used for any choice.

The tables of Parts A to V are in `docs/decision_v5.md` and `docs/domain_gap.md`; summary:

| criterion | result |
|---|---|
| B (possession configuration, DEG_BAR and DEG_40, 3 of 3) | FAIL |
| C (model, degraded AUC and balanced accuracy, 3 of 3) | PASS: G1 |
| E2E (new pipeline beats the v4-like one, 3 of 3) | FAIL |
| D (T1 transfer, T2 plausibility, T3 structure) | FAIL |
| V (pass-chain veto) | not evaluated |

## BAR-ATH: v4 and v5 (no ground truth)

| half | version | possessions | mean poss. (s) | % H | % A | % D | stoppages >= 2 s | mean (s) | median (s) | max (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | v4 | 171 | 11.2 | 36.2 | 24.2 | 39.6 | 55 | 22.9 | 10.3 | 212.5 |
| 1 | v5 | 171 | 11.2 | 36.2 | 24.2 | 39.6 | 55 | 22.9 | 10.3 | 212.5 |
| 2 | v4 | 182 | 9.9 | 31.5 | 25.1 | 43.4 | 73 | 18.9 | 10.5 | 253.8 |
| 2 | v5 | 182 | 9.9 | 31.5 | 25.1 | 43.4 | 73 | 18.9 | 10.5 | 253.8 |

Frames that change between v4 and v5:

| frames | count | % change | % v4 D, v5 live | % v4 live, v5 D | % live in both, other team |
|---|---|---|---|---|---|
| all graded frames | 158857 | 0.00 | 0.00 | 0.00 | 0.00 |
| ball seen | 93968 | 0.00 | 0.00 | 0.00 | 0.00 |
| ball not seen | 64889 | 0.00 | 0.00 | 0.00 | 0.00 |

## Public 5-minute window (reported only)

17 possessions: differences below ~0.3 spellF1 are noise.

```
ranked on spell_f1_mean; gaps below ~0.33 are within noise (widest bootstrap interval, 17 possessions, 1000 draws)
rank  submission                macroF1  spellF1   poss     dur  transF1  lat_f  shareE  frag
----  ------------------------  -------  -------  -----  ------  -------  -----  ------  ----
   1  v4                         0.7483   0.4889     19    14.5   0.2927   -6.5   0.024  1.05
   2  v5                         0.7483   0.4889     19    14.5   0.2927   -6.5   0.024  1.05
   3  v4b                        0.7534   0.4757     20    13.5   0.3182  -11.3   0.018  1.19
   4  v1                         0.7511   0.4324     20    13.5   0.2273   -7.6   0.019  1.19
   5  v3                         0.7511   0.4324     20    13.5   0.2273   -7.6   0.019  1.19
   6  baseline_smooth125         0.5812   0.4242     16    18.5   0.1667   -7.7   0.052  0.81
   7  v2                         0.6974   0.4229     18    15.6   0.1538   +0.3   0.007  0.95
   8  v4c                        0.7336   0.4222     19    14.5   0.1463   -4.7   0.003  1.05
   9  baseline_smooth50          0.6207   0.3256     26    10.8   0.0816   +4.0   0.048  1.43
  10  baseline_smooth25          0.6200   0.2868     36     7.6   0.2034   +8.2   0.056  1.90
  11  baseline_raw               0.5972   0.0378    131     2.0   0.1410   -3.1   0.046  6.52
```

```
v4
  macro F1 0.7483  (H 0.8330  A 0.8140  D 0.5980)   kappa 0.6605  acc 0.7929
  bootstrap macroF1 5-95 pct  0.5263 .. 0.8645   (21 segments, 1000 draws)
  per class          precision   recall       F1   support
    H                 0.7978   0.8715   0.8330      3454
    A                 0.7454   0.8964   0.8140      2577
    D                 0.9984   0.4268   0.5980      1469
  confusion  rows=GT cols=pred         H       A       D       X
                          H        3010     444       0       0
                          A         266    2310       1       0
                          D         497     345     627       0
                          X       49998   36093   65266   40462
  transitions  gt 20  pred 21
      tau=12  P 0.1429  R 0.1500  F1 0.1463   matched 3  missed 17  spurious 18
      tau=25  P 0.2857  R 0.3000  F1 0.2927   matched 6  missed 14  spurious 15
  latency      mean -6.5  median -11.0  p90|lat| 16  ambiguous 0
  possessions  gt 17  pred 19  (+2)   mean duration 14.5 s vs 14.2 s (+2%)
  spell F1     mean 0.4889   @0.3 0.6667  @0.4 0.6111  @0.5 0.5556  @0.6 0.3333  @0.7 0.2778
      bootstrap 5-95 pct  0.3444 .. 0.6333   (17 possessions, 1000 draws)
      IoU 0.5   P 0.5263  R 0.5882  F1 0.5556   matched 10  missed 7  spurious 9
  alive/dead F1 0.9347   team acc  gt-alive 0.8821  both-alive 0.8823
  share        gt home 0.5727  pred 0.5490  err 0.0237
  fragmentation  22 / 21 = 1.05
  abstained    0 frames of X on graded ground truth (0.00%)

```

```
v5
  macro F1 0.7483  (H 0.8330  A 0.8140  D 0.5980)   kappa 0.6605  acc 0.7929
  bootstrap macroF1 5-95 pct  0.5263 .. 0.8645   (21 segments, 1000 draws)
  per class          precision   recall       F1   support
    H                 0.7978   0.8715   0.8330      3454
    A                 0.7454   0.8964   0.8140      2577
    D                 0.9984   0.4268   0.5980      1469
  confusion  rows=GT cols=pred         H       A       D       X
                          H        3010     444       0       0
                          A         266    2310       1       0
                          D         497     345     627       0
                          X       49998   36093   65266   40462
  transitions  gt 20  pred 21
      tau=12  P 0.1429  R 0.1500  F1 0.1463   matched 3  missed 17  spurious 18
      tau=25  P 0.2857  R 0.3000  F1 0.2927   matched 6  missed 14  spurious 15
  latency      mean -6.5  median -11.0  p90|lat| 16  ambiguous 0
  possessions  gt 17  pred 19  (+2)   mean duration 14.5 s vs 14.2 s (+2%)
  spell F1     mean 0.4889   @0.3 0.6667  @0.4 0.6111  @0.5 0.5556  @0.6 0.3333  @0.7 0.2778
      bootstrap 5-95 pct  0.3444 .. 0.6333   (17 possessions, 1000 draws)
      IoU 0.5   P 0.5263  R 0.5882  F1 0.5556   matched 10  missed 7  spurious 9
  alive/dead F1 0.9347   team acc  gt-alive 0.8821  both-alive 0.8823
  share        gt home 0.5727  pred 0.5490  err 0.0237
  fragmentation  22 / 21 = 1.05
  abstained    0 frames of X on graded ground truth (0.00%)

```
