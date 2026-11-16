# Report v4

Calibration on the three public Metrica sample games (other matches, labels from the data provider; see `docs/external_data.md`). Nothing from BAR-ATH, not even the public 5 minutes, was used to choose or tune anything. The public window is only reported at the end.

## Part B: possession logic (true dead-ball mask)

The dead-ball mask is the true one, so only the H/A logic is measured. 'erased' removes 40% of the live-play ball frames in long gaps (5-10 s: 14% of the erased frames, 10-60 s: 86%, the E2 distribution), mean of 2 seeds. Grid: 324 configurations (control radius [1.0, 1.5, 2.0], margin [0.0, 0.3, 0.6], minimum control to switch [6, 12, 25] frames, two touches on/off, back-dating on/off, minimum possession [25, 38, 75] frames). Restart teams are not given to the logic (no Smart Tagging here): after a stoppage the first team to control the ball owns it.

**v1 configuration** (r=1.5 m=0.3 min=12 2touch=on backdate=on poss>=38):

| game | spellF1 true ball | spellF1 erased | macroF1 true ball | macroF1 erased |
|---|---|---|---|---|
| Sample_Game_1 | 0.7515 | 0.6255 | 0.9481 | 0.8775 |
| Sample_Game_2 | 0.7552 | 0.6498 | 0.9577 | 0.9037 |
| Sample_Game_3 | 0.7364 | 0.6047 | 0.9526 | 0.8853 |

**Cross-game selection** (configuration chosen on the two training games by the mean spellF1 over games and ball conditions, tie-break fewest changes from v1; scored on the third):

| held-out game | chosen on the other two | spellF1 true ball (chosen vs v1) | spellF1 erased (chosen vs v1) | beats v1 in both |
|---|---|---|---|---|
| Sample_Game_1 | r=1.0 m=0.3 min=6 2touch=on backdate=on poss>=25 | 0.8095 vs 0.7515 | 0.6595 vs 0.6255 | yes |
| Sample_Game_2 | r=1.0 m=0.3 min=6 2touch=on backdate=off poss>=25 | 0.7935 vs 0.7552 | 0.6794 vs 0.6498 | yes |
| Sample_Game_3 | r=1.0 m=0.3 min=6 2touch=on backdate=off poss>=25 | 0.7560 vs 0.7364 | 0.6073 vs 0.6047 | yes |

**Configuration adopted if B passes** (best on all three games, objective 0.7217 vs 0.6872 for v1): r=1.0 m=0.3 min=6 2touch=on backdate=off poss>=25

| game | v1 true | adopted true | v1 erased | adopted erased |
|---|---|---|---|---|
| Sample_Game_1 | 0.7515 | 0.8250 | 0.6255 | 0.6688 |
| Sample_Game_2 | 0.7552 | 0.7935 | 0.6498 | 0.6794 |
| Sample_Game_3 | 0.7364 | 0.7560 | 0.6047 | 0.6073 |

Mean spellF1 over the grid for each parameter value (all games, both conditions):

| parameter | value | mean spellF1 | best spellF1 |
|---|---|---|---|
| r_ctrl | 1.0 | 0.6584 | 0.7217 |
| r_ctrl | 1.5 | 0.6544 | 0.7103 |
| r_ctrl | 2.0 | 0.6434 | 0.7007 |
| margin | 0.0 | 0.6519 | 0.7153 |
| margin | 0.3 | 0.6544 | 0.7217 |
| margin | 0.6 | 0.6499 | 0.7214 |
| min_control | 6 | 0.6572 | 0.7217 |
| min_control | 12 | 0.6624 | 0.7152 |
| min_control | 25 | 0.6366 | 0.7120 |
| two_touch | True | 0.6649 | 0.7217 |
| two_touch | False | 0.6393 | 0.7117 |
| backdate | True | 0.6561 | 0.7214 |
| backdate | False | 0.6480 | 0.7217 |
| min_poss | 25 | 0.6736 | 0.7217 |
| min_poss | 38 | 0.6673 | 0.7129 |
| min_poss | 75 | 0.6153 | 0.6475 |

The adopted values of the control radius, the minimum control and the minimum possession sit at the lower edge of the grid; the grid was not extended.

## Part C: stopped ball without the ball

Logistic model (numpy, balanced classes) on player-only features without the number of tracked players, trained with the REAL dead-ball labels of two games and tested on the third. The comparison is the v1 pace rule R5 (5 s mean of the median player speed < 1.4 m/s) on the same frames.

| test game | classifier | frames | stopped | accuracy | balanced accuracy | AUC |
|---|---|---|---|---|---|---|
| Sample_Game_1 | model | 145005 | 56754 | 0.846 | 0.860 | 0.917 |
| Sample_Game_1 | rule R5 | 145005 | 56754 | 0.795 | 0.806 | 0.867 |
| Sample_Game_2 | model | 141155 | 57883 | 0.895 | 0.893 | 0.950 |
| Sample_Game_2 | rule R5 | 141155 | 57883 | 0.809 | 0.816 | 0.898 |
| Sample_Game_3 | model | 143760 | 46324 | 0.906 | 0.877 | 0.950 |
| Sample_Game_3 | rule R5 | 143760 | 46324 | 0.859 | 0.854 | 0.926 |

**Decoder calibration** by the spellF1 of the complete pipeline (decoded dead mask + possession logic with the Part B configuration, true ball and erased), nested cross-game validation: 252 decoders (HMM transition durations 10/25/50 s stopped x 20/40/80 s live, evidence tempering 1/10, 1/25, 1/100, or a plain threshold, each with a minimum stoppage of 25/50/100 frames).

| held-out game | decoder chosen on the other two | test spellF1 (true / erased) | v2-style HMM (25 s, 40 s, 1/25) | R5 pace rule |
|---|---|---|---|---|
| Sample_Game_1 | threshold 0.5, min 50 f | 0.640 / 0.495 | 0.681 / 0.493 | 0.555 / 0.419 |
| Sample_Game_2 | HMM stop 10.0 s, live 20.0 s, tempering 0.100, min 25 f | 0.608 / 0.458 | 0.590 / 0.432 | 0.482 / 0.374 |
| Sample_Game_3 | HMM stop 50.0 s, live 20.0 s, tempering 0.100, min 25 f | 0.579 / 0.394 | 0.565 / 0.364 | 0.485 / 0.347 |

Decoder chosen on all three games: **threshold P(stopped) >= 0.5** with a minimum stoppage of 50 frames (objective 0.5376). The choice among decoders was not stable across folds (a plain threshold in one, HMMs in the others), and the differences between them are small next to the gap to R5; v4 uses the simplest.

## BAR-ATH: v1 and v4 (no ground truth)

v4b = v1 + the new possession configuration only; v4c = v1 + the model in place of R5 only; v4 = both.

| half | version | possessions | mean poss. (s) | % H | % A | % D | stoppages | mean (s) | median (s) | max (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | v1 | 164 | 12.0 | 36.1 | 26.0 | 37.9 | 59 | 20.4 | 7.8 | 212.5 |
| 1 | v4b | 178 | 11.1 | 37.1 | 25.0 | 37.9 | 59 | 20.4 | 7.8 | 212.5 |
| 1 | v4c | 159 | 12.1 | 35.5 | 24.9 | 39.6 | 55 | 22.9 | 10.3 | 212.5 |
| 1 | v4 | 171 | 11.2 | 36.2 | 24.2 | 39.6 | 55 | 22.9 | 10.3 | 212.5 |
| 2 | v1 | 175 | 10.3 | 31.7 | 24.8 | 43.5 | 73 | 18.9 | 11.3 | 253.8 |
| 2 | v4b | 183 | 9.8 | 31.3 | 25.3 | 43.4 | 75 | 18.4 | 10.5 | 253.8 |
| 2 | v4c | 174 | 10.3 | 31.8 | 24.8 | 43.5 | 71 | 19.4 | 11.5 | 253.8 |
| 2 | v4 | 182 | 9.9 | 31.5 | 25.1 | 43.4 | 73 | 18.9 | 10.5 | 253.8 |

Whole match possessions: v1 339, v4b 361, v4c 333, v4 353; D share: v1 40.7%, v4b 40.7%, v4c 41.5%, v4 41.5%. v1 and v4 differ on 9.07% of the graded frames.

## Decision (criterion fixed beforehand)

- Part B (beats v1 in 3 of 3 held-out games, true and erased ball): **PASS**
- Part C (beats R5 on AUC and balanced accuracy in 3 of 3 held-out games): **PASS**
- v4 = v1 + B (possession configuration) + C (stopped-ball model)
- BAR-ATH plausibility of v4 (D 41.5% in 30-50%, possessions 353 vs 339 for v1, limit 508): **PASS**
- Result: **v4 is the final version**

## Public 5-minute window (reported only, not used for any choice)

17 possessions: differences below ~0.3 spellF1 are noise.

```
ranked on spell_f1_mean; gaps below ~0.33 are within noise (widest bootstrap interval, 17 possessions, 1000 draws)
rank  submission                macroF1  spellF1   poss     dur  transF1  lat_f  shareE  frag
----  ------------------------  -------  -------  -----  ------  -------  -----  ------  ----
   1  v4                         0.7483   0.4889     19    14.5   0.2927   -6.5   0.024  1.05
   2  v4b                        0.7534   0.4757     20    13.5   0.3182  -11.3   0.018  1.19
   3  v1                         0.7511   0.4324     20    13.5   0.2273   -7.6   0.019  1.19
   4  v3                         0.7511   0.4324     20    13.5   0.2273   -7.6   0.019  1.19
   5  baseline_smooth125         0.5812   0.4242     16    18.5   0.1667   -7.7   0.052  0.81
   6  v2                         0.6974   0.4229     18    15.6   0.1538   +0.3   0.007  0.95
   7  v4c                        0.7336   0.4222     19    14.5   0.1463   -4.7   0.003  1.05
   8  baseline_smooth50          0.6207   0.3256     26    10.8   0.0816   +4.0   0.048  1.43
   9  baseline_smooth25          0.6200   0.2868     36     7.6   0.2034   +8.2   0.056  1.90
  10  baseline_raw               0.5972   0.0378    131     2.0   0.1410   -3.1   0.046  6.52
```

```
v1
  macro F1 0.7511  (H 0.8279  A 0.8055  D 0.6200)   kappa 0.6553  acc 0.7887
  bootstrap macroF1 5-95 pct  0.5438 .. 0.8549   (21 segments, 1000 draws)
  per class          precision   recall       F1   support
    H                 0.7969   0.8613   0.8279      3454
    A                 0.7479   0.8727   0.8055      2577
    D                 0.9092   0.4704   0.6200      1469
  confusion  rows=GT cols=pred         H       A       D       X
                          H        2975     479       0       0
                          A         259    2249      69       0
                          D         499     279     691       0
                          X       50119   37360   63878   40462
  transitions  gt 20  pred 24
      tau=12  P 0.1667  R 0.2000  F1 0.1818   matched 4  missed 16  spurious 20
      tau=25  P 0.2083  R 0.2500  F1 0.2273   matched 5  missed 15  spurious 19
  latency      mean -7.6  median -10.0  p90|lat| 13  ambiguous 0
  possessions  gt 17  pred 20  (+3)   mean duration 13.5 s vs 14.2 s (-5%)
  spell F1     mean 0.4324   @0.3 0.5946  @0.4 0.5405  @0.5 0.4324  @0.6 0.3243  @0.7 0.2703
      bootstrap 5-95 pct  0.2811 .. 0.5730   (17 possessions, 1000 draws)
      IoU 0.5   P 0.4000  R 0.4706  F1 0.4324   matched 8  missed 9  spurious 12
  alive/dead F1 0.9337   team acc  gt-alive 0.8662  both-alive 0.8762
  share        gt home 0.5727  pred 0.5539  err 0.0189
  fragmentation  25 / 21 = 1.19
  abstained    0 frames of X on graded ground truth (0.00%)

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
