# Report v2

Live/stopped model trained on algorithmic pseudo-labels (no human labels, no ground truth), decoded with a two-state HMM, replacing rule R5 and the look-back/stoppage caps of v1. Everything below the cross-validation uses no ground truth; the public window at the end is only reported.

## Pseudo-labels (graded frames)

| source | frames |
|---|---|
| ball_out | 3003 |
| still | 2153 |
| set_piece | 6258 |
| goal | 16490 |
| half_start | 3015 |
| stopped (union) | 29275 |
| live | 33449 |
| live overridden by stopped | 1295 |
| unlabelled graded | 97428 |

## Cross-validation on pseudo-labels, no ground truth

Train on one half, test on the pseudo-labelled frames of the other. The rule is v1's R5 applied as a stopped/live classifier everywhere (5 s mean of the median player speed < 1.4 m/s = stopped), on exactly the same test frames (those where that mean exists). Model threshold 0.5 (class-balanced training). Accuracy depends on the class mix, so balanced accuracy and AUC are the comparison.

| direction | classifier | test frames | of which stopped | accuracy | balanced accuracy | AUC |
|---|---|---|---|---|---|---|
| train 1 -> test 2 | model | 30460 | 14999 | 0.831 | 0.829 | 0.898 |
| train 1 -> test 2 | rule R5 | 30460 | 14999 | 0.833 | 0.833 | 0.909 |
| train 2 -> test 1 | model | 30667 | 13974 | 0.843 | 0.845 | 0.920 |
| train 2 -> test 1 | rule R5 | 30667 | 13974 | 0.784 | 0.787 | 0.882 |

## Whole match: v1 and v2

| half | version | possessions | mean poss. (s) | % H | % A | % D | stoppages | mean (s) | median (s) | p90 (s) | max (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | v1 | 164 | 12.0 | 36.1 | 26.0 | 37.9 | 59 | 20.4 | 7.8 | 34.0 | 212.5 |
| 1 | v2 | 143 | 13.4 | 36.3 | 23.8 | 39.9 | 35 | 36.2 | 21.0 | 78.6 | 224.0 |
| 2 | v1 | 175 | 10.3 | 31.7 | 24.8 | 43.5 | 73 | 18.9 | 11.3 | 31.2 | 253.8 |
| 2 | v2 | 158 | 12.1 | 35.1 | 25.1 | 39.8 | 51 | 24.8 | 15.6 | 45.4 | 253.0 |

## Where v1 and v2 disagree

| frames | count | % differ | % v1 D, v2 live | % v1 live, v2 D | % live in both, other team |
|---|---|---|---|---|---|
| all graded frames | 158857 | 9.93 | 5.16 | 4.32 | 0.45 |
| ball seen | 93968 | 4.95 | 0.73 | 3.76 | 0.47 |
| ball not seen | 64889 | 17.15 | 11.58 | 5.15 | 0.42 |

## Decision criterion (fixed beforehand)

- (a) model beats R5 on AUC and balanced accuracy in both directions: **FAIL**
- (b) plausible whole match: D share 39.9% (30-50%), stoppages > 180 s outside goals and half openings: 0, possessions v2 301 vs v1 339 (limit 1.5x = 508): **PASS**
- Result: **v1 stays the final version**

## Model and HMM

Standardised logistic weights of the model trained on both halves (positive = more likely stopped):

| feature | weight |
|---|---|
| speed_p25_1s | 0.01 |
| speed_median_1s | 0.71 |
| speed_p75_1s | -0.40 |
| speed_p90_1s | -0.08 |
| frac_walk_1s | 0.34 |
| frac_run_1s | 0.08 |
| centroid_speed_1s | 0.28 |
| spread_1s | 0.10 |
| depth_std_1s | 0.40 |
| mean_abs_acc_1s | 0.16 |
| n_tracked_1s | 1.66 |
| speed_p25_5s | -0.66 |
| speed_median_5s | -1.70 |
| speed_p75_5s | -2.17 |
| speed_p90_5s | -0.89 |
| frac_walk_5s | -0.80 |
| frac_run_5s | 1.30 |
| centroid_speed_5s | 1.20 |
| spread_5s | -1.41 |
| depth_std_5s | 0.88 |
| mean_abs_acc_5s | -0.28 |
| n_tracked_5s | -4.26 |
| intercept | 0.68 |

| transition | per frame | expected duration |
|---|---|---|
| stopped -> live | 0.00160 | 25 s |
| live -> stopped | 0.00100 | 40 s |

Evidence tempering 0.040 (one independent observation per second).

## Public 5-minute window (reported only, not used for the decision)

17 possessions: differences below ~0.3 spellF1 are noise.

```
ranked on spell_f1_mean; gaps below ~0.33 are within noise (widest bootstrap interval, 17 possessions, 1000 draws)
rank  submission                macroF1  spellF1   poss     dur  transF1  lat_f  shareE  frag
----  ------------------------  -------  -------  -----  ------  -------  -----  ------  ----
   1  v1                         0.7511   0.4324     20    13.5   0.2273   -7.6   0.019  1.19
   2  baseline_smooth125         0.5812   0.4242     16    18.5   0.1667   -7.7   0.052  0.81
   3  v2                         0.6974   0.4229     18    15.6   0.1538   +0.3   0.007  0.95
   4  baseline_smooth50          0.6207   0.3256     26    10.8   0.0816   +4.0   0.048  1.43
   5  baseline_smooth25          0.6200   0.2868     36     7.6   0.2034   +8.2   0.056  1.90
   6  baseline_raw               0.5972   0.0378    131     2.0   0.1410   -3.1   0.046  6.52
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
v2
  macro F1 0.6974  (H 0.7923  A 0.7962  D 0.5038)   kappa 0.5951  acc 0.7559
  bootstrap macroF1 5-95 pct  0.4915 .. 0.8331   (21 segments, 1000 draws)
  per class          precision   recall       F1   support
    H                 0.7335   0.8613   0.7923      3454
    A                 0.7466   0.8529   0.7962      2577
    D                 0.9920   0.3376   0.5038      1469
  confusion  rows=GT cols=pred         H       A       D       X
                          H        2975     479       0       0
                          A         375    2198       4       0
                          D         706     267     496       0
                          X       52611   35936   62810   40462
  transitions  gt 20  pred 19
      tau=12  P 0.1579  R 0.1500  F1 0.1538   matched 3  missed 17  spurious 16
      tau=25  P 0.1579  R 0.1500  F1 0.1538   matched 3  missed 17  spurious 16
  latency      mean +0.3  median -1.0  p90|lat| 3  ambiguous 0
  possessions  gt 17  pred 18  (+1)   mean duration 15.6 s vs 14.2 s (+10%)
  spell F1     mean 0.4229   @0.3 0.6286  @0.4 0.5143  @0.5 0.4571  @0.6 0.2857  @0.7 0.2286
      bootstrap 5-95 pct  0.2743 .. 0.5714   (17 possessions, 1000 draws)
      IoU 0.5   P 0.4444  R 0.4706  F1 0.4571   matched 8  missed 9  spurious 10
  alive/dead F1 0.9250   team acc  gt-alive 0.8577  both-alive 0.8583
  share        gt home 0.5727  pred 0.5794  err 0.0067
  fragmentation  20 / 21 = 0.95
  abstained    0 frames of X on graded ground truth (0.00%)

```
