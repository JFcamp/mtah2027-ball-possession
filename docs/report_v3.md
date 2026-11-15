# Report v3

v3 = the v1 pipeline with no threshold changed, fed with a ball track completed by the broadcast ball where the tactical tracker has none. Sections 1-3 use no ground truth; the public window at the end is only reported.

## 1. Alignment of the broadcast onto the tactical timeline

Matching of same-team player positions only (the ball is not used). Tactical frame = broadcast frame + offset.

|  | broadcast frames | % of all | % of valid homography |
|---|---|---|---|
| total | 164488 | 100.0 | - |
| valid homography | 108248 | 65.8 | 100.0 |
| aligned | 98653 | 60.0 | 91.1 |

| half | aligned frames | coarse offset | median refined offset | range | within 2 frames of median |
|---|---|---|---|---|---|
| 1 | 50732 | 4004 | 4014 | 4006..4017 | 96.8% |
| 2 | 47921 | 26908 | 26920 | 26915..26928 | 95.5% |

The refined offset is constant to within a few frames in each half; the second half is ~22,900 frames later (halftime and the start of the broadcast).

Players: for every 10th aligned frame with >= 10 visible broadcast players, distance from each broadcast player to the nearest same-team tactical player (9394 frames, 167296 players): median 0.29 m, p90 1.16 m, p99 14.72 m.

Ball: frames where both cameras see the ball (71534 aligned frames; 71532 on graded frames).

| distance between the two ball positions (m) | p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|
| co-visible frames | 0.79 | 2.15 | 5.94 | 10.82 | 24.55 |
| graded only | 0.79 | 2.15 | 5.94 | 10.81 | 24.52 |

## 2. Ball coverage on graded frames

| frames | % raw tactical | % raw fused | % cleaned v1 | % cleaned v3 | frames from broadcast | gain after cleaning (pp) |
|---|---|---|---|---|---|---|
| both halves | 59.3 | 63.2 | 59.2 | 63.0 | 6116 | 3.86 |
| half 1 | 59.3 | 63.2 | 59.3 | 63.1 | 3036 | 3.80 |
| half 2 | 59.3 | 63.1 | 59.0 | 62.9 | 3080 | 3.92 |

The tactical ball is missing on 64643 graded frames; the aligned broadcast has a ball on 6116 of them (9.5%). Where one tracker loses the ball the other usually does too.

## 3. Whole match: v1 and v3

| half | version | possessions | mean poss. (s) | % H | % A | % D | stoppages | mean (s) | median (s) | max (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | v1 | 164 | 12.0 | 36.1 | 26.0 | 37.9 | 59 | 20.4 | 7.8 | 212.5 |
| 1 | v3 | 173 | 11.5 | 35.6 | 26.9 | 37.6 | 58 | 20.6 | 7.9 | 212.5 |
| 2 | v1 | 175 | 10.3 | 31.7 | 24.8 | 43.5 | 73 | 18.9 | 11.3 | 253.8 |
| 2 | v3 | 181 | 10.1 | 31.6 | 26.1 | 42.3 | 71 | 18.9 | 10.4 | 253.7 |

v1 and v3 labels differ on 3640 graded frames (2.29%).

## Decision criterion (fixed beforehand)

- (a) co-visible ball distance median <= 2 m and p90 <= 5 m: median 0.79 m, p90 5.94 m: **FAIL**
- (b) coverage gain >= 5 pp: +3.85 pp raw, +3.86 pp after cleaning: **FAIL**
- (c) D share 39.9% (30-50%), possessions v3 354 vs v1 339 (limit 508): **PASS**
- Result: **v1 stays the final version**

## Public 5-minute window (reported only, not used for the decision)

17 possessions: differences below ~0.3 spellF1 are noise.

```
ranked on spell_f1_mean; gaps below ~0.33 are within noise (widest bootstrap interval, 17 possessions, 1000 draws)
rank  submission                macroF1  spellF1   poss     dur  transF1  lat_f  shareE  frag
----  ------------------------  -------  -------  -----  ------  -------  -----  ------  ----
   1  v1                         0.7511   0.4324     20    13.5   0.2273   -7.6   0.019  1.19
   2  v3                         0.7511   0.4324     20    13.5   0.2273   -7.6   0.019  1.19
   3  baseline_smooth125         0.5812   0.4242     16    18.5   0.1667   -7.7   0.052  0.81
   4  v2                         0.6974   0.4229     18    15.6   0.1538   +0.3   0.007  0.95
   5  baseline_smooth50          0.6207   0.3256     26    10.8   0.0816   +4.0   0.048  1.43
   6  baseline_smooth25          0.6200   0.2868     36     7.6   0.2034   +8.2   0.056  1.90
   7  baseline_raw               0.5972   0.0378    131     2.0   0.1410   -3.1   0.046  6.52
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
v3
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
                          X       49639   39072   62646   40462
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
