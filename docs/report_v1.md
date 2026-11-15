# Report v1

Whole-match statistics use no ground truth. The public-window score at the end is only reported; no threshold was adjusted after seeing it.

## Whole match, graded frames

| half | graded frames | possessions | mean duration (s) | % H | % A | % D |
|---|---|---|---|---|---|---|
| 1 | 79467 | 164 | 12.0 | 36.1 | 26.0 | 37.9 |
| 2 | 79390 | 175 | 10.3 | 31.7 | 24.8 | 43.5 |
| both | 158857 | 339 | 11.1 | 33.9 | 25.4 | 40.7 |

Possession = maximal run of H or of A; D separates possessions. Before post-processing there were 387 possession runs; short ones are merged (`src/postprocess.py`).

## Dead-ball frames by rule

A frame covered by several rules is credited to the first of R1, R4, R3, R2, R6, R5 (listed in that order of priority). 'raw' counts every frame a rule marks, overlaps included. Both after the 2 s minimum-duration filter.

| half | R1 | R2 | R3 | R4 | R5 | R6 | total D |
|---|---|---|---|---|---|---|---|
| 1 | 0 (0.0%) / raw 0 | 2132 (2.7%) / raw 4191 | 5402 (6.8%) / raw 5960 | 10164 (12.8%) / raw 10164 | 11410 (14.4%) / raw 24071 | 957 (1.2%) / raw 1086 | 30065 |
| 2 | 3015 (3.8%) / raw 3015 | 4893 (6.2%) / raw 7862 | 6929 (8.7%) / raw 7179 | 6326 (8.0%) / raw 6326 | 12418 (15.6%) / raw 26222 | 811 (1.0%) / raw 1067 | 34392 |

Frames removed by the minimum duration (2 s), whole match: 531.

## phase_team: reading of the Smart Tagging Team tag

Agreement = share of graded frames with the ball visible, inside the phase, where the nearest team to the ball is the one named in the Team tag. No labels involved. 'Prior' comes from the code name; it is used only if the measured agreement lies on the same side of 0.5 as the prior predicts.

| code | frames | agreement with nearest team | reading used |
|---|---|---|---|
| ATTACKING TRANSITION | 8362 | 0.625 | owner |
| BALL IN FINAL THIRD | 39844 | 0.609 | owner |
| BALL IN THE BOX | 20247 | 0.538 | owner |
| BUILD UP | 10210 | 0.759 | owner |
| CREATING CHANCES | 12098 | 0.548 | owner |
| DEFENDING IN ATTACKING THIRD | 10210 | 0.241 | defender |
| DEFENDING IN DEFENSIVE THIRD | 12098 | 0.452 | defender |
| DEFENDING IN MIDDLE THIRD | 28837 | 0.303 | defender |
| DEFENSIVE TRANSITION | 8362 | 0.375 | defender |
| LONG BALL | 8304 | 0.621 | owner |
| PLAYERS IN THE BOX | 41730 | 0.532 | owner |
| PROGRESSION | 28837 | 0.697 | owner |

Combined `phase_team` covers 66.4% of graded frames and agrees with the nearest team on 63.3% of the graded frames where both exist.

## Restart team: tag vs first controller

For restarts whose team is known (Team tag of the set piece; for a Kick Off (After Goal), the team that did not score), is the first team to control the ball after the restart the same? Throw-ins and kick-offs at the start are N/A in the tags and are handled by the first controller.

| restart type | segments | with a controller | team known and controller seen | first controller = taker |
|---|---|---|---|---|
| Corner Kick | 10 | 10 | 10 | 3 |
| Free Kick | 3 | 3 | 2 | 1 |
| Goal Kick | 9 | 8 | 8 | 7 |
| Kick Off (After Goal) | 4 | 4 | 3 | 3 |
| Kick Off (Start) | 1 | 1 | 0 | 0 |
| Throw-in | 25 | 25 | 0 | 0 |

## Live segments

142 live segments; 245 owner switches in total, 5 of them placed inside a ball gap (phase change or middle of the gap); segments with no control episode: 7 (phase fallback 5, previous owner 2, nothing 0).

## Public 5-minute window (reported, not tuned on)

Frames 11185-18684, 17 possessions. Produced by `python evaluation/run_evaluation.py --bootstrap 1000` after copying the CSV to `evaluation/results/v1.csv`. With 17 possessions the bootstrap interval is wide: differences below ~0.3 spellF1 are noise.

```
ranked on spell_f1_mean; gaps below ~0.33 are within noise (widest bootstrap interval, 17 possessions, 1000 draws)
rank  submission                macroF1  spellF1   poss     dur  transF1  lat_f  shareE  frag
----  ------------------------  -------  -------  -----  ------  -------  -----  ------  ----
   1  v1                         0.7511   0.4324     20    13.5   0.2273   -7.6   0.019  1.19
   2  baseline_smooth125         0.5812   0.4242     16    18.5   0.1667   -7.7   0.052  0.81
   3  baseline_smooth50          0.6207   0.3256     26    10.8   0.0816   +4.0   0.048  1.43
   4  baseline_smooth25          0.6200   0.2868     36     7.6   0.2034   +8.2   0.056  1.90
   5  baseline_raw               0.5972   0.0378    131     2.0   0.1410   -3.1   0.046  6.52
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
