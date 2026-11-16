# Domain gap: BAR-ATH vs the external games, and the simulated degradation

No labels are used on either side. BAR-ATH numbers are on the graded frames. Constants of `src/degrade.py` are justified in its docstring.

## A.1 Measurements

**High-frequency noise**: |second difference| of positions, in mm (p50 / p90 / p99). The automatic BAR-ATH tracking is not noisier than the corrected games for players; only the ball is, by a few millimetres.

| data | object | p50 | p90 | p99 |
|---|---|---|---|---|
| BAR-ATH | players | 1.1 | 3.2 | 7.4 |
| BAR-ATH | ball | 4.8 | 21.1 | 44.9 |
| Sample_Game_1 | players | 1.1 | 4.2 | 9.5 |
| Sample_Game_1 | ball | 1.0 | 6.8 | 308.0 |
| Sample_Game_2 | players | 1.1 | 4.8 | 11.6 |
| Sample_Game_2 | ball | 1.0 | 7.5 | 319.8 |
| Sample_Game_3 | players | 1.4 | 5.4 | 13.6 |
| Sample_Game_3 | ball | 0.7 | 6.8 | 317.6 |

**Tracked players per frame** (graded frames): mean, sd, p1, p5, p25, median.

| data | mean | sd | p1 | p5 | p25 | p50 |
|---|---|---|---|---|---|---|
| BAR-ATH | 20.5 | 3.3 | 5 | 14 | 21 | 22 |
| Sample_Game_1 | 22.0 | 0.0 | 22 | 22 | 22 | 22 |
| Sample_Game_2 | 22.0 | 0.0 | 22 | 22 | 22 | 22 |
| Sample_Game_3 | 22.0 | 0.0 | 22 | 22 | 22 | 22 |

Track fragments: BAR-ATH has 46 anonymous slots (tracklets): 1,219 fragments in the graded frames, median 35.5 s, p10 2.6 s. The external games have one track per player (44-50 fragments of 24-48 minutes). No feature uses identities, so this is recorded, not simulated.

**Impossible ball jumps removed by `ball.py`** (% of observed ball frames):

| data | % removed |
|---|---|
| BAR-ATH | 0.30 |
| Sample_Game_1 | 0.08 |
| Sample_Game_2 | 0.12 |
| Sample_Game_3 | 0.02 |

**Disagreement between the two BAR-ATH cameras** (v3 alignment, `docs/report_v3.md`): players, nearest same-team neighbour, median 0.29 m, p90 1.16 m; ball, co-visible frames, median 0.79 m, p75 2.15 m, p90 5.94 m, p95 10.8 m. Autocorrelation of the ball disagreement (central part): 0.995 at 1 frame, 0.90 at 5, 0.28 at 25, 0.05 at 75. Wrong-ball episodes are **not simulated**: the disagreement above 5 m cannot be attributed to a tracker (balls in the air project differently in each camera) and the attributable jumps are only 0.3%.

**Ball absent in the frames v4 labels H/A** (algorithmic estimate of the ball loss during live play, so DEG_BAR erases 10.6% of the live frames; DEG_40 erases 40%):

| frames | % ball absent |
|---|---|
| both halves | 10.6 |
| half 1 | 10.5 |
| half 2 | 10.7 |

## A.2 Degradation parameters (`src/degrade.py`)

| component | value | source |
|---|---|---|
| player position error | sd 0.17 m per axis, smooth (kernel sd 25 frames) | camera disagreement, second differences |
| ball position error | sd 0.47 m per axis, smooth (kernel sd 10 frames) | camera disagreement, second differences |
| player dropouts | solo 2% (mean 5.0 s) + bursts 2.0/min (mean 2.0 s, up to 90% of players) | tracked-count distribution |
| ball erased, DEG_BAR / DEG_40 | 10.6% / 40% of the live frames | v4 labels / E4d |
| wrong-ball episodes | not simulated | cannot be estimated reliably |

Tracked players after degradation (target BAR-ATH: 20.5, 3.3, 5, 14, 21, 22):

| game | mean | sd | p1 | p5 | p25 | p50 | players 2nd diff p50 (mm) | ball 2nd diff p50 (mm) |
|---|---|---|---|---|---|---|---|---|
| Sample_Game_1 seed 0 | 20.8 | 3.3 | 4 | 13 | 21 | 22 | 1.3 | 3.7 |
| Sample_Game_1 seed 1 | 21.0 | 2.9 | 5 | 19 | 21 | 22 | 1.3 | 3.8 |
| Sample_Game_2 seed 0 | 20.9 | 3.0 | 4 | 17 | 21 | 22 | 1.4 | 3.8 |
| Sample_Game_2 seed 1 | 20.9 | 2.9 | 4 | 17 | 21 | 22 | 1.4 | 3.9 |
| Sample_Game_3 seed 0 | 20.9 | 2.9 | 5 | 16 | 21 | 22 | 1.6 | 3.6 |
| Sample_Game_3 seed 1 | 20.9 | 2.9 | 6 | 17 | 21 | 22 | 1.6 | 3.7 |

## A.4 Standardised difference of the model features (BAR-ATH minus external)

SMD = difference of means over the pooled standard deviation, graded frames; 'before' uses the clean external games, 'after' the degraded ones (both seeds pooled). The features are those of the v4 model (20 player features). Part of the gap is the matches themselves (style, pitch use), which no degradation should remove.

| feature | SMD before | SMD after |
|---|---|---|
| speed_p25_1s | -0.09 | -0.11 |
| speed_median_1s | -0.06 | -0.07 |
| speed_p75_1s | -0.05 | -0.06 |
| speed_p90_1s | -0.04 | -0.04 |
| frac_walk_1s | 0.11 | 0.13 |
| frac_run_1s | 0.04 | 0.04 |
| centroid_speed_1s | -0.07 | -0.07 |
| spread_1s | -0.28 | -0.25 |
| depth_std_1s | -0.12 | -0.07 |
| mean_abs_acc_1s | 0.23 | 0.17 |
| speed_p25_5s | -0.10 | -0.12 |
| speed_median_5s | -0.07 | -0.08 |
| speed_p75_5s | -0.06 | -0.07 |
| speed_p90_5s | -0.04 | -0.05 |
| frac_walk_5s | 0.12 | 0.14 |
| frac_run_5s | 0.05 | 0.04 |
| centroid_speed_5s | -0.08 | -0.08 |
| spread_5s | -0.28 | -0.26 |
| depth_std_5s | -0.12 | -0.08 |
| mean_abs_acc_5s | 0.29 | 0.22 |
| **mean abs SMD** | 0.12 | 0.11 |

Features closer after the degradation: 8 of 20.
