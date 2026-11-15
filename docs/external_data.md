# External validation data: Metrica sample games

Source: https://github.com/metrica-sports/sample-data (Sample_Game_1, 2, 3), downloaded to `data_external/` (git-ignored, not redistributed). Tracking and events were produced and corrected by Metrica Sports; nothing was labelled by hand by us. None of this data is from BAR-ATH and none of the BAR-ATH data is used to choose anything in this stage.

## Label rules

- **D**: the ball is absent from the corrected tracking, inside a period (the tracking of these games is corrected: the ball exists exactly while it is in play).
- **H / A**: the team of the latest ball event of the live spell. Ball events: PASS, SHOT, RECOVERY, SET PIECE and (game 3 only) CARRY. BALL LOST, CHALLENGE, FAULT RECEIVED, CARD and BALL OUT never change the owner, so the owner changes only when the opponent has a ball event of its own. Frames of a spell before its first event take that event's team.
- **X**: outside the periods, and the first frame of the second period (so the official scorer does not join the halves).

## Per game

| game | graded frames | possessions | mean duration (s) | % H | % A | % D | D runs | D runs >= 2 s | mean (s) | median (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Sample_Game_1 | 145005 | 357 | 9.9 | 30.2 | 30.7 | 39.1 | 156 | 96 | 23.5 | 19.7 |
| Sample_Game_2 | 141155 | 329 | 10.1 | 30.5 | 28.5 | 41.0 | 170 | 100 | 22.9 | 16.1 |
| Sample_Game_3 | 143760 | 384 | 10.1 | 34.8 | 33.0 | 32.2 | 161 | 102 | 18.0 | 12.8 |

## Coherence of D (ball absent) with the provider's events

If the ball is absent exactly while it is dead, ball events should find the ball present, and stoppage events should be followed by an absent ball.

| game | % frames with ball during PASS/SHOT/CARRY | % ball absent 1-3 s after BALL OUT | % ball absent 1-3 s before a restart (corner, free kick, throw-in...) | % ball present 0.4-2 s after the restart | % ball absent 0-2 s after FAULT RECEIVED | D runs >= 2 s with a stoppage event within 6 s |
|---|---|---|---|---|---|---|
| Sample_Game_1 | 100.0 (820 events) | 46.0 (51) | 96.8 (71) | 97.7 (71) | 93.6 (22) | 89 of 96 |
| Sample_Game_2 | 100.0 (988 events) | 61.4 (49) | 95.2 (73) | 94.0 (73) | 92.1 (20) | 89 of 100 |
| Sample_Game_3 | 100.0 (2535 events) | 67.2 (64) | 98.5 (86) | 97.5 (86) | 99.7 (12) | 94 of 102 |

## Official metrics on these labels (starter baseline logic)

Nearest player to the ball, majority vote of +/- w frames, D where the ball is absent (`starter/baseline.py` functions), scored with `evaluation/possession_metrics.py`. On BAR-ATH the public window gave spellF1 0.04-0.42 and macroF1 0.58-0.62 for the same four windows.

| game | baseline | macroF1 | spellF1 | predicted possessions | true possessions | mean duration (s) |
|---|---|---|---|---|---|---|
| Sample_Game_1 | raw | 0.8754 | 0.1339 | 2023 | 357 | 1.7 |
| Sample_Game_1 | smooth25 | 0.8742 | 0.4190 | 635 | 357 | 5.9 |
| Sample_Game_1 | smooth50 | 0.8586 | 0.4605 | 482 | 357 | 8.1 |
| Sample_Game_1 | smooth125 | 0.7732 | 0.3170 | 385 | 357 | 11.4 |
| Sample_Game_2 | raw | 0.8789 | 0.1106 | 2178 | 329 | 1.5 |
| Sample_Game_2 | smooth25 | 0.8814 | 0.3941 | 617 | 329 | 5.8 |
| Sample_Game_2 | smooth50 | 0.8647 | 0.4393 | 464 | 329 | 8.1 |
| Sample_Game_2 | smooth125 | 0.7832 | 0.3014 | 369 | 329 | 11.4 |
| Sample_Game_3 | raw | 0.8583 | 0.0949 | 2572 | 384 | 1.5 |
| Sample_Game_3 | smooth25 | 0.8635 | 0.3607 | 787 | 384 | 5.2 |
| Sample_Game_3 | smooth50 | 0.8422 | 0.4072 | 559 | 384 | 7.7 |
| Sample_Game_3 | smooth125 | 0.7510 | 0.3254 | 377 | 384 | 12.6 |

## Differences from the CHALLENGE.md definitions

- BAR-ATH labels come from corrected **tracking** (D where the corrected tracking marks the ball dead; otherwise the team the tracking names as owner). Here D is the same idea (ball absent in the corrected tracking) but H/A comes from **events**: the team of the latest ball event. A change of possession here is the opponent's first ball event (a pass, shot, recovery or carry), which is a touch, not necessarily the 'controlled possession' of CHALLENGE.md; a recovery that is immediately lost counts as a change.
- In BAR-ATH the ground truth names an owner during loose-ball phases through the tracking; here the owner is held until the opponent has an event, the same stay-with-the-last-team idea.
- Event timestamps are the touch times; the tracking ball can reappear a few frames before or after them, so spell boundaries here carry an error of a few frames.
- Pitch: 105 x 68 m (BAR-ATH: 105.3 x 68). Tracking is corrected and has no per-player speed, so player speeds are computed from positions (as in the pipeline). Corrected tracking is smoother than the automatic BAR-ATH tracking: speed-based thresholds may not transfer identically.
- No Smart Tagging exists for these games: no Team tag at restarts, no goal/set-piece anchors.
