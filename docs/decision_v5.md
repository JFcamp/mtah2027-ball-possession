# Decision v5

Written BEFORE the public ground truth was looked at; every number below comes from the external games or from algorithmic statistics of BAR-ATH (rules R1-R6, Smart Tagging). The public window appears only in `docs/report_v5.md`, after this file was committed.

## Part A: degradation

See `docs/domain_gap.md`. The degraded external games reproduce the measured BAR-ATH statistics (tracked players 20.8-21.0 mean vs 20.5, second differences of players 1.3-1.6 mm vs 1.1, ball 3.6-3.9 mm vs 4.8, ball erased in 10.6% of the live frames). The model features were already close (mean absolute standardised difference 0.116 for the clean games, 0.108 after degrading), so the degradation works as a robustness stress test more than as a correction of a large gap. Wrong-ball episodes are not simulated.

## Part B: possession configuration (true dead-ball mask)

v4 configuration: r 1.0 m 0.3 min 6 2touch on backdate off poss 25. Spell F1 on the test game, chosen on the other two:

| test game | chosen on the other two | DEG_BAR (chosen vs v4) | DEG_40 | CLEAN (reported) | beats v4 in both |
|---|---|---|---|---|---|
| Sample_Game_1 | r 1.0 m 0.0 min 25 2touch on backdate on poss 12 | 0.6951 vs 0.7001 | 0.6067 vs 0.6132 | 0.7728 vs 0.8250 | FAIL |
| Sample_Game_2 | r 1.0 m 0.3 min 12 2touch on backdate on poss 12 | 0.7300 vs 0.7028 | 0.6476 vs 0.6130 | 0.8134 vs 0.7935 | PASS |
| Sample_Game_3 | r 1.0 m 0.3 min 25 2touch on backdate on poss 12 | 0.6795 vs 0.6430 | 0.5863 vs 0.5516 | 0.7573 vs 0.7560 | PASS |

**Criterion B: FAIL.** Best configuration on the three games: r 1.0 m 0.3 min 25 2touch on backdate on poss 12 (objective 0.6627 vs 0.6373 for v4).

## Part C: stopped-ball model (leave-one-game-out)

| test game | model | AUC degraded | bal. acc. degraded | AUC clean | bal. acc. clean |
|---|---|---|---|---|---|
| Sample_Game_1 | L0 | 0.9125 | 0.8557 | 0.9169 | 0.8595 |
| Sample_Game_1 | L1 | 0.9134 | 0.8670 | 0.9146 | 0.8686 |
| Sample_Game_1 | G1 | 0.9187 | 0.8755 | 0.9203 | 0.8748 |
| Sample_Game_2 | L0 | 0.9463 | 0.8885 | 0.9505 | 0.8933 |
| Sample_Game_2 | L1 | 0.9511 | 0.8931 | 0.9526 | 0.8977 |
| Sample_Game_2 | G1 | 0.9554 | 0.8966 | 0.9558 | 0.8960 |
| Sample_Game_3 | L0 | 0.9341 | 0.8565 | 0.9504 | 0.8768 |
| Sample_Game_3 | L1 | 0.9579 | 0.8621 | 0.9606 | 0.8688 |
| Sample_Game_3 | G1 | 0.9566 | 0.8899 | 0.9592 | 0.8967 |

Mean degraded AUC: L0 0.9310, L1 0.9408, G1 0.9436. Passed: L1 PASS, G1 PASS. **Criterion C: PASS; chosen model: G1.**

## Part C2: decoder and end-to-end check

Chosen on the two training games (nested), scored on the third with the complete pipeline; v4-like = L0, P>=0.5, minimum 50 frames, v4 configuration.

| test game | decoder chosen on the other two | DEG_BAR (new vs v4-like) | DEG_40 | beats v4-like |
|---|---|---|---|---|
| Sample_Game_1 | P>=0.6 min 50 close 0 refine off | 0.5473 vs 0.5236 | 0.4459 vs 0.4309 | PASS |
| Sample_Game_2 | P>=0.6 min 100 close 0 refine off | 0.4937 vs 0.4941 | 0.3919 vs 0.4035 | FAIL |
| Sample_Game_3 | P>=0.5 min 50 close 0 refine on | 0.4792 vs 0.4796 | 0.3654 vs 0.3763 | FAIL |

**Criterion E2E: FAIL.** Decoder chosen on the three games: P>=0.5 min 75 close 50 refine off.

## Part D: model on every graded frame of BAR-ATH

Model used: L0. Algorithmic weak labels: 33951 weak-stopped frames, 24724 weak-live frames (3552 inside BUILD UP).

| T1 transfer | value | limit |  |
|---|---|---|---|
| AUC on the weak labels | 0.928 | >= 0.85 | PASS |
| mean hit rate over the weak-stopped sources | 0.670 | >= 0.70 | FAIL |
| false stopped in weak live | 0.016 | <= 0.20 | PASS |
| false stopped in BUILD UP (reported) | 0.053 | - | - |

| weak-stopped source | hit rate |
|---|---|
| R1 half opening | 0.429 |
| R2 ball outside | 0.738 |
| R3 set-piece core | 0.743 |
| R4 goal to kick-off | 0.771 |

The v4 model (L0) under the same test: AUC 0.928, mean hit 0.670, false stopped 0.016 (FAIL).

For information only (the gradient-boosting model G1 was not used because the end-to-end check failed): AUC 0.973, mean hit 0.757, false stopped 0.025, BUILD UP 0.079.

**Criterion D: FAIL** (T1 FAIL, T2 not evaluated, T3 not evaluated; T2 and T3 are evaluated only if T1 passes).

## Part V: pass-chain veto

Not evaluated (it only applies if D passed).

## Composition of v5

- possession configuration: r 1.0 m 0.3 min 6 2touch on backdate off poss 25
- model: L0 (`models/dead_model.json, the v4 model`), decoder: P>=0.5 min 50 close 0 refine off
- model applied on: ball gaps of 5 s or more (as v4)
- pass-chain veto: no
- **v5 differs from v4: no (v4 stays final)**

## Notes

- The gradient-boosting model passed Part C and the transfer test T1 (information row above), but the end-to-end check, a trava of the external set, failed in two of the three held-out games (DEG_BAR within 0.0004, DEG_40 lower by 0.011), so by the rule fixed beforehand the v4 configuration, model and decoder stay. The rule was not relaxed.
- Part B failed because the configuration chosen on games 2 and 3 lost to v4 on game 1.
- Part D could not pass with the v4 model: it misses the opening of the halves (hit rate 0.43) and its mean hit rate over the sources is 0.67 (< 0.70).

## BAR-ATH without ground truth

|  | D share % | possessions | mean possession (s) | stoppages >= 2 s | mean stoppage (s) |
|---|---|---|---|---|---|
| v4 | 41.5 | 353 | 10.5 | 128 | 20.6 |
| v5 | 41.5 | 353 | 10.5 | 128 | 20.6 |
