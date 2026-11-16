# branca analytics: Ball Possession (More than a Hack 2027, Phase 1)

For every frame of the tactical video of BAR-ATH, one label: `H` / `A` (team in possession),
`D` (dead ball) or `X` (not graded). Rule-based pipeline plus one small stopped-ball model trained on other matches (public Metrica sample
games, see below); no hand labels, no BAR-ATH labels.
Expected SHA-256 of `submissions/branca analytics.csv`:
`1caa369c9258f02ca7337e75b5256b300205a476e1cab666e29187f9326947be`.
The challenge statement is in [README_official.md](README_official.md) and the other official
documents; the technical document is [docs/branca analytics.md](docs/branca%20analytics.md).

## Install

Python 3.12 (developed on 3.12.3), exact versions pinned in `requirements.txt`
(`numpy==2.5.4`, `pandas==3.1.1`).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # tests: pip install -r requirements-dev.txt (adds pytest==9.2.0)
```

## Data

The match data is licensed for this competition only and is **not** in this repository
(`data/` is git-ignored). Unpack the organisers' bundle keeping the names it arrives with:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_smart_tagging.xml
(data/tactical/BAR-ATH_tactical_homography.csv and the videos are not used by the pipeline)
```

or point `MTAH_DATA_DIR` at the folder that contains `tactical/`. See [data/README.md](data/README.md).
The first run parses the 200 MB tracking file (about 20 s) and caches arrays in `data/cache/`.

## Regenerate the submission (one command)

```bash
python -m src.run --out "submissions/branca analytics.csv"
python evaluation/validate_submission.py "submissions/branca analytics.csv"
```

The output is deterministic: two runs produce byte-identical files (`tests/test_pipeline.py`).
`--version v1|v2|v3|v4` select the pipeline generation (see `docs/`); the default, v4, is the one
that produced the submitted file (`--version v1` reproduces the earlier safety submission). v3 also
reads `data/broadcast/`. v4 uses the stopped-ball model `models/dead_model.json`, trained on the
public Metrica sample games (regenerate: clone https://github.com/metrica-sports/sample-data into
`data_external/sample-data`, then `python scripts/train_dead_model.py`).

## Layout

| path | what |
|---|---|
| `src/io.py`, `src/ball.py` | tracking and Smart Tagging reading, ball cleaning |
| `src/features.py` | activity, ball out, nearest players, phase team |
| `src/dead_ball.py` | dead-ball rules R1-R6 |
| `src/possession.py`, `src/postprocess.py` | H/A on live frames, clean-up |
| `src/run.py` | single entry point |
| `scripts/` | diagnostics and reports (`docs/diagnostics_e2.md`, `docs/report_v*.md`) |
| `tests/` | `python -m pytest` |
| `evaluation/` | official scorer (public 5 minutes, only used to report) |

## External data and acknowledgement

The stopped-ball model (`models/dead_model.json`) was trained, and the possession parameters were
validated, on the public sample games of **Metrica Sports**,
https://github.com/metrica-sports/sample-data (Sample_Game_1, 2 and 3; tracking and events by Metrica
Sports, anonymised). We thank Metrica Sports for releasing them. They are not redistributed here
(`data_external/` is git-ignored); `python scripts/train_dead_model.py` regenerates the model after
cloning that repository into `data_external/sample-data`. The pipeline itself does not need them.

## Rules followed

No hand labels or corrections, no constants tied to frames, the public ground truth is only
reported after the pipeline is fixed, and no match data is committed.
