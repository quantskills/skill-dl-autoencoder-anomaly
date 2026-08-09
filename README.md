# skill-dl-autoencoder-anomaly

Claude Code skill for a daily post-market unsupervised anomaly detection on CSI300 constituents. A small MLP autoencoder is retrained every run on the trailing 60 trading days and scores today's 20-day windows by reconstruction error. See `SKILL.md` for full usage. Design lives in `docs/superpowers/specs/2026-07-30-dl-autoencoder-anomaly-design.md`; implementation plan in `docs/superpowers/plans/2026-07-30-dl-autoencoder-anomaly.md`.

## Runtime and dependencies

The validated runtime is CPython 3.10 in the `pandaai` environment:

| Component | Validated version |
|---|---:|
| Python | 3.10.20 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| torch | 2.13.0 |
| panda_data | 0.0.12 |
| pytest | 9.0.2 |

`panda_data` is a private package and still requires configured credentials for live data.
The exact top-level dependencies are declared in `requirements.txt`.

## Quick start

```bash
cd /Users/since/Code/quantskills/skill-dl-autoencoder-anomaly
PYTHON_BIN=/opt/miniconda3/envs/pandaai/bin/python
$PYTHON_BIN -m pip install -r requirements.txt
$PYTHON_BIN -m scripts.check_env                         # dependency/import/device check
$PYTHON_BIN -m pytest tests/ -q                          # unit + offline end-to-end
```

The offline end-to-end test runs the actual feature pipeline, PyTorch training, scoring,
and CSV/Markdown writers with deterministic synthetic data; it needs no network or credentials.
Run it alone with:

```bash
$PYTHON_BIN -m pytest tests/test_scan_e2e.py -v
```

For a live panda_data scan, first source credentials and run the field self-check:

```bash
set -a && source ~/.zshrc >/dev/null 2>&1 && set +a
$PYTHON_BIN -m scripts.check_env
$PYTHON_BIN -m scripts.data --self-check --date 20260729
$PYTHON_BIN scripts/scan.py --date 20260729 --seed 42
test -s output/anomaly_20260729.csv
test -s output/anomaly_20260729.md
```

Outputs land in `output/anomaly_YYYYMMDD.csv` + `.md`. The complete verification runbook is
in [`docs/verification.md`](docs/verification.md).
