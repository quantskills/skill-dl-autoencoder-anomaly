# skill-dl-autoencoder-anomaly

Claude Code skill for a daily post-market unsupervised anomaly detection on CSI300 constituents. A small MLP autoencoder is retrained every run on the trailing 60 trading days and scores today's 20-day windows by reconstruction error. See `SKILL.md` for full usage. Design lives in `docs/superpowers/specs/2026-07-30-dl-autoencoder-anomaly-design.md`; implementation plan in `docs/superpowers/plans/2026-07-30-dl-autoencoder-anomaly.md`.

## Quick start

```bash
conda activate pandaai
pip install -r requirements.txt
export PANDA_DATA_USERNAME=...
export PANDA_DATA_PASSWORD=...
pytest tests/                                          # unit tests
python -m scripts.data --self-check --date 20260729    # field self-check
python scripts/scan.py --date 20260729                 # single-day scan
```

Outputs land in `output/anomaly_YYYYMMDD.csv` + `.md`.
