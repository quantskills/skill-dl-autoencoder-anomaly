# Verification Runbook

This document defines the reproducible gates for the full anomaly-detection chain:
dependency check → data contract check → feature windows → model training → scoring →
CSV/Markdown output.

## 1. Install the validated dependencies

Use CPython 3.10. The repository was verified with the `pandaai` environment and the
versions pinned in `requirements.txt`.

```bash
cd /Users/since/Code/quantskills/skill-dl-autoencoder-anomaly
PYTHON_BIN=/opt/miniconda3/envs/pandaai/bin/python
$PYTHON_BIN -m pip install -r requirements.txt
$PYTHON_BIN -m scripts.check_env
```

`check_env` is network-free. It verifies Python/distribution versions, imports, the
available torch device, and only whether the two credential variables are set; it never
prints credential values.

Expected final line:

```text
[ok] runtime dependencies and imports verified
```

## 2. Verify the complete chain offline

This test mocks only the remote panda_data boundary. It runs the real `scan.main` path,
feature engineering, PyTorch training, reconstruction scoring, and both output writers.
It proves the chain without requiring credentials or network access.

```bash
$PYTHON_BIN -m pytest tests/test_scan_e2e.py -v
```

Expected result:

```text
1 passed
```

The test asserts that a non-empty `anomaly_<date>.csv` and `anomaly_<date>.md` are created,
that ranks are sequential, and that the key score/feature columns contain no missing values.

## 3. Verify the live data contract and scan

Source credentials into the non-interactive shell, then run the field check before training:

```bash
set -a && source ~/.zshrc >/dev/null 2>&1 && set +a
$PYTHON_BIN -m scripts.check_env
$PYTHON_BIN -m scripts.data --self-check --date 20260729
$PYTHON_BIN scripts/scan.py --date 20260729 --seed 42
```

Acceptance checks:

```bash
test -s output/anomaly_20260729.csv
test -s output/anomaly_20260729.md
$PYTHON_BIN - <<'PY'
import pandas as pd
from pathlib import Path

path = Path("output/anomaly_20260729.csv")
df = pd.read_csv(path)
required = {"symbol", "reconstruction_error", "top_feature", "detail_json"}
assert required <= set(df.columns)
assert df["reconstruction_error"].notna().all()
print(f"[ok] verified {len(df)} output rows in {path}")
PY
```

Exit codes for the live CLI are documented in `SKILL.md`: `0` success, `1` panda_data/auth
or network error, `2` no factor data, `3` empty universe/eligible score set, and `4` field
contract failure.

## 4. Reproducibility

The scan retrains from scratch on each run. Keep `--seed 42` and the same input date. The
offline test is deterministic and the live workflow should be compared by output rank/order
after two runs; device differences (MPS vs CPU) can cause small floating-point score changes.
