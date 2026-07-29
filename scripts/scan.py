"""Daily MLP-autoencoder anomaly scan — single-day CLI.

Usage:
    python scripts/scan.py [--date YYYYMMDD] [--index 000300.SH] [...]

Exit codes:
    0 = OK
    1 = panda_data interface / auth / network exception
    2 = target date has no factor data
    3 = universe is empty
    4 = column self-check failure (raised inside data.load_*)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow both `python scripts/scan.py` and `python -m scripts.scan`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import data as data_mod
from scripts import features as feat_mod
from scripts import report
from scripts import universe as uni_mod
from scripts.train import score_reconstruction, train_autoencoder

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLP Autoencoder 沪深300 异常检测")
    p.add_argument("--date", default=None,
                   help="扫描日 YYYYMMDD；默认取 get_last_trade_date")
    p.add_argument("--index", default="000300.SH",
                   help="Index symbol (default 000300.SH = CSI300)")
    p.add_argument("--lookback", type=int, default=20,
                   help="每个样本的特征窗口长度（交易日）")
    p.add_argument("--train_days", type=int, default=60,
                   help="训练窗口交易日数（T-train_days 到 T-1）")
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=str(REPO_ROOT / "output"))
    return p.parse_args()


def _resolve_scan_date(explicit: str | None) -> str:
    if explicit:
        return explicit
    got = data_mod.get_last_trade_date()
    if not got:
        print("[error] get_last_trade_date returned nothing", file=sys.stderr)
        sys.exit(2)
    return got


def _resolve_fetch_start(scan_date: str, lookback: int, train_days: int) -> str:
    """Compute the earliest date we need to fetch: T - (train_days + lookback) trading days.

    Uses get_prev_trade_date to walk back on the trading calendar. Adds a small
    safety buffer (5 extra days) so that the window truly covers `train_days`
    even if a stock has an isolated halt at the very start.
    """
    n_back = lookback + train_days + 5
    start = data_mod.get_prev_trade_date(scan_date, n=n_back)
    if not start:
        print(f"[error] get_prev_trade_date({scan_date}, n={n_back}) returned None",
              file=sys.stderr)
        sys.exit(2)
    return start


def _build_hits(
    per_sample_mse: np.ndarray,
    per_col_sq: np.ndarray,
    score_symbols: list[str],
    scan_date: str,
    factor_df: pd.DataFrame,
    post_df: pd.DataFrame,
    per_day_features: pd.DataFrame,
    feature_columns: list[str],
    lookback: int,
    train_samples: int,
) -> pd.DataFrame:
    """Assemble the hits table with per-symbol context columns."""
    if per_sample_mse.size == 0:
        return pd.DataFrame(columns=[
            "trade_date", "symbol", "name", "rank", "reconstruction_error",
            "top_feature", "ret_T", "turnover_T", "amplitude_T", "detail_json",
        ])

    n_feats = feat_mod.N_FEATURES

    # z-score of per_sample within today's cross-section (for readability).
    mu = float(per_sample_mse.mean())
    sd = float(per_sample_mse.std())
    sd_safe = sd if sd > 1e-8 else 1.0

    # Fetch name from post_df (per scan_date row) — get_factor does not return `name`.
    post_T = post_df[post_df["date"] == scan_date][["symbol", "name"]].drop_duplicates()
    name_map = dict(zip(post_T["symbol"], post_T["name"]))

    # Fetch T-day features for the ret / turnover / amplitude context columns.
    pd_T = per_day_features[per_day_features["date"] == scan_date].set_index("symbol")

    rows = []
    for i, sym in enumerate(score_symbols):
        col_sq = per_col_sq[i]  # (lookback * n_feats,)
        # Aggregate per raw-feature (sum across days) → 8 numbers.
        by_feat = col_sq.reshape(lookback, n_feats).sum(axis=0)
        top_feat_idx = int(by_feat.argmax())
        top_feat_name = feat_mod.FEATURE_NAMES[top_feat_idx]

        per_feat_dict = {
            feat_mod.FEATURE_NAMES[j]: float(by_feat[j]) for j in range(n_feats)
        }

        row_T = pd_T.loc[sym] if sym in pd_T.index else None
        ret_T = float(row_T["ret"]) if row_T is not None else float("nan")
        turnover_T = float(row_T["turnover"]) if row_T is not None else float("nan")
        amp_T = float(row_T["amplitude"]) if row_T is not None else float("nan")

        detail = {
            "per_feature_mse": per_feat_dict,
            "z_score": float((per_sample_mse[i] - mu) / sd_safe),
            "trained_samples": int(train_samples),
        }
        rows.append({
            "trade_date": scan_date,
            "symbol": sym,
            "name": name_map.get(sym, "") or "",
            "reconstruction_error": float(per_sample_mse[i]),
            "top_feature": top_feat_name,
            "ret_T": ret_T,
            "turnover_T": turnover_T,
            "amplitude_T": amp_T,
            "detail_json": json.dumps(detail, ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = _parse_args()

    # ---------------- 1. Auth & fetch ----------------
    try:
        data_mod.init_panda_data()
        scan_date = _resolve_scan_date(args.date)
        fetch_start = _resolve_fetch_start(scan_date, args.lookback, args.train_days)

        factor_df = data_mod.load_factor(fetch_start, scan_date, args.index)
        post_df = data_mod.load_stock_post(fetch_start, scan_date, args.index)
        index_df = data_mod.load_index_daily(args.index, fetch_start, scan_date)

        # Fetch weights over a small window ending on scan_date; universe.resolve_universe
        # will pick the exact-match or most-recent-prior date.
        wt_start = data_mod.get_prev_trade_date(scan_date, n=10) or fetch_start
        weights_df = data_mod.load_index_weights(args.index, scan_date)
        if weights_df.empty:
            # Widen to a small window as a fallback.
            weights_df = pd.concat([
                data_mod.load_index_weights(args.index, d)
                for d in (wt_start, scan_date)
            ], ignore_index=True)
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        # _assert_columns raised: field self-check failure.
        print(f"[error] field self-check failed: {e}", file=sys.stderr)
        return 4
    except Exception as e:
        print(f"[error] panda_data call failed: {e}", file=sys.stderr)
        return 1

    if factor_df.empty or factor_df[factor_df["date"] == scan_date].empty:
        print(f"[error] no factor data for --date {scan_date}", file=sys.stderr)
        return 2

    universe = uni_mod.resolve_universe(weights_df, scan_date)
    if not universe:
        print(f"[error] empty universe for {args.index} on {scan_date}", file=sys.stderr)
        return 3
    print(f"[info] universe: {len(universe)} stocks in {args.index} on {scan_date}",
          file=sys.stderr)

    # ---------------- 2. Feature engineering ----------------
    fbundle = feat_mod.build_features(
        factor_df=factor_df,
        post_df=post_df,
        index_df=index_df,
        universe=universe,
        scan_date=scan_date,
        lookback=args.lookback,
        train_days=args.train_days,
    )
    print(
        f"[info] samples — train: {fbundle.train_x.shape[0]}, score: {fbundle.score_x.shape[0]}",
        file=sys.stderr,
    )
    if fbundle.score_x.shape[0] == 0:
        print(f"[error] no eligible stocks on {scan_date} (all halted or missing data)",
              file=sys.stderr)
        return 3
    if fbundle.train_x.shape[0] < 100:
        print(f"[warn] only {fbundle.train_x.shape[0]} training samples — AE may underfit",
              file=sys.stderr)

    # ---------------- 3. Train ----------------
    result = train_autoencoder(
        fbundle.train_x,
        input_dim=fbundle.train_x.shape[1],
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(
        f"[info] trained {result.n_epochs_ran} epochs on {result.device}; "
        f"train_loss={result.final_train_loss:.4f}, val_loss={result.final_val_loss:.4f}",
        file=sys.stderr,
    )

    # ---------------- 4. Score ----------------
    per_sample, per_col_sq = score_reconstruction(result.model, fbundle.score_x)

    # We need per-day features for the T-day context columns; recompute from the merged frame.
    merged = factor_df.merge(
        post_df[["symbol", "date", "pre_close", "limit_up", "limit_down", "trade_status"]],
        on=["symbol", "date"],
        how="inner",
        validate="one_to_one",
    )
    per_day = feat_mod._compute_daily_features(merged, index_df)

    hits = _build_hits(
        per_sample_mse=per_sample,
        per_col_sq=per_col_sq,
        score_symbols=fbundle.score_symbols,
        scan_date=scan_date,
        factor_df=factor_df,
        post_df=post_df,
        per_day_features=per_day,
        feature_columns=fbundle.feature_columns,
        lookback=args.lookback,
        train_samples=fbundle.train_x.shape[0],
    )

    # ---------------- 5. Write ----------------
    train_start = min(fbundle.train_dates) if fbundle.train_dates else "?"
    train_end = max(fbundle.train_dates) if fbundle.train_dates else "?"
    meta = {
        "index": args.index,
        "universe_size": len(universe),
        "score_size": len(fbundle.score_symbols),
        "train_start": train_start,
        "train_end": train_end,
        "train_days": args.train_days,
        "train_samples": fbundle.train_x.shape[0],
        "epochs_ran": result.n_epochs_ran,
        "val_loss": result.final_val_loss,
        "device": result.device,
        "error_mean": float(per_sample.mean()),
        "error_std": float(per_sample.std()),
        "error_max": float(per_sample.max()),
    }
    out_dir = Path(args.output_dir)
    csv_path = out_dir / f"anomaly_{scan_date}.csv"
    md_path = out_dir / f"anomaly_{scan_date}.md"
    report.write_csv(hits, str(csv_path), top_n=args.top_n)
    report.write_markdown(hits, str(md_path), date=scan_date, top_n=args.top_n, meta=meta)

    print(f"[ok] wrote {csv_path} ({min(args.top_n, len(hits))} rows)")
    print(f"[ok] wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
