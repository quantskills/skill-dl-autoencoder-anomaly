"""Feature engineering for the autoencoder anomaly skill.

Given the merged daily panel (stock OHLCV + turnover + limits + trade_status) and
the benchmark index daily series, produce two tensors:

  - training_matrix: rows are (symbol, t) samples for every t in [T-train_days, T-1]
    where a full 20-day history ending at t is available. Shape (N_train, lookback*8).
  - scoring_matrix:  one row per surviving symbol on scan day T.
                     Shape (N_score, lookback*8).

Feature columns per day (order = FEATURE_NAMES):
    ret, log_vol, amplitude, turnover, gap, dist_limit_up, dist_limit_down, excess_ret

Design invariants (§5):
  - `date < T` is strictly enforced when building the training set.
  - Standardization mean/std are estimated on training samples only and reused
    (never re-fit) on the scoring row — no leakage of T-day statistics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_NAMES: list[str] = [
    "ret",
    "log_vol",
    "amplitude",
    "turnover",
    "gap",
    "dist_limit_up",
    "dist_limit_down",
    "excess_ret",
]
N_FEATURES: int = len(FEATURE_NAMES)


@dataclass
class FeatureBundle:
    """Container for the outputs of build_features()."""

    train_x: np.ndarray            # (N_train, lookback * N_FEATURES), z-scored
    score_x: np.ndarray            # (N_score, lookback * N_FEATURES), z-scored
    score_symbols: list[str]       # ordered symbols aligned with score_x rows
    train_symbols: list[str]       # ordered symbols aligned with train_x rows
    train_dates: list[str]         # date of the LAST day in each training window
    feat_mean: np.ndarray          # (lookback * N_FEATURES,) column means from training
    feat_std: np.ndarray           # (lookback * N_FEATURES,) column stds from training
    lookback: int
    n_features: int                # = N_FEATURES

    @property
    def feature_columns(self) -> list[str]:
        """Flat column names (feat_i where i = day_offset*N_FEATURES + feat_index)."""
        cols = []
        for day in range(self.lookback):
            for name in FEATURE_NAMES:
                cols.append(f"d{day}_{name}")
        return cols


def _compute_daily_features(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a per-(symbol,date) frame with FEATURE_NAMES columns.

    Args:
        stock_df: merged frame with columns
            [symbol, date, open, close, high, low, volume, turnover,
             pre_close, limit_up, limit_down, trade_status].
        index_df: benchmark index frame with columns [date, close, pre_close].
                  ret_index = close/pre_close - 1.

    Returns:
        DataFrame with columns [symbol, date, trade_status, <FEATURE_NAMES>].
    """
    df = stock_df.copy()

    # Cast defensively.
    for c in ("open", "close", "high", "low", "volume",
              "turnover", "pre_close", "limit_up", "limit_down"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Guard against pre_close == 0 (extremely rare but breaks division).
    pc = df["pre_close"].where(df["pre_close"] > 0, np.nan)
    df["ret"] = df["close"] / pc - 1.0
    df["log_vol"] = np.log(df["volume"].clip(lower=0.0) + 1.0)
    df["amplitude"] = (df["high"] - df["low"]) / pc
    df["gap"] = df["open"] / pc - 1.0
    df["dist_limit_up"] = (df["limit_up"] - df["close"]) / df["close"].where(df["close"] > 0, np.nan)
    df["dist_limit_down"] = (df["close"] - df["limit_down"]) / df["close"].where(
        df["close"] > 0, np.nan
    )
    # turnover: already a column of the merged frame; keep as-is.

    # Merge index return; index_df has one row per date.
    idx = index_df[["date", "close", "pre_close"]].copy()
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    idx["pre_close"] = pd.to_numeric(idx["pre_close"], errors="coerce")
    idx_pc = idx["pre_close"].where(idx["pre_close"] > 0, np.nan)
    idx["index_ret"] = idx["close"] / idx_pc - 1.0
    idx = idx[["date", "index_ret"]]

    df = df.merge(idx, on="date", how="left")
    df["excess_ret"] = df["ret"] - df["index_ret"]

    keep = ["symbol", "date", "trade_status"] + FEATURE_NAMES
    return df[keep].sort_values(["symbol", "date"]).reset_index(drop=True)


def _stack_windows(
    per_day: pd.DataFrame,
    scan_date: str,
    lookback: int,
    train_days: int,
    max_missing_in_train: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Slice per_day into fixed-length windows.

    Returns:
        train_x_raw: (N_train, lookback * N_FEATURES)
        score_x_raw: (N_score, lookback * N_FEATURES)
        train_symbols: symbol per training row
        train_dates:   last-day-of-window date per training row
        score_symbols: symbol per scoring row (sorted)
    """
    train_rows: list[np.ndarray] = []
    train_symbols: list[str] = []
    train_dates: list[str] = []
    score_rows: list[np.ndarray] = []
    score_symbols: list[str] = []

    for symbol, g in per_day.groupby("symbol", sort=True):
        g_sorted = g.sort_values("date").reset_index(drop=True)

        # -------- Training samples (strictly date < scan_date) --------
        past = g_sorted[g_sorted["date"] < scan_date].reset_index(drop=True)
        # Take only the last `train_days` trading days of `past` — the window over which
        # we intend to sample. This bounds `max_missing_in_train` sensibly.
        past = past.tail(train_days).reset_index(drop=True)

        # Suspension budget: >5 halted days ⇒ drop this symbol from training entirely.
        halted_in_train = int((past["trade_status"] != 0).sum())
        include_in_train = halted_in_train <= max_missing_in_train

        if include_in_train and len(past) >= lookback:
            values = past[FEATURE_NAMES].to_numpy(dtype=np.float64)
            dates = past["date"].tolist()
            # A window is valid iff all lookback rows are non-NaN and non-halted.
            statuses = past["trade_status"].to_numpy()
            for end_idx in range(lookback - 1, len(past)):
                start_idx = end_idx - lookback + 1
                window = values[start_idx : end_idx + 1]
                window_stat = statuses[start_idx : end_idx + 1]
                if np.isnan(window).any():
                    continue
                if (window_stat != 0).any():
                    continue
                train_rows.append(window.reshape(-1))
                train_symbols.append(str(symbol))
                train_dates.append(str(dates[end_idx]))

        # -------- Scoring sample (window ENDING at scan_date, inclusive) --------
        recent = g_sorted[g_sorted["date"] <= scan_date].reset_index(drop=True)
        if len(recent) < lookback:
            continue
        tail = recent.tail(lookback).reset_index(drop=True)
        # Scoring window's last row must be scan_date itself; otherwise the stock
        # didn't trade on T and should not appear on the leaderboard.
        if str(tail["date"].iloc[-1]) != scan_date:
            continue
        values = tail[FEATURE_NAMES].to_numpy(dtype=np.float64)
        statuses = tail["trade_status"].to_numpy()
        if np.isnan(values).any():
            continue
        if (statuses != 0).any():
            continue
        score_rows.append(values.reshape(-1))
        score_symbols.append(str(symbol))

    train_x = np.stack(train_rows, axis=0) if train_rows else np.zeros(
        (0, lookback * N_FEATURES), dtype=np.float64
    )
    score_x = np.stack(score_rows, axis=0) if score_rows else np.zeros(
        (0, lookback * N_FEATURES), dtype=np.float64
    )
    return train_x, score_x, train_symbols, train_dates, score_symbols


def build_features(
    factor_df: pd.DataFrame,
    post_df: pd.DataFrame,
    index_df: pd.DataFrame,
    universe: list[str],
    scan_date: str,
    lookback: int = 20,
    train_days: int = 60,
) -> FeatureBundle:
    """Full feature pipeline. See module docstring.

    Args:
        factor_df: from data.load_factor — carries OHLCV + turnover + market_cap.
        post_df:   from data.load_stock_post — carries pre_close + limit_up/down + trade_status.
        index_df:  from data.load_index_daily — benchmark for excess_ret.
        universe:  list of stock_symbols scoped to CSI300 constituents on `scan_date`.
        scan_date: YYYYMMDD.
        lookback:  window length in trading days (default 20).
        train_days: trailing trading-day pool from which to draw training windows.

    Returns:
        FeatureBundle with z-scored train_x / score_x and the fit-time mean/std.
    """
    # 1. Restrict to universe.
    uni_set = set(universe)
    f = factor_df[factor_df["symbol"].isin(uni_set)].copy()
    p = post_df[post_df["symbol"].isin(uni_set)].copy()

    # 2. Join factor + post on (symbol, date).
    merged = f.merge(
        p[["symbol", "date", "pre_close", "limit_up", "limit_down", "trade_status"]],
        on=["symbol", "date"],
        how="inner",
        validate="one_to_one",
    )

    # 3. Compute daily features.
    per_day = _compute_daily_features(merged, index_df)

    # 4. Stack into windows.
    train_x, score_x, train_syms, train_dates, score_syms = _stack_windows(
        per_day, scan_date=scan_date, lookback=lookback, train_days=train_days,
    )

    # 5. z-score with train-only statistics; guard std against zero.
    if train_x.shape[0] == 0:
        feat_mean = np.zeros(lookback * N_FEATURES, dtype=np.float64)
        feat_std = np.ones(lookback * N_FEATURES, dtype=np.float64)
    else:
        feat_mean = train_x.mean(axis=0)
        feat_std = train_x.std(axis=0)
        feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)

    train_z = (train_x - feat_mean) / feat_std if train_x.shape[0] else train_x
    score_z = (score_x - feat_mean) / feat_std if score_x.shape[0] else score_x

    return FeatureBundle(
        train_x=train_z.astype(np.float32),
        score_x=score_z.astype(np.float32),
        score_symbols=score_syms,
        train_symbols=train_syms,
        train_dates=train_dates,
        feat_mean=feat_mean,
        feat_std=feat_std,
        lookback=lookback,
        n_features=N_FEATURES,
    )
