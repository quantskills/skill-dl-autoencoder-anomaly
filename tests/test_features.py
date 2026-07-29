"""Unit tests for scripts.features.

Covers:
  - Correct output shape of the flattened window (lookback * 8).
  - No-lookahead: no training row includes the scan-day itself.
  - Amplitude formula.
  - Sign conventions on distance-to-limit.
  - Excess return zero when stock and index move identically.
  - Standardization: fit-on-train, applied-on-score (no re-fit on T).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import features
from scripts.features import FEATURE_NAMES, N_FEATURES, build_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dates(start: str, n: int) -> list[str]:
    """Build n consecutive YYYYMMDD strings starting at `start` (calendar days ok for tests)."""
    ds = pd.date_range(start=pd.to_datetime(start), periods=n, freq="D")
    return [d.strftime("%Y%m%d") for d in ds]


def _factor_frame(symbol: str, dates: list[str], base_price: float = 10.0) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(dates):
        close = base_price + 0.05 * i
        rows.append({
            "date": d,
            "symbol": symbol,
            "name": "TEST",
            "open": close - 0.01,
            "close": close,
            "high": close + 0.02,
            "low": close - 0.03,
            "volume": 1e6 + 1000 * i,
            "amount": close * (1e6 + 1000 * i),
            "turnover": 0.5 + 0.001 * i,
            "market_cap": 1e10,
        })
    return pd.DataFrame(rows)


def _post_frame(symbol: str, dates: list[str], base_price: float = 10.0) -> pd.DataFrame:
    rows = []
    prev_close = base_price - 0.05
    for i, d in enumerate(dates):
        close = base_price + 0.05 * i
        rows.append({
            "date": d,
            "symbol": symbol,
            "pre_close": prev_close,
            "limit_up": prev_close * 1.10,
            "limit_down": prev_close * 0.90,
            "trade_status": 0,
        })
        prev_close = close
    return pd.DataFrame(rows)


def _index_frame(dates: list[str]) -> pd.DataFrame:
    rows = []
    prev = 4000.0
    for i, d in enumerate(dates):
        cur = 4000.0 + 2.0 * i
        rows.append({
            "symbol": "000300.SH", "date": d,
            "close": cur, "pre_close": prev,
        })
        prev = cur
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_feature_shape_and_columns():
    dates = _make_dates("20260501", 30)
    scan = dates[-1]
    fbundle = build_features(
        factor_df=_factor_frame("000001.SZ", dates),
        post_df=_post_frame("000001.SZ", dates),
        index_df=_index_frame(dates),
        universe=["000001.SZ"],
        scan_date=scan,
        lookback=20,
        train_days=25,
    )
    assert fbundle.score_x.shape == (1, 20 * N_FEATURES)
    # 25 trailing pre-T rows, need 20 for one window → 6 training samples (dates 20..25).
    assert fbundle.train_x.shape == (6, 20 * N_FEATURES)
    assert fbundle.feature_columns[0] == "d0_ret"
    assert fbundle.feature_columns[-1] == f"d19_{FEATURE_NAMES[-1]}"


def test_no_lookahead_scan_date_absent_from_training_windows():
    """No training window may end on the scan date."""
    dates = _make_dates("20260501", 40)
    scan = dates[-1]
    fbundle = build_features(
        factor_df=_factor_frame("000001.SZ", dates),
        post_df=_post_frame("000001.SZ", dates),
        index_df=_index_frame(dates),
        universe=["000001.SZ"],
        scan_date=scan,
        lookback=20,
        train_days=30,
    )
    # Every training window's last date is strictly < scan date.
    assert all(d < scan for d in fbundle.train_dates)
    # Scoring window ends exactly on scan.
    assert fbundle.score_x.shape[0] == 1


def test_amplitude_formula():
    """amplitude = (high - low) / pre_close."""
    # 30 days, all trivial; check via internal daily-features computation.
    dates = _make_dates("20260501", 5)
    # Force known numbers: pre_close=10, high=12, low=8 ⇒ amplitude = 0.4.
    f = pd.DataFrame([{
        "date": dates[0], "symbol": "AAA", "name": "T",
        "open": 10.0, "close": 11.0, "high": 12.0, "low": 8.0,
        "volume": 1.0, "amount": 11.0, "turnover": 0.1, "market_cap": 1e9,
    }])
    p = pd.DataFrame([{
        "date": dates[0], "symbol": "AAA", "pre_close": 10.0,
        "limit_up": 11.0, "limit_down": 9.0, "trade_status": 0,
    }])
    idx = pd.DataFrame([{"symbol": "000300.SH", "date": dates[0],
                         "close": 4000.0, "pre_close": 4000.0}])
    merged = f.merge(
        p[["symbol", "date", "pre_close", "limit_up", "limit_down", "trade_status"]],
        on=["symbol", "date"], how="inner",
    )
    per_day = features._compute_daily_features(merged, idx)
    assert per_day["amplitude"].iloc[0] == pytest.approx((12 - 8) / 10.0)


def test_dist_to_limits_positive_when_price_inside_band():
    """close between limit_down and limit_up ⇒ both distances ≥ 0."""
    dates = _make_dates("20260501", 1)
    f = pd.DataFrame([{
        "date": dates[0], "symbol": "AAA", "name": "T",
        "open": 10.0, "close": 10.5, "high": 10.6, "low": 10.4,
        "volume": 1.0, "amount": 10.5, "turnover": 0.1, "market_cap": 1e9,
    }])
    p = pd.DataFrame([{
        "date": dates[0], "symbol": "AAA", "pre_close": 10.0,
        "limit_up": 11.0, "limit_down": 9.0, "trade_status": 0,
    }])
    idx = pd.DataFrame([{"symbol": "000300.SH", "date": dates[0],
                         "close": 4000.0, "pre_close": 4000.0}])
    per_day = features._compute_daily_features(
        f.merge(p[["symbol", "date", "pre_close", "limit_up", "limit_down", "trade_status"]],
                on=["symbol", "date"], how="inner"),
        idx,
    )
    assert per_day["dist_limit_up"].iloc[0] > 0
    assert per_day["dist_limit_down"].iloc[0] > 0


def test_excess_ret_zero_when_stock_matches_index_return():
    """If stock ret = index ret, excess_ret = 0."""
    # Stock: pre=10, close=10.5 → ret = 0.05. Index: pre=4000, close=4200 → ret = 0.05.
    f = pd.DataFrame([{
        "date": "20260501", "symbol": "AAA", "name": "T",
        "open": 10.0, "close": 10.5, "high": 10.6, "low": 9.9,
        "volume": 1.0, "amount": 10.5, "turnover": 0.1, "market_cap": 1e9,
    }])
    p = pd.DataFrame([{
        "date": "20260501", "symbol": "AAA", "pre_close": 10.0,
        "limit_up": 11.0, "limit_down": 9.0, "trade_status": 0,
    }])
    idx = pd.DataFrame([{"symbol": "000300.SH", "date": "20260501",
                         "close": 4200.0, "pre_close": 4000.0}])
    per_day = features._compute_daily_features(
        f.merge(p[["symbol", "date", "pre_close", "limit_up", "limit_down", "trade_status"]],
                on=["symbol", "date"], how="inner"),
        idx,
    )
    assert per_day["excess_ret"].iloc[0] == pytest.approx(0.0, abs=1e-12)


def test_standardization_uses_train_stats_not_score():
    """feat_mean / feat_std must come from training samples only (defence against leakage)."""
    dates = _make_dates("20260501", 25)
    scan = dates[-1]
    fbundle = build_features(
        factor_df=_factor_frame("000001.SZ", dates),
        post_df=_post_frame("000001.SZ", dates),
        index_df=_index_frame(dates),
        universe=["000001.SZ"],
        scan_date=scan,
        lookback=20,
        train_days=20,
    )
    # Recompute mean of the training set explicitly and compare.
    expected_mean = fbundle.train_x.mean(axis=0) * fbundle.feat_std + fbundle.feat_mean
    # Because train_x is already z-scored with feat_mean/feat_std, the recovered raw mean
    # should equal feat_mean (train x has zero mean in z-scored space).
    np.testing.assert_allclose(expected_mean, fbundle.feat_mean, atol=1e-6)


def test_halted_days_exclude_score_row():
    """A stock halted on scan_date must not appear in score_symbols."""
    dates = _make_dates("20260501", 25)
    scan = dates[-1]
    p = _post_frame("AAA", dates)
    # Halt the last day.
    p.loc[p["date"] == scan, "trade_status"] = 1
    fbundle = build_features(
        factor_df=_factor_frame("AAA", dates),
        post_df=p,
        index_df=_index_frame(dates),
        universe=["AAA"],
        scan_date=scan,
        lookback=20,
        train_days=20,
    )
    assert "AAA" not in fbundle.score_symbols
