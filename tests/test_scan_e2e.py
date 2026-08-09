"""Offline end-to-end verification of data plumbing → training → report output.

The remote panda_data boundary is mocked, but the real scan orchestration, feature
engineering, PyTorch training, scoring, CSV writer, and Markdown writer all run.
This keeps the full chain verifiable without credentials or network access.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from scripts import scan


def _synthetic_inputs(scan_date: str):
    symbols = [f"{600000 + i:06d}.SH" for i in range(12)]
    dates = [d.strftime("%Y%m%d") for d in pd.bdate_range("2026-04-01", periods=85)]
    rng = np.random.default_rng(42)

    factor_rows = []
    post_rows = []
    index_rows = []
    index_close = 100.0
    for i, date in enumerate(dates):
        index_pre = index_close
        index_close = index_pre * (1.0 + 0.0005 + 0.0002 * np.sin(i / 7.0))
        index_rows.append({
            "symbol": "000300.SH", "date": date,
            "close": index_close, "pre_close": index_pre,
        })
        for j, symbol in enumerate(symbols):
            pre_close = 100.0 + j + i * 0.05
            ret = 0.001 * np.sin(i / 5.0 + j / 3.0) + rng.normal(0, 0.0002)
            close = pre_close * (1.0 + ret)
            factor_rows.append({
                "date": date, "symbol": symbol,
                "open": pre_close * (1.0 + ret / 3.0),
                "close": close, "high": max(close, pre_close) * 1.002,
                "low": min(close, pre_close) * 0.998,
                "volume": 1_000_000.0 + j * 10_000.0,
                "amount": 100_000_000.0,
                "turnover": 0.15 + 0.01 * np.cos(i / 6.0 + j),
                "market_cap": 10_000_000_000.0 + j * 100_000_000.0,
            })
            post_rows.append({
                "date": date, "symbol": symbol, "name": f"测试股票{j + 1}",
                "pre_close": pre_close,
                "limit_up": pre_close * 1.10,
                "limit_down": pre_close * 0.90,
                "trade_status": 0,
            })

    weights = pd.DataFrame([
        {"index_symbol": "000300.SH", "date": scan_date, "stock_symbol": symbol}
        for symbol in symbols
    ])
    return (
        pd.DataFrame(factor_rows),
        pd.DataFrame(post_rows),
        pd.DataFrame(index_rows),
        weights,
    )


def test_scan_end_to_end_writes_csv_and_markdown(monkeypatch, tmp_path):
    scan_date = "20260728"
    factor_df, post_df, index_df, weights_df = _synthetic_inputs(scan_date)

    monkeypatch.setattr(scan.data_mod, "init_panda_data", lambda: None)
    monkeypatch.setattr(scan.data_mod, "get_prev_trade_date", lambda *args, **kwargs: "20260401")
    monkeypatch.setattr(scan.data_mod, "load_factor", lambda *args, **kwargs: factor_df)
    monkeypatch.setattr(scan.data_mod, "load_stock_post", lambda *args, **kwargs: post_df)
    monkeypatch.setattr(scan.data_mod, "load_index_daily", lambda *args, **kwargs: index_df)
    monkeypatch.setattr(scan.data_mod, "load_index_weights", lambda *args, **kwargs: weights_df)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scan.py", "--date", scan_date,
            "--lookback", "20", "--train_days", "60",
            "--epochs", "2", "--batch_size", "64", "--top_n", "5",
            "--seed", "42", "--output_dir", str(tmp_path),
        ],
    )

    assert scan.main() == 0

    csv_path = tmp_path / f"anomaly_{scan_date}.csv"
    md_path = tmp_path / f"anomaly_{scan_date}.md"
    assert csv_path.is_file() and csv_path.stat().st_size > 0
    assert md_path.is_file() and md_path.stat().st_size > 0

    result = pd.read_csv(csv_path)
    assert len(result) == 5
    assert list(result["rank"]) == [1, 2, 3, 4, 5]
    assert result["reconstruction_error"].notna().all()
    assert result["top_feature"].notna().all()
    text = md_path.read_text(encoding="utf-8")
    assert "训练样本" in text
    assert "验证集 MSE" in text
    assert "Top 5 异常" in text
