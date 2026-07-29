"""Unit tests for scripts.universe.resolve_universe."""
import pandas as pd

from scripts.universe import resolve_universe


def _weights(rows):
    return pd.DataFrame(rows, columns=["index_symbol", "date", "stock_symbol"])


def test_exact_date_match():
    df = _weights([
        ("000300.SH", "20260728", "600519.SH"),
        ("000300.SH", "20260728", "601318.SH"),
        ("000300.SH", "20260729", "600519.SH"),
        ("000300.SH", "20260729", "601318.SH"),
        ("000300.SH", "20260729", "000001.SZ"),
    ])
    assert resolve_universe(df, "20260729") == ["000001.SZ", "600519.SH", "601318.SH"]


def test_fallback_to_most_recent_prior_date():
    df = _weights([
        ("000300.SH", "20260726", "600519.SH"),
        ("000300.SH", "20260726", "601318.SH"),
        ("000300.SH", "20260728", "600519.SH"),
        ("000300.SH", "20260728", "000001.SZ"),
    ])
    # T=20260729 has no row; expect the 20260728 pool (600519 + 000001), NOT 20260726.
    assert resolve_universe(df, "20260729") == ["000001.SZ", "600519.SH"]


def test_empty_frame_returns_empty_list():
    assert resolve_universe(pd.DataFrame(columns=["index_symbol", "date", "stock_symbol"]),
                            "20260729") == []


def test_no_dates_le_scan_date_returns_empty():
    df = _weights([("000300.SH", "20260801", "600519.SH")])
    # Scan date is earlier than all rows → still empty.
    assert resolve_universe(df, "20260729") == []


def test_duplicates_deduplicated_and_sorted():
    df = _weights([
        ("000300.SH", "20260729", "601318.SH"),
        ("000300.SH", "20260729", "600519.SH"),
        ("000300.SH", "20260729", "601318.SH"),
    ])
    assert resolve_universe(df, "20260729") == ["600519.SH", "601318.SH"]
