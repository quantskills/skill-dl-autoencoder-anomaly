"""Unit tests for scripts/data — error handling and column validation.

Same rationale as skill-etf-flow-radar/tests/test_data.py: when panda_data returns
HTTP 5xx or missing env vars, we want one clean stderr line and exit code 1,
not a 40-line traceback. Column mismatches must raise ValueError so callers can
convert them to exit code 4.
"""
import sys
import types

import pandas as pd
import pytest

from scripts import data


def _install_fake_panda_data(monkeypatch, init_token_impl=lambda **kw: None):
    """Register a stub `panda_data` module so `import panda_data` inside data.py works."""
    fake = types.ModuleType("panda_data")
    fake.init_token = init_token_impl
    exceptions_mod = types.ModuleType("panda_data.exceptions")

    class ServiceError(Exception):
        pass

    exceptions_mod.ServiceError = ServiceError
    fake.exceptions = exceptions_mod
    monkeypatch.setitem(sys.modules, "panda_data", fake)
    monkeypatch.setitem(sys.modules, "panda_data.exceptions", exceptions_mod)
    return fake, ServiceError


# ---------------------------------------------------------------------------
# _main / auth error paths (mirrors etf-radar test_data.py)
# ---------------------------------------------------------------------------


def test_main_returns_1_and_prints_short_error_on_service_error(monkeypatch, capsys):
    fake, ServiceError = _install_fake_panda_data(
        monkeypatch,
        init_token_impl=lambda **kw: (_ for _ in ()).throw(
            sys.modules["panda_data.exceptions"].ServiceError("登录失败: HTTP 503")
        ),
    )
    monkeypatch.setenv("PANDA_DATA_USERNAME", "u")
    monkeypatch.setenv("PANDA_DATA_PASSWORD", "p")
    monkeypatch.setattr(sys, "argv", ["data.py", "--self-check", "--date", "20260729"])

    rc = data._main()

    assert rc == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert (
        "登录失败" in captured.err
        or "HTTP 503" in captured.err
        or "panda_data" in captured.err.lower()
    )


def test_main_returns_1_on_missing_credentials(monkeypatch, capsys):
    monkeypatch.delenv("PANDA_DATA_USERNAME", raising=False)
    monkeypatch.delenv("PANDA_DATA_PASSWORD", raising=False)
    monkeypatch.setattr(sys, "argv", ["data.py", "--self-check", "--date", "20260729"])

    rc = data._main()

    assert rc == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "PANDA_DATA_USERNAME" in captured.err


# ---------------------------------------------------------------------------
# Column self-check
# ---------------------------------------------------------------------------


def test_load_factor_raises_on_missing_columns(monkeypatch):
    """If get_factor drops one of the required columns, _assert_columns must complain."""
    fake, _ = _install_fake_panda_data(monkeypatch)
    # Return a frame missing "turnover".
    partial = pd.DataFrame({
        "date": ["20260729"],
        "symbol": ["000001.SZ"],
        "name": ["平安银行"],
        "open": [10.0], "close": [10.5], "high": [10.6], "low": [9.9],
        "volume": [1e6], "amount": [1e7], "market_cap": [2e11],
    })
    fake.get_factor = lambda **kw: partial

    with pytest.raises(ValueError, match="turnover"):
        data.load_factor("20260601", "20260729", "000300.SH")


def test_load_stock_post_returns_empty_frame_on_none(monkeypatch):
    fake, _ = _install_fake_panda_data(monkeypatch)
    fake.get_stock_daily_post = lambda **kw: None

    df = data.load_stock_post("20260601", "20260729", "000300.SH")
    assert df.empty
    assert set(df.columns) == data.EXPECTED_COLUMNS["stock_post"]


def test_load_index_weights_stringifies_columns(monkeypatch):
    fake, _ = _install_fake_panda_data(monkeypatch)
    fake.get_index_weights = lambda **kw: pd.DataFrame({
        "index_symbol": ["000300.SH"],
        "date": [20260729],           # int on purpose — must be cast to str
        "stock_symbol": ["600519.SH"],
        "weight": [0.05],
    })

    df = data.load_index_weights("000300.SH", "20260729")
    assert df["date"].dtype == object
    assert df["date"].iloc[0] == "20260729"
    assert df["stock_symbol"].iloc[0] == "600519.SH"


def test_index_component_strips_suffix():
    assert data._index_component_of("000300.SH") == "000300"
    assert data._index_component_of("000905.SH") == "000905"
    assert data._index_component_of("000852") == "000852"


def test_get_last_trade_date_wrapper(monkeypatch):
    fake, _ = _install_fake_panda_data(monkeypatch)
    fake.get_last_trade_date = lambda **kw: pd.DataFrame({"date": ["20260729"]})

    assert data.get_last_trade_date() == "20260729"


def test_get_last_trade_date_none_when_empty(monkeypatch):
    fake, _ = _install_fake_panda_data(monkeypatch)
    fake.get_last_trade_date = lambda **kw: pd.DataFrame(columns=["date"])

    assert data.get_last_trade_date() is None


def test_get_prev_trade_date_wrapper(monkeypatch):
    fake, _ = _install_fake_panda_data(monkeypatch)
    captured = {}

    def _stub(date, exchange, n):
        captured["date"] = date
        captured["n"] = n
        return pd.DataFrame({"date": ["20260501"]})

    fake.get_prev_trade_date = _stub
    result = data.get_prev_trade_date("20260729", n=60)
    assert result == "20260501"
    assert captured == {"date": "20260729", "n": 60}


def test_get_last_trade_date_wrapper_accepts_bare_string(monkeypatch):
    """Live panda_data (checked 2026-07-30) returns a plain str, not a DataFrame."""
    fake, _ = _install_fake_panda_data(monkeypatch)
    fake.get_last_trade_date = lambda **kw: "20260729"
    assert data.get_last_trade_date() == "20260729"


def test_get_prev_trade_date_wrapper_accepts_bare_string(monkeypatch):
    fake, _ = _install_fake_panda_data(monkeypatch)
    fake.get_prev_trade_date = lambda **kw: "20260430"
    assert data.get_prev_trade_date("20260729", n=60) == "20260430"
