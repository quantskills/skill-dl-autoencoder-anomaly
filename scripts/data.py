"""panda_data thin wrappers for skill-dl-autoencoder-anomaly.

Six interfaces are used (see references/need_used_api.md):
  - get_last_trade_date, get_prev_trade_date  (calendar utilities)
  - get_index_weights                          (CSI300 constituents)
  - get_factor                                 (OHLCV + turnover + market_cap)
  - get_stock_daily_post                       (pre_close + limit_up/down + trade_status)
  - get_index_daily                            (benchmark for excess-return feature)

Column names are validated against a required-superset set (EXPECTED_COLUMNS) on every
load; mismatch triggers exit code 4 via self_check().

panda_data is a private package imported lazily inside each function so that this module
can be imported (and its EXPECTED_COLUMNS inspected) without panda_data installed —
useful for unit-testing callers that mock the loaders.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# Columns we DEPEND ON downstream. Upstream may return more; missing any of these breaks things.
EXPECTED_COLUMNS: dict[str, set[str]] = {
    "index_weights": {"index_symbol", "date", "stock_symbol"},
    # NOTE: get_factor docs list `name` as a stock-type column but the live response
    # (checked 2026-07-30) does NOT include it. We pull `name` from stock_post instead.
    "factor": {
        "date", "symbol",
        "open", "close", "high", "low",
        "volume", "amount", "turnover", "market_cap",
    },
    "stock_post": {
        "date", "symbol", "name",
        "pre_close", "limit_up", "limit_down", "trade_status",
    },
    "index_daily": {"symbol", "date", "close", "pre_close"},
}

# Factor names for get_factor(factors=...). The response frame carries these columns plus
# the mandatory date/symbol/name.
FACTOR_NAMES: list[str] = [
    "open", "close", "high", "low",
    "volume", "amount", "turnover", "market_cap",
]


def init_panda_data() -> None:
    """Authenticate with panda_data using env vars. Raises RuntimeError if unset."""
    user = os.environ.get("PANDA_DATA_USERNAME")
    pwd = os.environ.get("PANDA_DATA_PASSWORD")
    if not user or not pwd:
        raise RuntimeError(
            "Missing env vars PANDA_DATA_USERNAME / PANDA_DATA_PASSWORD. "
            "Export them before running the scan."
        )
    import panda_data
    panda_data.init_token(username=user, password=pwd)


def _assert_columns(df: pd.DataFrame, kind: str) -> None:
    expected = EXPECTED_COLUMNS[kind]
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"panda_data {kind} response missing columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}."
        )


def _index_component_of(index_symbol: str) -> str:
    """Strip suffix for the `index_component` / `indicator` argument (e.g. 000300.SH → 000300)."""
    return index_symbol.split(".")[0]


def get_last_trade_date(exchange: str = "SH") -> str | None:
    """Wrap panda_data.get_last_trade_date; returns a single YYYYMMDD string or None.

    Observation (2026-07-30): live API returns a plain `str` even though the doc shows
    a one-row DataFrame. Handle both shapes.
    """
    import panda_data
    result = panda_data.get_last_trade_date(exchange=exchange)
    if result is None:
        return None
    if isinstance(result, str):
        return result or None
    if hasattr(result, "empty") and result.empty:
        return None
    if hasattr(result, "iloc"):
        return str(result["date"].iloc[0])
    return str(result)


def get_prev_trade_date(date: str, n: int = 1, exchange: str = "SH") -> str | None:
    """Wrap panda_data.get_prev_trade_date; returns YYYYMMDD or None. See note above."""
    import panda_data
    result = panda_data.get_prev_trade_date(date=date, exchange=exchange, n=n)
    if result is None:
        return None
    if isinstance(result, str):
        return result or None
    if hasattr(result, "empty") and result.empty:
        return None
    if hasattr(result, "iloc"):
        return str(result["date"].iloc[0])
    return str(result)


def load_index_weights(index_symbol: str, date: str) -> pd.DataFrame:
    """CSI300 (or other index) constituents on a single day.

    Returns a DataFrame with columns [index_symbol, date, stock_symbol]; empty frame
    (with schema) if panda_data returned nothing.
    """
    import panda_data
    df = panda_data.get_index_weights(
        index_symbol=index_symbol,
        start_date=date,
        end_date=date,
    )
    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame(columns=sorted(EXPECTED_COLUMNS["index_weights"]))
    _assert_columns(df, "index_weights")
    df["date"] = df["date"].astype(str)
    df["stock_symbol"] = df["stock_symbol"].astype(str)
    return df


def load_factor(
    start_date: str,
    end_date: str,
    index_symbol: str,
) -> pd.DataFrame:
    """get_factor over [start_date, end_date] filtered to `index_symbol` universe.

    Returns OHLCV + turnover + market_cap. Empty frame (with schema) if the response is
    empty.
    """
    import panda_data
    df = panda_data.get_factor(
        start_date=start_date,
        end_date=end_date,
        factors=FACTOR_NAMES,
        type="stock",
        index_component=_index_component_of(index_symbol),
    )
    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame(columns=sorted(EXPECTED_COLUMNS["factor"]))
    _assert_columns(df, "factor")
    df["date"] = df["date"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    return df


def load_stock_post(
    start_date: str,
    end_date: str,
    index_symbol: str,
) -> pd.DataFrame:
    """get_stock_daily_post over the same window and universe as load_factor.

    Only the columns we do not already have from get_factor are used downstream
    (pre_close, limit_up, limit_down, trade_status).
    """
    import panda_data
    df = panda_data.get_stock_daily_post(
        start_date=start_date,
        end_date=end_date,
        indicator=_index_component_of(index_symbol),
        st=False,
    )
    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame(columns=sorted(EXPECTED_COLUMNS["stock_post"]))
    _assert_columns(df, "stock_post")
    df["date"] = df["date"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    return df


def load_index_daily(index_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Benchmark daily OHLCV for `index_symbol` over [start_date, end_date]."""
    import panda_data
    df = panda_data.get_index_daily(
        symbol=index_symbol,
        start_date=start_date,
        end_date=end_date,
    )
    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame(columns=sorted(EXPECTED_COLUMNS["index_daily"]))
    _assert_columns(df, "index_daily")
    df["date"] = df["date"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    return df


def self_check(date: str, index_symbol: str = "000300.SH") -> int:
    """Manually invoke each loader for `date` and print column diagnostics.

    Returns 0 on success, 4 on any column mismatch (matches design §7 exit code).
    """
    init_panda_data()
    import panda_data
    exit_code = 0

    ic = _index_component_of(index_symbol)
    checks = (
        ("index_weights", lambda: panda_data.get_index_weights(
            index_symbol=index_symbol, start_date=date, end_date=date,
        )),
        ("factor", lambda: panda_data.get_factor(
            start_date=date, end_date=date,
            factors=FACTOR_NAMES, type="stock", index_component=ic,
        )),
        ("stock_post", lambda: panda_data.get_stock_daily_post(
            start_date=date, end_date=date, indicator=ic, st=False,
        )),
        ("index_daily", lambda: panda_data.get_index_daily(
            symbol=index_symbol, start_date=date, end_date=date,
        )),
    )
    for kind, loader in checks:
        print(f"--- {kind} ---")
        try:
            df = loader()
        except Exception as e:
            print(f"[ERROR] {kind} raised: {e}")
            exit_code = 4
            continue
        if df is None or (hasattr(df, "empty") and df.empty):
            print(f"[WARN] {kind} returned empty on {date}")
            continue
        got = set(df.columns)
        expected = EXPECTED_COLUMNS[kind]
        missing = expected - got
        extra = got - expected
        print(f"got columns:      {sorted(got)}")
        print(f"missing required: {sorted(missing)}")
        print(f"extra (ignored):  {sorted(extra)}")
        if missing:
            exit_code = 4
    return exit_code


def _main() -> int:
    p = argparse.ArgumentParser(
        description="panda_data field self-check for skill-dl-autoencoder-anomaly",
    )
    p.add_argument("--self-check", action="store_true", required=True)
    p.add_argument("--date", required=True, help="YYYYMMDD")
    p.add_argument("--index", default="000300.SH", help="Index symbol (default 000300.SH = CSI300)")
    args = p.parse_args()

    # Lazily resolve panda_data.exceptions.ServiceError. If panda_data isn't installed,
    # init_panda_data() will fail earlier with RuntimeError (missing env vars) or ImportError;
    # either way we won't need to catch ServiceError, so a placeholder tuple is safe.
    try:
        from panda_data.exceptions import ServiceError as _ServiceError
        service_error_cls: tuple = (_ServiceError,)
    except ImportError:
        service_error_cls = ()

    try:
        return self_check(args.date, index_symbol=args.index)
    except RuntimeError as e:
        # Missing PANDA_DATA_USERNAME / PANDA_DATA_PASSWORD, etc.
        print(f"[error] {e}", file=sys.stderr)
        return 1
    except service_error_cls as e:  # type: ignore[misc]  # empty tuple = catch nothing
        print(f"[error] panda_data service error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_main())
