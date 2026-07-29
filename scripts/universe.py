"""CSI300 (or arbitrary index) constituents on scan-day T.

Design §4: universe is the list of constituent stocks of the given index on T. If T
has no weights row (e.g. index reconstitution lag on a fresh listing day), fall back
to the most recent available date within a small lookback window.

Since CSI300 constituents are by construction non-ST and highly liquid, we do NOT
overlay a separate size/turnover filter here — the point-in-time suspension mask is
applied later in features.py.
"""
from __future__ import annotations

import pandas as pd


def resolve_universe(
    weights_df: pd.DataFrame,
    date: str,
) -> list[str]:
    """Return the sorted list of stock_symbols making up the universe on `date`.

    Args:
        weights_df: rows with columns [index_symbol, date, stock_symbol],
                    typically the response of get_index_weights over a range that
                    *includes* `date` (a fallback range gives more resilience if
                    `date` itself has no publication).
        date: scan day T (YYYYMMDD).

    Returns:
        Sorted list of stock_symbols; empty list if no row is available up to `date`.
    """
    if weights_df.empty:
        return []

    # Prefer exact match; else the most recent date ≤ T.
    exact = weights_df[weights_df["date"] == date]
    if not exact.empty:
        pool = exact
    else:
        prior = weights_df[weights_df["date"] < date]
        if prior.empty:
            return []
        latest_date = prior["date"].max()
        pool = prior[prior["date"] == latest_date]

    symbols = pool["stock_symbol"].astype(str).unique().tolist()
    return sorted(symbols)
