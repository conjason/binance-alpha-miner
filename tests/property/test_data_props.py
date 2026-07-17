# Feature: mt5-alphagpt-refactor, Property 1: MT5DataFetcher 返回规范 DataFrame
# Feature: mt5-alphagpt-refactor, Property 2: 多品种时间轴对齐不变量
"""
Property-based tests for data_pipeline.
Property 1 Validates: Requirements 2.3, 2.5, 2.6
Property 2 Validates: Requirements 3.2, 3.3
"""

import sys
import types
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from hypothesis import given, settings, strategies as st, assume


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_fake_rates(n: int = 5) -> np.ndarray:
    """Build a fake numpy structured array matching the MT5 rates format."""
    dtype = np.dtype([
        ("time", np.int64),
        ("open", np.float64),
        ("high", np.float64),
        ("low", np.float64),
        ("close", np.float64),
        ("tick_volume", np.int64),
        ("spread", np.int32),
        ("real_volume", np.int64),
    ])
    data = np.zeros(n, dtype=dtype)
    data["time"] = np.arange(n, dtype=np.int64) * 3600
    data["open"] = 1800.0 + np.arange(n, dtype=np.float64)
    data["high"] = data["open"] + 2.0
    data["low"] = data["open"] - 2.0
    data["close"] = data["open"] + 0.5
    data["tick_volume"] = np.ones(n, dtype=np.int64) * 100
    return data


# ── Strategies ────────────────────────────────────────────────────────────────

# Valid printable ASCII symbol strings (1–12 chars), e.g. "XAUUSD", "EURUSD"
symbol_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=1,
    max_size=12,
)

# MT5 timeframe constants are positive integers
timeframe_strategy = st.integers(min_value=1, max_value=49153)


# ── Property 1: MT5DataFetcher 返回规范 DataFrame ─────────────────────────────
# Validates: Requirements 2.3, 2.5, 2.6

EXPECTED_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume"]


@settings(max_examples=100)
@given(
    symbol=symbol_strategy,
    timeframe=timeframe_strategy,
)
def test_property1_fetcher_returns_canonical_dataframe(symbol: str, timeframe: int):
    """
    For any valid symbol / timeframe combination, when MT5DataFetcher.fetch()
    succeeds, the returned DataFrame must contain exactly the columns
    [time, open, high, low, close, tick_volume] and at least 1 row.

    Validates: Requirements 2.3, 2.5, 2.6
    """
    fake_rates = _make_fake_rates(n=5)

    # Build a minimal mock mt5 module so _MT5_AVAILABLE is True at runtime
    mock_mt5 = MagicMock()
    mock_mt5.copy_rates_from_pos.return_value = fake_rates
    mock_mt5.initialize.return_value = True
    mock_mt5.last_error.return_value = (0, "No error")

    with patch("data_pipeline.fetcher.mt5", mock_mt5), \
         patch("data_pipeline.fetcher._MT5_AVAILABLE", True):

        from data_pipeline.fetcher import MT5DataFetcher
        fetcher = MT5DataFetcher()
        df = fetcher.fetch(symbol, timeframe, count=5)

    # ── Assertions ────────────────────────────────────────────────────────────
    assert isinstance(df, pd.DataFrame), "fetch() must return a pandas DataFrame"
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"DataFrame columns must be exactly {EXPECTED_COLUMNS}, got {list(df.columns)}"
    )
    assert len(df) > 0, "DataFrame must have at least 1 row when MT5 returns data"

    # Also verify copy_rates_from_pos was called with the correct positional args
    mock_mt5.copy_rates_from_pos.assert_called_once_with(symbol, timeframe, 0, 5)


# Feature: mt5-alphagpt-refactor, Property 5: 目标收益率 open-to-open 公式
# Validates: Requirements 3.4

import math
import torch
from data_pipeline.data_manager import MT5DataManager


# ── Property 5: target_ret[t] == log(open[t+2] / open[t+1])，边界为 0 ─────────
# Validates: Requirements 3.4

@settings(max_examples=100)
@given(
    open_prices=st.lists(
        st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=200,
    ),
    n_symbols=st.integers(min_value=1, max_value=2),
)
def test_property5_target_ret_formula(open_prices: list, n_symbols: int):
    """
    For any sequence of positive open prices of length T (T >= 10),
    MT5DataManager._compute_target_ret() must satisfy:
      - output shape == (N, T)
      - target_ret[n, t] == log(open[n, t+2] / open[n, t+1]) for all t in [0, T-3]
      - target_ret[n, T-2] == 0.0  (boundary)
      - target_ret[n, T-1] == 0.0  (boundary)

    Validates: Requirements 3.4
    """
    t = len(open_prices)
    n = n_symbols

    # Build open tensor of shape [N, T] — each symbol row gets the same prices
    # (we only assert on row 0 for the formula; boundaries apply to all rows)
    row = torch.tensor(open_prices, dtype=torch.float32)
    open_tensor = row.unsqueeze(0).expand(n, t).clone()  # [N, T]

    target_ret = MT5DataManager._compute_target_ret(open_tensor)

    # ── Shape invariant ───────────────────────────────────────────────────
    assert target_ret.shape == (n, t), (
        f"Expected shape ({n}, {t}), got {target_ret.shape}"
    )

    # ── Formula correctness for all valid indices (row 0) ─────────────────
    for idx in range(t - 2):
        expected = math.log(open_prices[idx + 2] / open_prices[idx + 1])
        actual = target_ret[0, idx].item()
        assert abs(actual - expected) < 1e-4, (
            f"target_ret[0, {idx}] = {actual:.6f}, expected log({open_prices[idx+2]}"
            f" / {open_prices[idx+1]}) = {expected:.6f}"
        )

    # ── Boundary invariant ────────────────────────────────────────────────
    assert target_ret[0, t - 2].item() == 0.0, (
        f"Boundary target_ret[0, T-2] must be 0.0, got {target_ret[0, t - 2].item()}"
    )
    assert target_ret[0, t - 1].item() == 0.0, (
        f"Boundary target_ret[0, T-1] must be 0.0, got {target_ret[0, t - 1].item()}"
    )


# ── Property 2: 多品种时间轴对齐不变量 ───────────────────────────────────────
# Feature: mt5-alphagpt-refactor, Property 2: 多品种时间轴对齐不变量
# Validates: Requirements 3.2, 3.3

# Fields that raw_dict must contain
RAW_DICT_FIELDS = ["open", "high", "low", "close", "volume"]


def _make_symbol_df(timestamps: list[int], base_price: float = 100.0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame for a given list of timestamps."""
    n = len(timestamps)
    opens = np.full(n, base_price, dtype=np.float64)
    highs = opens + 2.0
    lows  = opens - 2.0
    closes = opens + 0.5
    volumes = np.ones(n, dtype=np.float64) * 500.0
    return pd.DataFrame({
        "time":        np.array(timestamps, dtype=np.int64),
        "open":        opens,
        "high":        highs,
        "low":         lows,
        "close":       closes,
        "tick_volume": volumes.astype(np.int64),
    })


@st.composite
def multi_symbol_dfs(draw) -> dict[str, pd.DataFrame]:
    """
    Generate 2–4 symbols, each with a different set of timestamps.
    Some symbols share timestamps, some have gaps — modelling real MT5 data.
    """
    # Build a common pool of timestamps (sorted integers, step = 3600)
    pool_size = draw(st.integers(min_value=10, max_value=60))
    base_ts = 1_700_000_000
    all_timestamps = [base_ts + i * 3600 for i in range(pool_size)]

    num_symbols = draw(st.integers(min_value=2, max_value=4))

    dfs: dict[str, pd.DataFrame] = {}
    for idx in range(num_symbols):
        symbol = f"SYM{idx}"
        # Each symbol gets a random subset of the pool (at least 10 timestamps)
        chosen_count = draw(st.integers(min_value=10, max_value=pool_size))
        # Always take a contiguous slice so the symbol is "valid" (>= MIN_BARS mock)
        start = draw(st.integers(min_value=0, max_value=pool_size - chosen_count))
        timestamps = all_timestamps[start : start + chosen_count]
        base_price = draw(st.floats(min_value=1.0, max_value=5000.0,
                                    allow_nan=False, allow_infinity=False))
        dfs[symbol] = _make_symbol_df(timestamps, base_price=base_price)

    return dfs


@settings(max_examples=100)
@given(raw_dfs=multi_symbol_dfs())
def test_property2_timeline_alignment_t_dimension_identical(raw_dfs: dict):
    """
    For any group of symbol DataFrames with different starting timestamps or
    gaps, after MT5DataManager._align_timelines() and _build_raw_dict(), all
    fields in raw_dict must have the same T dimension, i.e., every symbol
    shares a single common timeline length.

    Validates: Requirements 3.2, 3.3
    """
    from data_pipeline.data_manager import MT5DataManager

    # We need a minimal MT5DataFetcher stub — no real MT5 needed
    mock_fetcher = MagicMock()

    mgr = MT5DataManager(fetcher=mock_fetcher)
    # Manually set _symbols so _build_raw_dict knows the order
    mgr._symbols = list(raw_dfs.keys())

    # ── Step 1: align timelines ───────────────────────────────────────────
    aligned = mgr._align_timelines(raw_dfs)

    # All aligned DataFrames must have the same number of rows
    row_counts = {sym: len(df) for sym, df in aligned.items()}
    assert len(set(row_counts.values())) == 1, (
        f"Aligned DataFrames have different row counts: {row_counts}"
    )

    # ── Step 2: build raw_dict ────────────────────────────────────────────
    raw_dict = mgr._build_raw_dict(aligned)

    # All fields must exist
    for field in RAW_DICT_FIELDS:
        assert field in raw_dict, f"raw_dict is missing field '{field}'"

    # Collect T dimension for every field
    t_dims = {field: raw_dict[field].shape[1] for field in RAW_DICT_FIELDS}
    unique_t = set(t_dims.values())

    assert len(unique_t) == 1, (
        f"raw_dict fields have inconsistent T dimensions: {t_dims}"
    )

    # ── Step 3: shape must be [N, T] ──────────────────────────────────────
    n = len(raw_dfs)
    t = next(iter(unique_t))
    for field in RAW_DICT_FIELDS:
        shape = raw_dict[field].shape
        assert shape == (n, t), (
            f"raw_dict['{field}'] shape is {shape}, expected ({n}, {t})"
        )
