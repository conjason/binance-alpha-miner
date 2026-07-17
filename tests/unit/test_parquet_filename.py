"""Parquet filename symbol/timeframe parsing with aliases."""
from __future__ import annotations

import pytest

from data_pipeline.parquet_manager import normalize_timeframe_token, parse_parquet_filename


@pytest.mark.parametrize(
    "name, symbol, timeframe",
    [
        ("AAPL_H1.parquet", "AAPL", "H1"),
        ("XAUUSD_H1.parquet", "XAUUSD", "H1"),
        ("US30.cash_H1.parquet", "US30.cash", "H1"),
        ("002008_60min.parquet", "002008", "H1"),
        ("002008_60m.parquet", "002008", "H1"),
        ("BTCUSDT_1h.parquet", "BTCUSDT", "H1"),
        ("600519_5min.parquet", "600519", "M5"),
        ("600519_15m.parquet", "600519", "M15"),
        ("ETHUSDT_4h.parquet", "ETHUSDT", "H4"),
        ("000001_1d.parquet", "000001", "D1"),
        ("foo_D1.parquet", "foo", "D1"),
        ("bar_m30.parquet", "bar", "M30"),
    ],
)
def test_parse_parquet_filename_aliases(name, symbol, timeframe):
    assert parse_parquet_filename(name) == (symbol, timeframe)


def test_normalize_timeframe_token():
    assert normalize_timeframe_token("60min") == "H1"
    assert normalize_timeframe_token("H1") == "H1"
    assert normalize_timeframe_token("nope") is None


def test_parse_rejects_unknown_tf():
    with pytest.raises(ValueError, match="周期"):
        parse_parquet_filename("002008_xyz.parquet")
