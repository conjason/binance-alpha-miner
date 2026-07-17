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


# ── 无周期后缀（币安式 BTCUSDT.parquet）的周期推断 ──────────────────────────

def test_suffixless_uses_default_timeframe():
    """无后缀、无目录线索时，回退到 Config.DEFAULT_TIMEFRAME（默认 5m → M5）。"""
    assert parse_parquet_filename("BTCUSDT.parquet") == ("BTCUSDT", "M5")
    assert parse_parquet_filename("1000BONKUSDT.parquet") == ("1000BONKUSDT", "M5")


def test_suffixless_infers_from_parent_dir():
    """父目录名含周期 token 时优先用之（如币安 1m 目录 ..._1m）。"""
    assert parse_parquet_filename("data_1m/BTCUSDT.parquet") == ("BTCUSDT", "M1")
    assert parse_parquet_filename("klines_1h/ETHUSDT.parquet") == ("ETHUSDT", "H1")


def test_explicit_timeframe_wins_over_inference():
    assert parse_parquet_filename("data_1m/BTCUSDT.parquet", "5m") == ("BTCUSDT", "M5")


def test_env_default_timeframe_override(monkeypatch):
    monkeypatch.setattr("config.Config.DEFAULT_TIMEFRAME", "15m")
    assert parse_parquet_filename("ADAUSDT.parquet") == ("ADAUSDT", "M15")


def test_unknown_suffix_folds_into_symbol_with_default_tf():
    """后缀非法时不再报错，整段当品种名 + 默认周期。"""
    assert parse_parquet_filename("002008_xyz.parquet") == ("002008_xyz", "M5")
