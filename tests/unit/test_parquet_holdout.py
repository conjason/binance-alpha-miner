"""Parquet 训练窗与末尾真样本外切片。"""

import numpy as np
import pandas as pd
import pytest

from config import Config
from data_pipeline.parquet_manager import ParquetDataManager


def _write_parquet(path, rows: int = 12) -> None:
    x = np.arange(rows, dtype="float64") + 100.0
    pd.DataFrame(
        {
            "ts": np.arange(rows, dtype="int64") * 300_000 + 1_700_000_000_000,
            "open": x,
            "high": x + 1,
            "low": x - 1,
            "close": x + 0.5,
            "volume": np.ones(rows),
        }
    ).to_parquet(path, index=False)


def test_end_offset_then_max_bars_produces_disjoint_training_window(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "MIN_BARS", 5)
    path = tmp_path / "BTCUSDT_5m.parquet"
    _write_parquet(path)

    manager = ParquetDataManager(path, max_bars=5, end_offset=3)
    manager.load()

    # 完整序列 100..111；封存 109..111 后，从训练端点向前取 104..108。
    assert manager.raw_dict["open"][0].tolist() == pytest.approx(
        [104.0, 105.0, 106.0, 107.0, 108.0]
    )


def test_negative_end_offset_is_rejected(tmp_path):
    path = tmp_path / "BTCUSDT_5m.parquet"
    _write_parquet(path)
    with pytest.raises(ValueError, match="end_offset"):
        ParquetDataManager(path, end_offset=-1)
