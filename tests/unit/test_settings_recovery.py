"""Recover invalid last_data_file paths in web settings."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import web.settings as settings_mod
from web.settings import load_settings, save_settings


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    monkeypatch.setattr(settings_mod, "STRATEGIES_DIR", strategies)
    return tmp_path


def test_load_settings_recovers_missing_temp_path(project: Path) -> None:
    parquet = project / "XAUUSD_H1.parquet"
    parquet.write_bytes(b"PAR1")
    strat = project / "strategies" / "best_XAUUSD.json"
    strat.write_text(
        json.dumps({"symbol": "XAUUSD", "data_file": str(parquet.resolve())}),
        encoding="utf-8",
    )
    settings_path = settings_mod.SETTINGS_PATH
    settings_path.write_text(
        json.dumps(
            {
                "last_data_file": r"C:\Users\x\AppData\Local\Temp\2\pytest-of-x\test0\XAUUSD_H1.parquet",
                "last_strategy_file": str(strat.resolve()),
            }
        ),
        encoding="utf-8",
    )

    loaded = load_settings()
    assert loaded["last_data_file"] == str(parquet.resolve())
    if settings_mod._is_production_settings_path():
        persisted = json.loads(settings_path.read_text(encoding="utf-8"))
        assert persisted["last_data_file"] == str(parquet.resolve())


def test_save_settings_ignores_ephemeral_path(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prod_settings = project / "web_settings.json"
    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", prod_settings)
    monkeypatch.setattr(settings_mod, "PROJECT_ROOT", project)
    save_settings({"last_data_file": "D:\\real\\XAUUSD_H1.parquet"})
    save_settings(
        {
            "last_data_file": r"C:\Users\x\AppData\Local\Temp\2\pytest-of-x\test0\X.parquet"
        }
    )
    loaded = load_settings()
    assert loaded["last_data_file"] == "D:\\real\\XAUUSD_H1.parquet"
