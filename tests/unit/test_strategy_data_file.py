"""Strategy data_file fallback from training settings."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from web import strategy_file as sf
from web.settings import save_settings


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    strategies = tmp_path / "strategies"
    strategies.mkdir()
    settings_path = tmp_path / "web_settings.json"
    import web.progress as progress_mod

    monkeypatch.setattr(progress_mod, "STRATEGIES_DIR", strategies)
    monkeypatch.setattr(sf, "STRATEGIES_DIR", strategies)
    monkeypatch.setattr(sf, "checkpoint_glob", lambda _symbol: [])
    import web.settings as settings_mod

    monkeypatch.setattr(settings_mod, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(settings_mod, "STRATEGIES_DIR", strategies)
    monkeypatch.chdir(tmp_path)

    def fake_inspect(path: str) -> dict:
        return {"symbol": "XAUUSD", "timeframe": "H1", "bars": 100, "valid": True}

    monkeypatch.setattr(sf, "inspect_parquet_file", fake_inspect)
    return tmp_path


def test_inspect_strategy_fills_data_file_from_settings(project: Path) -> None:
    parquet = project / "XAUUSD_H1.parquet"
    parquet.write_bytes(b"PAR1")
    save_settings({"last_data_file": str(parquet.resolve())})

    strat_path = project / "strategies" / "best_XAUUSD.json"
    strat_path.write_text(
        json.dumps({"symbol": "XAUUSD", "formula": [1, 2, 3], "best_score": 1.5}),
        encoding="utf-8",
    )

    info = sf.inspect_strategy_file(str(strat_path))
    assert info["data_file"] == str(parquet.resolve())
    assert info["data_file_exists"] is True
    assert info["timeframe"] == "H1"


def test_sync_best_writes_data_file(project: Path) -> None:
    parquet = project / "XAUUSD_H1.parquet"
    parquet.write_bytes(b"PAR1")
    save_settings({"last_data_file": str(parquet.resolve())})

    strat_path = project / "strategies" / "best_XAUUSD.json"
    strat_path.write_text(
        json.dumps(
            {
                "symbol": "XAUUSD",
                "formula": [1, 2, 3],
                "best_score": 1.5,
                "formula_decoded": "A → B",
            }
        ),
        encoding="utf-8",
    )

    info = sf.sync_best_strategy_for_symbol("XAUUSD")
    assert info is not None
    assert info["data_file"] == str(parquet.resolve())

    saved = json.loads(strat_path.read_text(encoding="utf-8"))
    assert saved["data_file"] == str(parquet.resolve())
