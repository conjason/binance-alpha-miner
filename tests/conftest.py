"""Shared pytest fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_web_settings(tmp_path, monkeypatch):
    """Never let unit tests overwrite the real web_settings.json."""
    settings_path = tmp_path / "web_settings.json"
    monkeypatch.setattr("web.settings.SETTINGS_PATH", settings_path)
