"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

# Hypothesis：本仓的属性测试会真实计算 77 维特征张量/时序算子，单例耗时随机器与
# 负载浮动，默认 200ms deadline 在并发跑整套时会误报 DeadlineExceeded。禁用 deadline
# 是 Hypothesis 官方对"含真实计算、时间不稳定"测试的推荐做法。
try:
    from hypothesis import HealthCheck, settings

    settings.register_profile(
        "no-deadline", deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    settings.load_profile("no-deadline")
except Exception:  # hypothesis 未安装时静默跳过
    pass


@pytest.fixture(autouse=True)
def isolate_web_settings(tmp_path, monkeypatch):
    """Never let unit tests overwrite the real web_settings.json."""
    settings_path = tmp_path / "web_settings.json"
    monkeypatch.setattr("web.settings.SETTINGS_PATH", settings_path)
