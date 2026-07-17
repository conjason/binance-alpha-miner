"""
tests/unit/test_risk.py — MT5RiskEngine 单元测试

Requirements: 9.6
"""

from unittest.mock import MagicMock, patch

import pytest

from strategy_manager.risk import MT5RiskEngine


def _make_symbol_info(
    trade_tick_value: float = 1.0,
    volume_step: float = 0.01,
    volume_min: float = 0.01,
    volume_max: float = 100.0,
) -> MagicMock:
    """构造品种规格 mock。"""
    info = MagicMock()
    info.trade_tick_value = trade_tick_value
    info.volume_step = volume_step
    info.volume_min = volume_min
    info.volume_max = volume_max
    return info


def _make_account_info(margin_free: float) -> MagicMock:
    """构造账户信息 mock。"""
    acct = MagicMock()
    acct.margin_free = margin_free
    return acct


class TestInsufficientMarginReturnsZero:
    """Requirement 9.6: 保证金不足时返回 0.0 并记录 WARNING。"""

    def test_insufficient_free_margin_returns_zero(self):
        """
        保证金极低（0.001）时 calculate_lot() 必须返回 0.0。

        保证金检查在手数计算之后执行（9.6）。
        """
        engine = MT5RiskEngine(risk_per_trade=0.01)

        mock_symbol_info = _make_symbol_info()
        mock_acct = _make_account_info(margin_free=0.001)

        with (
            patch.object(engine, "_get_symbol_info", return_value=mock_symbol_info),
            patch.object(engine, "_get_account_info", return_value=mock_acct),
        ):
            result = engine.calculate_lot("XAUUSD", equity=10000.0, stop_pips=20.0)

        assert result == 0.0, (
            f"保证金不足时应返回 0.0，实际返回 {result}"
        )

    def test_sufficient_free_margin_returns_nonzero_lot(self):
        """
        保证金充裕（999999.0）时 calculate_lot() 应返回正手数。
        """
        engine = MT5RiskEngine(risk_per_trade=0.01)

        mock_symbol_info = _make_symbol_info()
        mock_acct = _make_account_info(margin_free=999999.0)

        with (
            patch.object(engine, "_get_symbol_info", return_value=mock_symbol_info),
            patch.object(engine, "_get_account_info", return_value=mock_acct),
        ):
            result = engine.calculate_lot("XAUUSD", equity=10000.0, stop_pips=20.0)

        assert result > 0.0, (
            f"保证金充裕时应返回正手数，实际返回 {result}"
        )

    def test_symbol_info_unavailable_returns_zero(self):
        """
        无法获取品种信息（_get_symbol_info 返回 None）时应返回 0.0。
        """
        engine = MT5RiskEngine(risk_per_trade=0.01)

        with patch.object(engine, "_get_symbol_info", return_value=None):
            result = engine.calculate_lot("XAUUSD", equity=10000.0, stop_pips=20.0)

        assert result == 0.0, (
            f"品种信息不可用时应返回 0.0，实际返回 {result}"
        )
