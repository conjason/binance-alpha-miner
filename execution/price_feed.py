"""
execution/price_feed.py — MT5 实时价格获取模块

MT5PriceFeed 负责通过 MetaTrader5 Python API 获取指定品种的最新 bid/ask 报价。
全同步接口，无 asyncio（Req 7.4）。
"""
try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    # 测试环境占位，无需真实 MT5 安装
    class _MT5Stub:
        def symbol_info_tick(self, symbol):  # noqa: D401
            return None

        def last_error(self):
            return (0, "MT5 not available")

    mt5 = _MT5Stub()

from loguru import logger


class MT5PriceFeed:
    """从 MT5 终端获取实时 bid/ask/mid 报价。

    所有方法均为同步调用，符合 MetaTrader5 Python API 同步特性（Req 7.4）。
    """

    @staticmethod
    def get_tick(symbol: str) -> dict | None:
        """获取指定品种的最新报价。

        调用 ``mt5.symbol_info_tick(symbol)`` 取得当前 tick 数据，计算 mid
        价格并以字典形式返回（Req 7.1、7.2）。

        若 tick 数据无法获取（symbol 不存在、MT5 未连接等），记录警告日志并
        返回 None（Req 7.3）。无论日志记录本身是否成功，均保证返回 None。

        Args:
            symbol: MT5 品种标识符，例如 ``"XAUUSD"``、``"EURUSD"``。

        Returns:
            成功时返回::

                {"bid": float, "ask": float, "mid": float}

            失败时返回 ``None``。
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            try:
                logger.warning(
                    f"MT5PriceFeed: symbol_info_tick('{symbol}') returned None"
                )
            except Exception:  # pragma: no cover — 日志失败不影响返回值
                pass
            return None

        bid: float = float(tick.bid)
        ask: float = float(tick.ask)
        mid: float = (bid + ask) / 2.0

        return {"bid": bid, "ask": ask, "mid": mid}
