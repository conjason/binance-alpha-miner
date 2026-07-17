"""Probe whether this machine can reach TradingView via tvdatafeed.

Ported from PA_Agent ``pa_agent.data.tradingview_connectivity``.
"""
from __future__ import annotations

import concurrent.futures
import logging
import time

logger = logging.getLogger(__name__)

_DEFAULT_PROBE_TIMEOUT_S = 20.0
_DEFAULT_PROBE_ATTEMPTS = 3
_DEFAULT_RETRY_DELAY_S = 3.0

# Frontend / watch message sentinel — mirrors PA_Agent VPN dialog trigger
TV_CONNECTIVITY_BLOCKED = "TV_CONNECTIVITY_BLOCKED"

TV_CLOUD_SERVER_WIKI_URL = "https://my.feishu.cn/wiki/FuqnwkPwdiCLhQkPloKc7r1lntg"

TV_CONNECTIVITY_MESSAGE = (
    "当前设备无法连接 TradingView 数据服务，将无法获取以下 K 线数据：\n"
    "  · A 股（上证 SSE、深证 SZSE）\n"
    "  · 港股（HKEX）\n"
    "  · 美股及指数（NYSE、NASDAQ、SP）\n"
    "  · 外汇、贵金属、商品期货\n\n"
    "解决方案：\n"
    "  · 把你的VPN工具设成全局，并开启TUN(虚拟网卡)模式，如果还不行：\n"
    "  · 使用云服务器部署本程序（推荐）—— 云服务器可正常连接 TradingView\n"
    "  · 或切换回 MT5 数据源，仅使用 MT5 提供的品种数据"
)


def _probe_once(*, timeout_s: float) -> tuple[bool, str | None, bool]:
    """Single probe. Returns (ok, failure_detail, retryable)."""

    def _probe() -> None:
        from tvDatafeed import Interval, TvDatafeed  # type: ignore[import]

        tv = TvDatafeed()
        try:
            setattr(tv, "_TvDatafeed__ws_timeout", min(10.0, float(timeout_s)))
        except Exception:
            pass
        df = tv.get_hist(
            symbol="XAUUSD",
            exchange="OANDA",
            interval=Interval.in_1_minute,
            n_bars=2,
        )
        if df is None or getattr(df, "empty", True):
            raise RuntimeError("TradingView 返回空数据")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_probe)
            fut.result(timeout=timeout_s)
        return True, None, False
    except concurrent.futures.TimeoutError:
        logger.warning(
            "TradingView connectivity probe timed out after %.0fs", timeout_s
        )
        return False, "连接超时", True
    except ImportError as exc:
        logger.warning(
            "TradingView connectivity probe: tvDatafeed not installed: %s", exc
        )
        return False, str(exc), False
    except Exception as exc:  # noqa: BLE001
        logger.warning("TradingView connectivity probe failed: %s", exc)
        return False, str(exc), True


def check_tradingview_connectivity(
    *,
    timeout_s: float = _DEFAULT_PROBE_TIMEOUT_S,
    max_attempts: int = _DEFAULT_PROBE_ATTEMPTS,
    retry_delay_s: float = _DEFAULT_RETRY_DELAY_S,
) -> tuple[bool, str | None]:
    """Try a minimal OANDA:XAUUSD fetch with retries; return (ok, failure_detail)."""
    attempts = max(1, int(max_attempts))
    last_detail: str | None = None

    for attempt in range(1, attempts + 1):
        ok, detail, retryable = _probe_once(timeout_s=timeout_s)
        if ok:
            return True, None

        last_detail = detail
        if not retryable or attempt >= attempts:
            break
        time.sleep(max(0.0, float(retry_delay_s)))

    return False, last_detail
