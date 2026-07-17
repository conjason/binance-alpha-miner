"""
download_full_history.py — 从 MT5 下载所有品种的全量历史 H1 数据

原理：
    MT5 copy_rates_from_pos 单次最多 50000 根。
    本脚本用 50000 根拉取（对 H1 约覆盖 2018~至今，共 8 年）。

用法：
    python download_full_history.py              # 下载 Config.SYMBOLS 里的所有品种
    python download_full_history.py EURUSD XAUUSD   # 只下载指定品种

输出：D:/K线数据/{symbol}_H1.parquet
"""
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    print("ERROR: MetaTrader5 package not installed")
    sys.exit(1)

from config import Config

CACHE_DIR  = Path(Config.KLINE_CACHE_DIR)
TIMEFRAME  = mt5.TIMEFRAME_H1
_COLUMNS   = ["time", "open", "high", "low", "close", "tick_volume"]

# MT5 单次请求上限（实测 50000 是安全上限）
MAX_BARS_PER_REQUEST = 50000


def download_symbol(symbol: str) -> int:
    """下载单个品种的全量 H1 历史数据，返回总 bar 数。0 表示失败。"""
    path = CACHE_DIR / f"{symbol}_H1.parquet"

    # 检查 MT5 是否知道这个品种
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.warning(f"[{symbol}] MT5 不认识此品种，跳过")
        return 0

    # 先确保品种在 MarketWatch 中可见
    if not info.visible:
        mt5.symbol_select(symbol, True)
        time.sleep(0.2)

    logger.info(f"[{symbol}] 开始下载（最多 {MAX_BARS_PER_REQUEST} 根 H1）...")
    t0 = time.time()

    # MT5 大请求后有限速，加重试机制
    rates = None
    for attempt in range(5):
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, MAX_BARS_PER_REQUEST)
        if rates is not None and len(rates) > 0:
            break
        err = mt5.last_error()
        if attempt < 4:
            wait = 35 * (attempt + 1)  # 35s, 70s, 105s...
            logger.warning(f"[{symbol}] 第{attempt+1}次失败 error={err}，等待 {wait}s 重试...")
            time.sleep(wait)

    if rates is None or len(rates) == 0:
        logger.warning(f"[{symbol}] 5次重试后仍无数据（error={mt5.last_error()}）")
        return 0

    df = pd.DataFrame(rates)[_COLUMNS].astype({
        "time":         "int64",
        "open":         "float32",
        "high":         "float32",
        "low":          "float32",
        "close":        "float32",
        "tick_volume":  "int64",
    })
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    date_start = datetime.utcfromtimestamp(df["time"].iloc[0]).strftime("%Y-%m-%d")
    date_end   = datetime.utcfromtimestamp(df["time"].iloc[-1]).strftime("%Y-%m-%d")
    elapsed = time.time() - t0

    logger.success(
        f"[{symbol}] {len(df):,} bars  {date_start} ~ {date_end}  "
        f"({elapsed:.1f}s)  → {path.name}"
    )
    return len(df)


def main():
    # 确定要下载的品种列表
    if len(sys.argv) > 1:
        symbols = [s for s in sys.argv[1:] if not s.startswith("--")]
    else:
        # 默认：Config.SYMBOLS + FEATURE_SYMBOLS（去重）
        symbols = list(dict.fromkeys(
            Config.SYMBOLS + getattr(Config, "FEATURE_SYMBOLS", [])
        ))

    print(f"{'='*62}")
    print(f"  MT5 全量历史数据下载")
    print(f"  品种数: {len(symbols)}")
    print(f"  保存至: {CACHE_DIR}")
    print(f"{'='*62}\n")

    # 连接 MT5
    if not mt5.initialize():
        print(f"ERROR: MT5 连接失败: {mt5.last_error()}")
        sys.exit(1)
    print(f"MT5 已连接\n")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    failed  = []

    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym}...")
        try:
            n = download_symbol(sym)
            results[sym] = n
            if n == 0:
                failed.append(sym)
        except Exception as e:
            logger.error(f"[{sym}] 异常: {e}")
            results[sym] = -1
            failed.append(sym)
        # 大请求限速：每个品种后等待 5 秒
        time.sleep(5)

    mt5.shutdown()

    # 汇总报告
    print(f"\n{'='*62}")
    print(f"  下载完成")
    print(f"{'='*62}")
    success = {s: n for s, n in results.items() if n > 0}
    print(f"  成功: {len(success)}/{len(symbols)} 个品种")
    if success:
        max_bars = max(success.values())
        min_bars = min(success.values())
        avg_bars = sum(success.values()) // len(success)
        print(f"  数据量: 最多 {max_bars:,} bars，最少 {min_bars:,} bars，平均 {avg_bars:,} bars")
    if failed:
        print(f"  失败品种 ({len(failed)}): {', '.join(failed)}")

    # 输出详细表格
    print(f"\n  {'品种':20s} {'bars':>8}")
    print(f"  {'-'*30}")
    for sym in symbols:
        n = results.get(sym, 0)
        status = f"{n:>8,}" if n > 0 else "  失败/无数据"
        print(f"  {sym:20s} {status}")
    print()


if __name__ == "__main__":
    main()
