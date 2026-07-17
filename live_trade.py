"""
live_trade.py — 自动交易启动脚本

默认品种（2026-07-08）：
  XAUUSD（precious_metals_v1，固定 0.01 手）+ 4 个指数有效因子

已停用、永不自动交易：
  XAGUSD — 白银合约乘数过大，实盘盈亏波动远超指数，2026-07-08 起排除。
"""
import sys
import os
import json

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from config import Config
from strategy_manager.runner import MT5StrategyRunner
from loguru import logger

# 显式默认（不依赖 scan 自动纳入 XAGUSD）
_DEFAULT_SYMBOLS = [
    "XAUUSD",
    "US100.cash",
    "US500.cash",
    "US2000.cash",
    "US30.cash",
]


def _apply_trade_filters(symbols: list[str]) -> list[str]:
    """去掉禁用品种，确保 XAUUSD 在列表中（若有策略）。"""
    excluded = set(getattr(Config, "EXCLUDED_TRADE_SYMBOLS", []) or [])
    out = [s for s in symbols if s not in excluded]
    # 黄金：始终启用（有 best_XAUUSD.json）
    if "XAUUSD" not in out and "XAUUSD" not in excluded:
        out.insert(0, "XAUUSD")
    return out


def _load_valid_from_scan() -> list[str] | None:
    p = os.path.join(_dir, "backtest_output", "factor_scan.json")
    if not os.path.exists(p):
        return None
    try:
        data = json.load(open(p, encoding="utf-8"))
        syms = [r["symbol"] for r in data.get("valid", []) if r.get("valid")]
        return _apply_trade_filters(syms) or None
    except Exception:
        return None


def main():
    dry_run = "--dry-run" in sys.argv
    single  = "--single"  in sys.argv
    sym_override = None
    if "--symbols" in sys.argv:
        idx = sys.argv.index("--symbols")
        sym_override = sys.argv[idx + 1 :]

    if sym_override:
        Config.SYMBOLS = _apply_trade_filters(sym_override)
    else:
        scanned = _load_valid_from_scan()
        Config.SYMBOLS = scanned or _DEFAULT_SYMBOLS

    logger.info(f"[live_trade] 交易品种: {Config.SYMBOLS}")
    excluded = getattr(Config, "EXCLUDED_TRADE_SYMBOLS", [])
    if excluded:
        logger.info(f"[live_trade] 已禁用（永不自动交易）: {excluded}")

    if dry_run:
        logger.info("[live_trade] DRY RUN 模式：只打印信号，不下单")
    if single:
        logger.info("[live_trade] 单公式模式：所有品种共用 best_mt5_strategy.json")

    logger.info("=" * 60)
    logger.info("  AlphaGPT 自动交易 [XAUUSD + 指数有效因子]")
    logger.info(f"  品种:     {Config.SYMBOLS}")
    logger.info(f"  周期:     H1")
    logger.info(f"  XAUUSD:   best_XAUUSD.json (precious_metals_v1) 固定 0.01 手")
    logger.info(f"  XAGUSD:   已停用，不自动交易")
    logger.info(
        f"  仓位:     以 {Config.VOL_TARGET_REFERENCE_SYMBOL} "
        f"{Config.VOL_TARGET_REFERENCE_LOT} 手的一根 ATR 美元波动为基准"
    )
    logger.info(f"  信号模式: {Config.SIGNAL_MODE}")
    logger.info("=" * 60)

    runner = MT5StrategyRunner()
    try:
        runner.run()
    except KeyboardInterrupt:
        logger.info("[live_trade] 收到 Ctrl+C，正在停止...")
    finally:
        runner.shutdown()
        logger.info("[live_trade] 已停止。")


if __name__ == "__main__":
    main()
