"""train_binance.py — 从单个币安 Parquet K 线文件训练（AlphaForge 安全 fork）

与上游 train_file.py 的区别：
  - 支持无周期后缀的文件名（币安数据是 ``BTCUSDT.parquet`` 这种），用 --timeframe 指定；
  - 支持 --max-bars 截断到近端窗口（5m 全量 40 万+ 根，逐步滚动算子会很慢）；
  - --steps 覆盖训练步数；--reward-mode 默认 standard（上游默认 ftmo 是外汇 prop 考试盘专属）。

用法示例：
    python train_binance.py --data-file D:\\币安币种数据\\BTCUSDT.parquet --timeframe 5m
    python train_binance.py --data-file D:\\币安币种数据\\BTCUSDT.parquet --timeframe 5m --max-bars 60000 --steps 3000
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import pathlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.train_logging import configure_train_stdio

configure_train_stdio()

from config import Config
from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine
from model_core.vocab import VOCAB_VERSION


def train_from_file(
    data_file: str,
    *,
    timeframe: str,
    from_scratch: bool = False,
    max_bars: int | None = None,
    steps: int | None = None,
    reward_mode: str = "standard",
) -> AlphaEngine | None:
    if steps is not None:
        ModelConfig.TRAIN_STEPS = int(steps)
    ModelConfig.REWARD_MODE = reward_mode

    info = inspect_parquet_file(data_file, timeframe=timeframe)
    symbol = info["symbol"]
    tf = info["timeframe"]

    print(f"\n{'='*60}")
    print(f"  AlphaForge 币安训练 — {info['filename']}")
    print(f"{'='*60}")
    print(f"  品种: {symbol}")
    print(f"  周期: {tf}")
    print(f"  另类数据列: {info.get('extra_fields') or '无（仅 OHLCV）'}")
    print(f"  文件: {Path(data_file).resolve()}")
    print(f"  训练步数: {ModelConfig.TRAIN_STEPS}   奖励模式: {reward_mode}")
    print(f"  K线数(全量): {info['bars']}" + (f"，截断到近 {max_bars} 根" if max_bars else ""))
    print(f"  模式: {'重新训练（从头）' if from_scratch else '自动续训'}")
    print(f"{'='*60}")

    try:
        mgr = ParquetDataManager(data_file, timeframe=timeframe, max_bars=max_bars)
        mgr.load()
        T = mgr.raw_dict["open"].shape[1]
        print(f"  数据加载成功，共 {T} 根K线；实际启用另类列: {mgr.extra_fields or '无'}")
    except Exception as e:
        print(f"  [错误] 数据加载失败: {e}")
        return None

    engine = AlphaEngine(data_manager=mgr, target_symbol=symbol)
    engine.timeframe = tf
    engine.data_file = str(Path(data_file).resolve())
    engine.mode = "binance_parquet"
    engine.train_steps = ModelConfig.TRAIN_STEPS

    ckpt_pattern = str(pathlib.Path("checkpoints") / f"ckpt_{symbol}_step_*.pt")
    ckpt_files = sorted(_glob.glob(ckpt_pattern))
    start_step = 0

    if from_scratch:
        for p in ckpt_files:
            try:
                pathlib.Path(p).unlink(missing_ok=True)
            except OSError as e:
                print(f"  [警告] 无法删除检查点 {p}: {e}")
        hist_path = pathlib.Path(f"training_history_{symbol}.json")
        if hist_path.exists():
            hist_path.unlink(missing_ok=True)
        print("  [重新训练] 已清除检查点，从第 0 步开始")
        _seed_best_from_strategy(engine, symbol)
        ckpt_files = []
    elif ckpt_files:
        latest = ckpt_files[-1]
        try:
            start_step = engine.load_checkpoint(latest)
            print(f"  [续训] 从 {latest} 恢复，起始步={start_step}")
        except Exception as e:
            print(f"  [警告] 检查点加载失败: {e}，将从头开始")

    if start_step >= ModelConfig.TRAIN_STEPS:
        print(f"  [完成] {symbol} 已完成全部 {ModelConfig.TRAIN_STEPS} 步，跳过训练")
        _save_strategy(engine, symbol, tf, data_file)
        return engine

    engine.train(start_step=start_step)
    _save_strategy(engine, symbol, tf, data_file)
    return engine


def _seed_best_from_strategy(engine: AlphaEngine, symbol: str) -> None:
    path = pathlib.Path("strategies") / f"best_{symbol}.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    formula = data.get("formula")
    score = data.get("best_score")
    if not formula or score is None:
        return
    # vocab 版本不一致时不要沿用旧公式（token 语义已变）
    if data.get("vocab_version") and data.get("vocab_version") != VOCAB_VERSION:
        print("  [重新训练] 检测到旧 vocab_version，忽略旧公式下限")
        return
    try:
        engine.best_formula = [int(t) for t in formula]
        engine.best_score = float(score)
        print(f"  [重新训练] 保留已有最优分数下限={engine.best_score:.4f}")
    except (TypeError, ValueError):
        pass


def _save_strategy(engine: AlphaEngine, symbol: str, timeframe: str, data_file: str) -> None:
    path = pathlib.Path("strategies") / f"best_{symbol}.json"
    path.parent.mkdir(exist_ok=True)
    data = {
        "vocab_version": VOCAB_VERSION,
        "symbol": symbol,
        "timeframe": timeframe,
        "data_file": str(Path(data_file).resolve()),
        "mode": "binance_parquet",
        "formula": engine.best_formula,
        "formula_decoded": engine._decode_formula(engine.best_formula)
        if engine.best_formula
        else None,
        "best_score": engine.best_score,
        "train_steps": ModelConfig.TRAIN_STEPS,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  策略已保存: {path}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AlphaForge 币安 Parquet 训练")
    ap.add_argument("--data-file", required=True, help="parquet 路径，如 D:\\币安币种数据\\BTCUSDT.parquet")
    ap.add_argument("--timeframe", default=None, help="周期，如 5m/1m/H1（文件名无周期后缀时必填）")
    ap.add_argument("--from-scratch", action="store_true", help="清除检查点从头训练")
    ap.add_argument("--max-bars", type=int, default=None, help="只用最近 N 根 K 线")
    ap.add_argument("--steps", type=int, default=None, help="覆盖训练步数")
    ap.add_argument("--reward-mode", default="standard", choices=["standard", "ftmo", "forex"],
                    help="奖励模式，默认 standard（币安合约推荐）")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    t0 = time.time()
    eng = train_from_file(
        args.data_file,
        timeframe=args.timeframe,
        from_scratch=args.from_scratch,
        max_bars=args.max_bars,
        steps=args.steps,
        reward_mode=args.reward_mode,
    )
    elapsed = time.time() - t0
    if eng:
        sym = eng.target_symbol or "?"
        print(f"\n<<< [{sym}] 训练完成: 最优分数={eng.best_score:.4f}，耗时 {elapsed/60:.1f} 分钟")
        if eng.best_formula:
            print(f"    {eng._decode_formula(eng.best_formula)}")
    else:
        print("\n<<< 训练失败")
        sys.exit(1)
