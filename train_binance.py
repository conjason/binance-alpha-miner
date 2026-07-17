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
import datetime as _dt
import glob as _glob
import json
import pathlib
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

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
    timeframe: str | None,
    from_scratch: bool = False,
    max_bars: int | None = None,
    holdout_bars: int = 0,
    steps: int | None = None,
    batch_size: int | None = None,
    n_folds: int = 5,
    device: str = "cpu",
    seed: int = 20260717,
    use_lord: bool = True,
    reuse_best: bool = False,
    reward_mode: str = "standard",
) -> AlphaEngine | None:
    selected_device = _configure_training_runtime(
        device=device, batch_size=batch_size, seed=seed
    )
    if steps is not None:
        if steps < 1:
            raise ValueError("steps 必须 >= 1")
        ModelConfig.TRAIN_STEPS = int(steps)
    if n_folds < 2:
        raise ValueError("folds 必须 >= 2")
    if holdout_bars < 0:
        raise ValueError("holdout-bars 不能为负数")
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
    print(
        f"  训练步数: {ModelConfig.TRAIN_STEPS}   批量: {ModelConfig.BATCH_SIZE}"
        f"   折数: {n_folds}   奖励模式: {reward_mode}"
    )
    print(f"  设备: {selected_device}   随机种子: {seed}   LoRD: {'开' if use_lord else '关'}")
    print(f"  K线数(全量): {info['bars']}" + (f"，截断到近 {max_bars} 根" if max_bars else ""))
    if holdout_bars:
        print(f"  真样本外封存: 最近 {holdout_bars} 根（训练完全不可见）")
    print(f"  模式: {'重新训练（从头）' if from_scratch else '自动续训'}")
    print(f"{'='*60}")

    try:
        mgr = ParquetDataManager(
            data_file,
            timeframe=timeframe,
            max_bars=max_bars,
            end_offset=holdout_bars,
        )
        mgr.load()
        T = mgr.raw_dict["open"].shape[1]
        print(f"  数据加载成功，共 {T} 根K线；实际启用另类列: {mgr.extra_fields or '无'}")
        train_window = _manager_time_window(mgr)
        print(f"  训练窗口: {train_window['start']} → {train_window['end']}")
    except Exception as e:
        print(f"  [错误] 数据加载失败: {e}")
        return None

    engine = AlphaEngine(
        data_manager=mgr,
        target_symbol=symbol,
        n_folds=n_folds,
        use_lord_regularization=use_lord,
    )
    engine.timeframe = tf
    engine.data_file = str(Path(data_file).resolve())
    engine.mode = "binance_parquet"
    engine.train_steps = ModelConfig.TRAIN_STEPS
    engine.training_config = {
        "max_bars": max_bars,
        "holdout_bars": holdout_bars,
        "batch_size": ModelConfig.BATCH_SIZE,
        "folds": n_folds,
        "device": str(selected_device),
        "seed": seed,
        "reward_mode": reward_mode,
        "lord_regularization": use_lord,
        "reuse_best": reuse_best,
    }
    engine.train_window = train_window

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
        if reuse_best:
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
        parsed = [int(t) for t in formula]
        if not engine.sampler.formula_allowed(parsed):
            print("  [重新训练] 旧公式含单品种禁用的截面 token，忽略旧公式下限")
            return
        engine.best_formula = parsed
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
        "training_config": getattr(engine, "training_config", None),
        "train_window": getattr(engine, "train_window", None),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  策略已保存: {path}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="AlphaForge 币安 Parquet 训练")
    ap.add_argument("--data-file", required=True, help="parquet 路径，如 D:\\币安币种数据\\BTCUSDT.parquet")
    ap.add_argument("--timeframe", default=None, help="周期，如 5m/1m/H1（文件名无周期后缀时必填）")
    ap.add_argument("--from-scratch", action="store_true", help="清除检查点从头训练")
    ap.add_argument("--max-bars", type=int, default=None, help="只用最近 N 根 K 线")
    ap.add_argument(
        "--holdout-bars", type=int, default=0,
        help="封存末尾 N 根作为真样本外，训练完全不可见",
    )
    ap.add_argument("--steps", type=int, default=None, help="覆盖训练步数")
    ap.add_argument("--batch-size", type=int, default=None, help="每步采样公式数")
    ap.add_argument("--folds", type=int, default=5, help="walk-forward 分段数，实际验证折数为 folds-1")
    ap.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "auto"],
        help="训练设备；默认 cpu，auto 在 CUDA 可用时选 cuda",
    )
    ap.add_argument("--seed", type=int, default=20260717, help="随机种子")
    ap.add_argument("--no-lord", action="store_true", help="关闭 LoRD 正则化（用于性能对照）")
    ap.add_argument(
        "--reuse-best", action="store_true",
        help="从头训练时仍复用同配置旧冠军；默认真正清零，避免跨窗口分数污染",
    )
    ap.add_argument("--reward-mode", default="standard", choices=["standard", "ftmo", "forex"],
                    help="奖励模式，默认 standard（币安合约推荐）")
    return ap.parse_args(argv)


def _configure_training_runtime(
    *, device: str, batch_size: int | None, seed: int
) -> torch.device:
    requested = device.lower().strip()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前 PyTorch 未检测到可用 CUDA 设备")
    selected = torch.device(requested)
    if batch_size is not None:
        if batch_size < 4:
            raise ValueError("batch-size 必须 >= 4")
        ModelConfig.BATCH_SIZE = int(batch_size)
    ModelConfig.DEVICE = selected
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return selected


def _manager_time_window(mgr: ParquetDataManager) -> dict[str, object]:
    ts = mgr.raw_dict["time"][0]
    tz = _dt.timezone(_dt.timedelta(hours=8))

    def _fmt(value: int) -> str:
        return _dt.datetime.fromtimestamp(value, tz=_dt.timezone.utc).astimezone(tz).isoformat()

    return {
        "start": _fmt(int(ts[0].item())),
        "end": _fmt(int(ts[-1].item())),
        "bars": int(ts.numel()),
    }


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    t0 = time.time()
    eng = train_from_file(
        args.data_file,
        timeframe=args.timeframe,
        from_scratch=args.from_scratch,
        max_bars=args.max_bars,
        holdout_bars=args.holdout_bars,
        steps=args.steps,
        batch_size=args.batch_size,
        n_folds=args.folds,
        device=args.device,
        seed=args.seed,
        use_lord=not args.no_lord,
        reuse_best=args.reuse_best,
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
