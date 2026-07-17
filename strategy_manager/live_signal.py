"""实时信号计算：因子公式 + 实时 K 线 → 方向 + 强度。

与回测走完全相同的计算链（compute_features → StackVM → tanh → 阈值），
保证实时信号与回测/训练目标一致。信号取最后一根已收盘 bar。
"""
from __future__ import annotations

import math
from typing import Any

import torch

from model_core.features import MT5FeatureEngineer
from model_core.vm import StackVM

# 与回测/实盘共用的无信号阈值（Config.MIN_TRADE_EXPOSURE）
try:
    from config import Config
    _MIN_EXPOSURE = float(getattr(Config, "MIN_TRADE_EXPOSURE", 0.05))
except Exception:  # noqa: BLE001
    _MIN_EXPOSURE = 0.05

# 特征滚动窗口需要足够历史才能稳定（_NORM_WINDOW=200 等）
MIN_BARS = 200

_VM = StackVM()

DIR_LONG = "LONG"
DIR_SHORT = "SHORT"
DIR_FLAT = "FLAT"


def min_exposure() -> float:
    return _MIN_EXPOSURE


def evaluate_signal(formula: list[int], raw_dict: dict[str, Any]) -> dict[str, Any]:
    """在实时 K 线上计算因子信号。

    Args:
        formula:  策略因子的 token 序列。
        raw_dict: {open,high,low,close,volume} torch 张量 [1, T]，升序。

    Returns:
        dict：state / direction / strength / factor_value / position / bars_used / message
    """
    close = raw_dict.get("close")
    if close is None or close.ndim != 2:
        return {"state": "error", "message": "行情数据格式无效"}

    n_bars = int(close.shape[1])
    if n_bars < MIN_BARS:
        return {
            "state": "insufficient",
            "bars_used": n_bars,
            "message": f"历史 bar 不足（{n_bars}/{MIN_BARS}），无法稳定计算特征",
        }

    try:
        feats = MT5FeatureEngineer.compute_features(raw_dict)  # [1, F, T]
    except Exception as exc:  # noqa: BLE001
        return {"state": "error", "bars_used": n_bars, "message": f"特征计算失败: {exc}"}

    try:
        factor = _VM.execute([int(t) for t in formula], feats)  # [1, T] or None
    except Exception as exc:  # noqa: BLE001
        return {"state": "error", "bars_used": n_bars, "message": f"公式执行失败: {exc}"}

    if factor is None or factor.ndim != 2 or factor.shape[1] == 0:
        return {"state": "error", "bars_used": n_bars, "message": "公式无有效输出"}

    factor_last = float(factor[0, -1])
    if not math.isfinite(factor_last):
        return {"state": "error", "bars_used": n_bars, "message": "因子值非有限"}

    position = math.tanh(factor_last)          # 连续仓位 [-1, 1]
    strength = abs(position)                    # 信号强度 [0, 1]
    thr = _MIN_EXPOSURE

    if position >= thr:
        direction = DIR_LONG
    elif position <= -thr:
        direction = DIR_SHORT
    else:
        direction = DIR_FLAT

    return {
        "state": "ok",
        "direction": direction,
        "strength": round(strength, 4),
        "position": round(position, 4),
        "factor_value": round(factor_last, 6),
        "threshold": thr,
        "bars_used": n_bars,
        "message": "",
    }
