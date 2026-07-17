"""backtest_binance.py — 用已训练策略在币安 Parquet 上回测（AlphaForge 安全 fork）

信号口径与训练/上游完全一致：
    factor = StackVM.execute(公式, 特征)
    仓位   = tanh(factor)，|仓位| < MIN_TRADE_EXPOSURE 记为空仓
    每 bar pnl = 仓位 * 未来对数收益 - 换手 * (手续费+滑点)

关键点：
  - 成本默认按币安 USDT-M 合约给（单边 5bp 手续费 + 2bp 滑点），远高于上游外汇默认 1bp；
  - 用 --split 划分「样本内 / 样本外」，直接暴露过拟合（样本外掉多少）；
  - 永续资金费未计入 pnl（另有 funding 现金流），如需可另行叠加。

用法：
    python backtest_binance.py --strategy strategies\\best_BTCUSDT.json --data-file D:\\币安币种数据\\BTCUSDT.parquet --timeframe 5m
    python backtest_binance.py --strategy strategies\\best_BTCUSDT.json --data-file ... --timeframe 5m --cost 0.0005 --slippage 0.0002 --split 0.7
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from data_pipeline.parquet_manager import ParquetDataManager
from model_core.vm import StackVM
from strategy_manager.signal import compute_target_positions_stateless

_BARS_PER_YEAR = {
    "M1": 525600, "M5": 105120, "M15": 35040, "M30": 17520,
    "H1": 8760, "H4": 2190, "D1": 365,
}


def _metrics(pos: torch.Tensor, ret: torch.Tensor, cost_rate: float, bars_per_year: float) -> dict:
    """pos, ret: 1D tensors（同长度）。返回一组绩效指标。"""
    pos = torch.nan_to_num(pos.float(), nan=0.0, posinf=0.0, neginf=0.0)
    ret = torch.nan_to_num(ret.float(), nan=0.0, posinf=0.0, neginf=0.0)
    prev = torch.cat([torch.zeros(1), pos[:-1]])
    turnover = (pos - prev).abs()
    pnl = pos * ret - turnover * cost_rate       # 每 bar 对数收益贡献（近似）
    pnl = torch.nan_to_num(pnl, nan=0.0, posinf=0.0, neginf=0.0)
    eq = torch.cumsum(pnl, dim=0)                 # 累计对数收益
    equity = torch.exp(eq.clamp(-50, 50))         # 净值曲线（clamp 防 exp 溢出）

    n = pnl.numel()
    total_ret = float(torch.exp(pnl.sum().clamp(-50, 50)) - 1.0)
    ann_ret = float(torch.exp((pnl.mean() * bars_per_year).clamp(-50, 50)) - 1.0)
    std = float(pnl.std(unbiased=True))
    sharpe = float(pnl.mean() / (std + 1e-12) * math.sqrt(bars_per_year)) if std > 0 else 0.0
    downside = pnl[pnl < 0]
    dstd = float(downside.std(unbiased=True)) if downside.numel() > 1 else 0.0
    sortino = float(pnl.mean() / (dstd + 1e-12) * math.sqrt(bars_per_year)) if dstd > 0 else 0.0

    running_max = torch.cummax(equity, dim=0).values
    dd = (equity - running_max) / (running_max + 1e-12)
    max_dd = float(dd.min())  # 负数

    active = pos.abs() > 1e-9
    exposure = float(active.float().mean())
    # 一笔"交易"= 仓位方向变化次数
    sign = torch.sign(pos)
    prev_sign = torch.cat([torch.zeros(1), sign[:-1]])
    trades = int((sign != prev_sign).sum())
    # 胜率：按持仓段近似——用单 bar 盈利占比
    bar_wins = float((pnl[active] > 0).float().mean()) if active.any() else 0.0

    calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else 0.0
    return {
        "bars": n,
        "total_return": total_ret,
        "annual_return": ann_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "exposure": exposure,
        "trades": trades,
        "bar_winrate": bar_wins,
        "avg_turnover": float(turnover.mean()),
    }


def _fmt(m: dict, title: str) -> str:
    return (
        f"\n[{title}]  K线数={m['bars']}\n"
        f"  总收益率       : {m['total_return']*100:+.2f}%\n"
        f"  年化收益率     : {m['annual_return']*100:+.2f}%\n"
        f"  夏普比率       : {m['sharpe']:+.2f}\n"
        f"  索提诺比率     : {m['sortino']:+.2f}\n"
        f"  卡玛比率       : {m['calmar']:+.2f}\n"
        f"  最大回撤       : {m['max_drawdown']*100:.2f}%\n"
        f"  在场时间占比   : {m['exposure']*100:.1f}%\n"
        f"  换手方向变化   : {m['trades']} 次\n"
        f"  单bar胜率      : {m['bar_winrate']*100:.1f}%\n"
        f"  平均换手       : {m['avg_turnover']:.4f}"
    )


def run_backtest(
    strategy_file: str,
    data_file: str,
    *,
    timeframe: str | None = None,
    cost: float = 0.0005,
    slippage: float = 0.0002,
    split: float = 0.0,
    max_bars: int | None = None,
) -> dict:
    strat = json.loads(Path(strategy_file).read_text(encoding="utf-8"))
    formula = strat.get("formula")
    if not formula:
        raise ValueError(f"策略文件无 formula: {strategy_file}")
    tf = timeframe or strat.get("timeframe")
    cost_rate = float(cost) + float(slippage)

    mgr = ParquetDataManager(data_file, timeframe=tf, max_bars=max_bars)
    mgr.load()
    feat = mgr.feat_tensor                      # [1, F, T]
    ret = mgr.target_ret[0]                     # [T]

    vm = StackVM()
    factor = vm.execute([int(t) for t in formula], feat)
    if factor is None:
        raise RuntimeError("公式无法执行（可能 vocab 版本不匹配）；请用当前 vocab 重训。")
    # NaN/Inf 因子值 = 无有效信号 = 空仓（实盘也必须这样兜底）。
    # 注：单品种上使用截面算子(CS_RANK/CS_SCALE 等)会退化并在暖机区产生 NaN，
    # 这属于策略层面的缺陷（截面算子需要多品种），此处按空仓处理以得到可读回测。
    factor = torch.nan_to_num(factor, nan=0.0, posinf=0.0, neginf=0.0)
    if strat.get("vocab_version") and strat["vocab_version"] != _current_vocab():
        print(f"  [警告] 策略 vocab_version={strat['vocab_version']} 与当前不一致，结果可能无意义，建议重训。")
    pos = compute_target_positions_stateless(factor)[0]   # [T]

    bpy = _BARS_PER_YEAR.get(tf, 105120)

    print(f"\n{'='*60}")
    print(f"  AlphaForge 币安回测")
    print(f"{'='*60}")
    print(f"  品种: {strat.get('symbol')}  周期: {tf}  公式: {strat.get('formula_decoded')}")
    print(f"  成本(单边): 手续费 {cost*100:.3f}% + 滑点 {slippage*100:.3f}% = {cost_rate*100:.3f}%")

    out = {}
    out["full"] = _metrics(pos, ret, cost_rate, bpy)
    print(_fmt(out["full"], "全样本"))

    if 0.0 < split < 1.0:
        k = int(len(ret) * split)
        out["in_sample"] = _metrics(pos[:k], ret[:k], cost_rate, bpy)
        out["out_sample"] = _metrics(pos[k:], ret[k:], cost_rate, bpy)
        print(_fmt(out["in_sample"], f"样本内(前 {split*100:.0f}%)"))
        print(_fmt(out["out_sample"], f"样本外(后 {(1-split)*100:.0f}%)"))
        deg = out["in_sample"]["sharpe"] - out["out_sample"]["sharpe"]
        print(f"\n  >>> 样本外夏普较样本内变化: {-deg:+.2f}（越负=过拟合越重）")

    return out


def _current_vocab() -> str:
    from model_core.vocab import VOCAB_VERSION
    return VOCAB_VERSION


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="AlphaForge 币安回测")
    ap.add_argument("--strategy", required=True, help="策略 JSON，如 strategies\\best_BTCUSDT.json")
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--cost", type=float, default=0.0005, help="单边手续费率，默认 0.0005=5bp")
    ap.add_argument("--slippage", type=float, default=0.0002, help="单边滑点率，默认 0.0002=2bp")
    ap.add_argument("--split", type=float, default=0.0, help="样本内比例，如 0.7；0=不切分")
    ap.add_argument("--max-bars", type=int, default=None)
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args(sys.argv[1:])
    run_backtest(
        a.strategy, a.data_file,
        timeframe=a.timeframe, cost=a.cost, slippage=a.slippage,
        split=a.split, max_bars=a.max_bars,
    )
