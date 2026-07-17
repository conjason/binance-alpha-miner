"""
generate_factor_equity_curves.py — 各品种 best_{symbol}.json 因子深度回测 + 资金曲线

数据：D:\\K线数据 离线 H1（可用 --mt5 连 MT5 增量更新）
输出：backtest_output/equity_{symbol}.png
      backtest_output/all_factors_equity.png
      backtest_output/factor_equity_summary.json
"""
from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.features import MT5FeatureEngineer
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from model_core.vm import StackVM
from strategy_manager.signal import compute_target_positions_stateless

_H1_PER_YEAR = 6240
OUTPUT_DIR = Path("backtest_output")
SKIP_STEMS = {"index", "precious_metals", "forex", "metals_comm"}

COST = {"metals": 0.00020, "index": 0.00030, "forex": 0.00015}


def decode(tokens: list[int]) -> str:
    names = FORMULA_VOCAB.token_names
    return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}" for t in tokens)


def calc_sharpe(pnl: np.ndarray) -> float:
    m, s = pnl.mean(), pnl.std()
    return float(m / (s + 1e-10) * math.sqrt(_H1_PER_YEAR))


def calc_sortino(pnl: np.ndarray) -> float:
    m = pnl.mean()
    down = pnl[pnl < 0]
    ds = down.std() if len(down) > 0 else 1e-10
    ds = max(ds, abs(m), 1e-10)
    return float(np.clip(m / ds * math.sqrt(_H1_PER_YEAR), -20, 20))


def calc_mdd(cum: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum)
    return float((peak - cum).max())


def calc_annual_return(total_ret: float, T: int) -> float:
    return float(total_ret / (T / _H1_PER_YEAR)) if T > 0 else 0.0


def _cost(sym: str) -> float:
    if sym.startswith("XAU") or sym.startswith("XAG"):
        return COST["metals"]
    if ".cash" in sym or sym.startswith("US") or sym.startswith("JP"):
        return COST["index"]
    return COST["forex"]


def _safe_filename(sym: str) -> str:
    return re.sub(r"[^\w.\-]", "_", sym)


def load_best(path: Path) -> dict | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("vocab_version") != VOCAB_VERSION:
        return None
    formula = data.get("formula") or data.get("formula_tokens")
    if not formula:
        return None
    sym = data.get("symbol")
    if not sym:
        m = re.match(r"best_(.+)\.json", path.name)
        sym = m.group(1) if m else None
    if not sym or sym in SKIP_STEMS:
        return None
    tokens = [int(t) for t in formula]
    return {
        "symbol": sym,
        "formula": tokens,
        "path": path.name,
        "readable": data.get("formula_readable") or decode(tokens),
        "best_score": data.get("best_score"),
        "live_trading": data.get("live_trading"),
        "status": data.get("status"),
        "disabled_reason": data.get("disabled_reason"),
    }


def solo_backtest(
    formula: list[int],
    raw_dict: dict,
    target_ret: torch.Tensor,
    symbol: str,
    cost_rate: float,
) -> dict | None:
    feat = MT5FeatureEngineer.compute_features(raw_dict)
    vm = StackVM()
    factor = vm.execute(formula, feat)
    if factor is None:
        return None

    pos = compute_target_positions_stateless(factor)
    pos_np = pos.detach().cpu().numpy().squeeze(0)
    target_np = target_ret.detach().cpu().numpy().squeeze(0)

    prev = np.zeros_like(pos_np)
    prev[1:] = pos_np[:-1]
    turnover = np.abs(pos_np - prev)
    pnl = pos_np * target_np - turnover * cost_rate
    T = len(pnl)
    cum = np.cumsum(pnl)

    half = T // 2
    p1, p2 = pnl[:half], pnl[half:]

    years_n = max(1, T // _H1_PER_YEAR)
    yearly_ret = [float(pnl[y * _H1_PER_YEAR:min((y + 1) * _H1_PER_YEAR, T)].sum()) for y in range(years_n)]

    total_ret = float(cum[-1])
    mdd = calc_mdd(cum)
    ann = calc_annual_return(total_ret, T)

    return {
        "T": T,
        "years_span": T / _H1_PER_YEAR,
        "readable": decode(formula),
        "pnl": pnl,
        "cum": cum,
        "total_ret": total_ret,
        "ann_ret": ann,
        "sharpe": calc_sharpe(pnl),
        "sortino": calc_sortino(pnl),
        "mdd": mdd,
        "calmar": ann / (mdd + 1e-8),
        "half1_ret": float(np.cumsum(p1)[-1]) if len(p1) else 0.0,
        "half2_ret": float(np.cumsum(p2)[-1]) if len(p2) else 0.0,
        "half1_sharpe": calc_sharpe(p1) if len(p1) else 0.0,
        "half2_sharpe": calc_sharpe(p2) if len(p2) else 0.0,
        "yearly_ret": yearly_ret,
        "per_sym": {symbol: {"pnl": pnl, "cum": cum, "ann_ret": ann, "sharpe": calc_sharpe(pnl), "mdd": mdd}},
    }


def _date_labels(times: np.ndarray | None, T: int) -> tuple[list[int], list[str]] | None:
    if times is None or len(times) != T:
        return None
    step = max(1, T // 10)
    ticks = list(range(0, T, step))
    labels = [
        datetime.fromtimestamp(int(times[i]), tz=timezone.utc).strftime("%Y-%m-%d")
        for i in ticks
    ]
    return ticks, labels


def plot_solo_equity(
    symbol: str,
    result: dict,
    times: np.ndarray | None,
    cost_rate: float,
    meta: dict,
    output_dir: Path,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    pnl = result["pnl"]
    cum = result["cum"]
    T = len(pnl)
    x = np.arange(T)

    fig = plt.figure(figsize=(16, 10), dpi=110)
    gs = gridspec.GridSpec(3, 2, height_ratios=[2.8, 1.0, 1.2], hspace=0.22, wspace=0.18)

    ax_eq = fig.add_subplot(gs[0, :])
    ax_dd = fig.add_subplot(gs[1, :], sharex=ax_eq)
    ax_yr = fig.add_subplot(gs[2, 0])
    ax_roll = fig.add_subplot(gs[2, 1])

    ann, sharpe, mdd = result["ann_ret"], result["sharpe"], result["mdd"]
    ax_eq.plot(x, cum, linewidth=2.0, color="#1565c0",
               label=f"{symbol}  Ann={ann*100:+.2f}%  Sharpe={sharpe:+.2f}  MDD={mdd*100:.2f}%")
    ax_eq.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_eq.set_ylabel("Cumulative Log Return", fontsize=10)
    ax_eq.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax_eq.grid(alpha=0.25)

    status_note = "  [DISABLED LIVE]" if meta.get("status") == "disabled_live" else ""
    ax_eq.set_title(
        f"{symbol} Solo Factor Backtest{status_note}\n"
        f"Formula: {result['readable']}  |  cost={cost_rate}  |  "
        f"T={T} bars ({result['years_span']:.2f}y)  |  score={meta.get('best_score', 0):.4f}",
        fontsize=10, pad=8,
    )

    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    ax_dd.fill_between(x, dd, 0, alpha=0.4, color="#1565c0")
    ax_dd.axhline(0, color="gray", linewidth=0.5)
    ax_dd.set_ylabel("Drawdown", fontsize=9)
    ax_dd.grid(alpha=0.2)

    yearly = result.get("yearly_ret") or []
    if yearly:
        yr_colors = ["#2e7d32" if v >= 0 else "#c62828" for v in yearly]
        ax_yr.bar(range(1, len(yearly) + 1), [v * 100 for v in yearly], color=yr_colors, alpha=0.85)
        ax_yr.axhline(0, color="gray", linewidth=0.5)
        ax_yr.set_xlabel("Year", fontsize=9)
        ax_yr.set_ylabel("Return (%)", fontsize=9)
        ax_yr.set_title("Yearly Returns", fontsize=9)
        ax_yr.grid(alpha=0.2)

    window = 250
    if T > window:
        roll = []
        for i in range(window, T):
            seg = pnl[i - window:i]
            m, s = seg.mean(), seg.std()
            roll.append(m / (s + 1e-10) * math.sqrt(_H1_PER_YEAR))
        ax_roll.plot(range(window, T), roll, linewidth=0.9, color="#2e7d32", alpha=0.75)
        ax_roll.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax_roll.axhline(1.0, color="orange", linewidth=0.5, linestyle=":", alpha=0.5)
        ax_roll.set_ylabel("Rolling Sharpe", fontsize=8)
        ax_roll.set_title(f"Rolling Sharpe ({window} bars)", fontsize=9)
        ax_roll.grid(alpha=0.2)

    dt = _date_labels(times, T)
    if dt:
        ticks, labels = dt
        ax_dd.set_xticks(ticks)
        ax_dd.set_xticklabels(labels, fontsize=7, rotation=20)
    plt.setp(ax_eq.get_xticklabels(), visible=False)

    out = output_dir / f"equity_{_safe_filename(symbol)}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_combined(results_map: dict, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    syms = list(results_map.keys())
    if not syms:
        return ""

    # 对齐到最短序列（各品种历史长度不同）
    T = min(len(results_map[s]["pnl"]) for s in syms)

    fig = plt.figure(figsize=(18, 10), dpi=110)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.12)
    ax_eq = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)

    colors = ["#1565c0", "#00897b", "#e65100", "#6a1b9a", "#558b2f", "#b71c1c", "#26418f", "#795548"]
    all_pnls = []
    x = np.arange(T)

    for i, sym in enumerate(syms):
        pnl = results_map[sym]["pnl"][-T:]
        cum = np.cumsum(pnl)
        all_pnls.append(pnl)
        ann = calc_annual_return(float(cum[-1]), T)
        ax_eq.plot(x, cum, linewidth=1.0, alpha=0.75, color=colors[i % len(colors)],
                   label=f"{sym} ({ann*100:+.1f}%)")

    port_pnl = np.stack(all_pnls, axis=0).mean(axis=0)
    port_cum = np.cumsum(port_pnl)
    x = np.arange(T)
    ax_eq.plot(x, port_cum, linewidth=2.4, color="black",
               label=f"Equal-Weight ({calc_annual_return(float(port_cum[-1]), T)*100:+.1f}%)")
    ax_eq.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_eq.set_ylabel("Cumulative Log Return", fontsize=10)
    ax_eq.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.85)
    ax_eq.grid(alpha=0.25)
    ax_eq.set_title(
        f"All Factor Strategies — Solo Equity Curves (aligned last {T} H1 bars)  |  "
        f"Portfolio Sharpe={calc_sharpe(port_pnl):+.2f}  MDD={calc_mdd(port_cum)*100:.2f}%",
        fontsize=11, pad=8,
    )

    peak = np.maximum.accumulate(port_cum)
    dd = port_cum - peak
    ax_dd.fill_between(x, dd, 0, alpha=0.45, color="#b71c1c")
    ax_dd.axhline(0, color="gray", linewidth=0.5)
    ax_dd.set_ylabel("Drawdown", fontsize=9)
    ax_dd.set_xlabel("Bar Index (H1)", fontsize=9)
    ax_dd.grid(alpha=0.2)

    out = output_dir / "all_factors_equity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def run_solo(symbol: str, formula: list[int], fetcher: MT5DataFetcher) -> tuple[dict | None, np.ndarray | None]:
    cost = _cost(symbol)
    orig = Config.SYMBOLS[:]
    Config.SYMBOLS = [symbol]
    try:
        mgr = MT5DataManager(fetcher)
        mgr.load()
        if symbol not in mgr.symbols:
            return None, None
        times_t = mgr.raw_dict.get("time")
        times = times_t[0].detach().cpu().numpy() if times_t is not None else None
        result = solo_backtest(formula, mgr.raw_dict, mgr.target_ret, symbol, cost)
        return result, times
    finally:
        Config.SYMBOLS = orig


def main():
    offline = "--mt5" not in sys.argv
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded_list = [x for p in sorted(Path("strategies").glob("best_*.json")) if (x := load_best(p))]

    print(f"\nFactor Equity Curves | vocab={VOCAB_VERSION} | "
          f"{'offline D:/K线数据' if offline else 'MT5+缓存'}\n")
    print(f"{'品种':<16} {'年化%':>8} {'Sharpe':>8} {'MDD%':>8}")
    print("-" * 50)

    summary_rows = []
    curve_data: dict[str, dict] = {}

    with MT5DataFetcher(offline=offline) as fetcher:
        if not offline:
            fetcher.connect()
        for meta in loaded_list:
            sym = meta["symbol"]
            result, times = run_solo(sym, meta["formula"], fetcher)
            if result is None:
                print(f"{sym:<16} SKIP (no data)")
                continue

            plot_path = plot_solo_equity(sym, result, times, _cost(sym), meta, OUTPUT_DIR)
            print(f"{sym:<16} {result['ann_ret']*100:>8.2f} {result['sharpe']:>8.3f} "
                  f"{result['mdd']*100:>8.2f}")

            curve_data[sym] = {
                "pnl": result["pnl"],
                "cum": result["cum"],
                "ann_ret": result["ann_ret"],
            }
            summary_rows.append({
                "symbol": sym,
                "strategy_file": meta["path"],
                "formula": meta["formula"],
                "formula_readable": meta["readable"],
                "best_score": meta.get("best_score"),
                "status": meta.get("status"),
                "live_trading": meta.get("live_trading"),
                "disabled_reason": meta.get("disabled_reason"),
                "cost_rate": _cost(sym),
                "T": result["T"],
                "years": round(result["years_span"], 2),
                "total_ret": round(result["total_ret"], 6),
                "ann_ret": round(result["ann_ret"], 6),
                "sharpe": round(result["sharpe"], 4),
                "sortino": round(result["sortino"], 4),
                "mdd": round(result["mdd"], 6),
                "calmar": round(result["calmar"], 4),
                "half1_ret": round(result["half1_ret"], 6),
                "half2_ret": round(result["half2_ret"], 6),
                "half1_sharpe": round(result["half1_sharpe"], 4),
                "half2_sharpe": round(result["half2_sharpe"], 4),
                "yearly_ret": [round(v, 6) for v in result["yearly_ret"]],
                "equity_chart": plot_path,
                "valid_ann_positive": result["ann_ret"] > 0,
            })

    if curve_data:
        combined = plot_combined(curve_data, OUTPUT_DIR)
        print(f"\n组合图 -> {combined}")

    summary = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "vocab_version": VOCAB_VERSION,
        "data_mode": "offline" if offline else "mt5",
        "symbols": summary_rows,
        "valid_count": sum(1 for r in summary_rows if r["valid_ann_positive"]),
        "total_count": len(summary_rows),
    }
    json_path = OUTPUT_DIR / "factor_equity_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"摘要 JSON -> {json_path}")
    print(f"单品种图 -> {OUTPUT_DIR}/equity_*.png")


if __name__ == "__main__":
    main()
