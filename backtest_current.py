"""
backtest_current.py — 对已训练的 forex 组因子做完整回测分析

用法：
    python backtest_current.py --offline

加载 strategies/best_forex*.json 中的因子，在 EURUSD + USDJPY 全量历史数据上回测，
输出：对比表、品种级详情、过拟合诊断、资金曲线图。
"""
import sys, json, math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from model_core.features import MT5FeatureEngineer
from backtest_elite import backtest_one, decode, calc_sharpe, calc_sortino, calc_mdd, calc_ic

_H1_PER_YEAR = 6240
OUTPUT_DIR = "backtest_output"
FOREX_SYMS = ["EURUSD", "USDJPY"]


def load_formula(path: Path) -> tuple[list[int], str, float] | None:
    """从 JSON 加载因子，返回 (formula, readable, score)。"""
    if not path.exists():
        return None
    data = json.load(open(path))
    if data.get("vocab_version", "unknown") != VOCAB_VERSION:
        print(f"  [跳过] {path.name}: vocab 版本不符")
        return None
    return data["formula"], decode(data["formula"]), data.get("best_score", 0.0)


def plot_equity_curves(results: list[dict], times_arr: np.ndarray, output_dir: str):
    """绘制各因子的组合资金曲线对比 + 回撤。"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n = len(results)
    fig = plt.figure(figsize=(16, 9), dpi=110)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.12)
    ax_eq = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)

    colors = ["#1565c0", "#e65100", "#00897b", "#6a1b9a", "#b71c1c"]
    T = len(results[0]["port_pnl"])
    x = np.arange(T)

    for i, r in enumerate(results):
        cum = r["port_cum"]
        label = f"{r['label']}  (Sortino={r['port_sortino']:+.2f}, Ret={cum[-1]:+.3f})"
        ax_eq.plot(x, cum, linewidth=1.8, color=colors[i % len(colors)], label=label)
        # 回撤
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        ax_dd.fill_between(x, dd, 0, alpha=0.35, color=colors[i % len(colors)],
                           label=r["label"])

    ax_eq.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_eq.set_ylabel("Cumulative Log Return", fontsize=10)
    ax_eq.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax_eq.grid(alpha=0.25)
    ax_eq.set_title("Forex 组因子回测对比  |  EURUSD + USDJPY 等权组合", fontsize=12, pad=8)

    ax_dd.axhline(0, color="gray", linewidth=0.5)
    ax_dd.set_ylabel("Drawdown", fontsize=9)
    ax_dd.grid(alpha=0.2)

    # X 轴时间刻度
    if times_arr is not None and len(times_arr) == T:
        from datetime import datetime, timezone
        step = max(1, T // 10)
        ticks = x[::step]
        labels = [datetime.fromtimestamp(int(times_arr[i]), tz=timezone.utc).strftime("%y-%m-%d")
                  for i in range(0, T, step)]
        ax_dd.set_xticks(ticks)
        ax_dd.set_xticklabels(labels[:len(ticks)], fontsize=8, rotation=20)
    plt.setp(ax_eq.get_xticklabels(), visible=False)

    path = str(Path(output_dir) / "forex_factors_equity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  资金曲线图已保存 → {path}")
    return path


def plot_per_symbol(results: list[dict], output_dir: str):
    """各因子分品种的资金曲线（子图）。"""
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), dpi=110, sharex=True)
    if n == 1:
        axes = [axes]
    colors = {"EURUSD": "#1565c0", "USDJPY": "#e65100"}

    for i, r in enumerate(results):
        ax = axes[i]
        for sym, d in r["per_sym"].items():
            ax.plot(d["cum"], linewidth=1.4, color=colors.get(sym, "gray"),
                    label=f"{sym} (Sortino={d['sortino']:+.2f})")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Cum PnL", fontsize=9)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
        ax.grid(alpha=0.25)
        ax.set_title(f"{r['label']}  score={r['score']:.3f}", fontsize=10, loc="left")
    axes[-1].set_xlabel("Bar Index", fontsize=9)

    path = str(Path(output_dir) / "forex_factors_per_symbol.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  分品种图已保存 → {path}")


def main():
    offline = "--offline" in sys.argv

    # ── 1. 加载因子 ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Forex 组已训练因子回测分析  |  offline={offline}")
    print(f"{'='*72}\n")

    candidates = []
    for name, label in [
        ("best_forex.json",         "当前训练(score=0.485)"),
        ("best_forex_rank2.json",   "rank2(历史)"),
        ("best_forex_rank3.json",   "rank3(历史)"),
    ]:
        path = Path("strategies") / name
        loaded = load_formula(path)
        if loaded is None:
            print(f"  [缺失] {name}")
            continue
        formula, readable, score = loaded
        candidates.append({"label": label, "file": name,
                           "formula": formula, "readable": readable, "score": score})
        print(f"  {label}: {readable}")

    if not candidates:
        print("[ERROR] 没有找到任何因子文件"); sys.exit(1)
    print()

    # ── 2. 加载数据（forex 组独立加载，全量历史）────────────────────
    print("加载数据（forex 组，全量历史）...")
    # 临时只加载 forex 组品种
    original_symbols = Config.SYMBOLS[:]
    Config.SYMBOLS = FOREX_SYMS
    try:
        with MT5DataFetcher(offline=offline) as fetcher:
            mgr = MT5DataManager(fetcher)
            mgr.load()
            raw_dict = mgr.raw_dict
            syms = mgr.symbols
            T = raw_dict["open"].shape[1]
            times_all = raw_dict.get("time", None)
            print(f"  品种: {syms}  T={T} bars")
            print(f"  时间范围: {int(times_all[0].min())} .. {int(times_all[0].max())}\n")

            feat = MT5FeatureEngineer.compute_features(raw_dict)  # [N, F, T]
            target_ret = mgr.target_ret

            # ── 3. 逐因子回测 ───────────────────────────────────────
            results = []
            for c in candidates:
                # backtest_one 内部硬编码了 sym_names = ["EURUSD", "USDJPY"][:N]
                res = backtest_one(c["formula"], feat, raw_dict, target_ret, cost_rate=0.0001)
                if res is None:
                    print(f"  [失败] {c['label']}: 公式执行错误")
                    continue
                # 补充组合资金序列
                port_pnl = np.mean([d["pnl"] for d in res["per_sym"].values()], axis=0)
                port_cum = np.cumsum(port_pnl)
                res["port_pnl"] = port_pnl
                res["port_cum"] = port_cum
                res["label"] = c["label"]
                res["score"] = c["score"]
                res["readable"] = c["readable"]
                results.append(res)
    finally:
        Config.SYMBOLS = original_symbols

    if not results:
        print("[ERROR] 所有因子回测失败"); sys.exit(1)

    # ── 4. 对比汇总表 ───────────────────────────────────────────────
    print(f"{'='*72}")
    print(f"  回测对比汇总（{T} bars H1，EURUSD+USDJPY 等权组合）")
    print(f"{'='*72}")
    hdr = f"  {'因子':22s} {'Score':>7} {'TotRet':>8} {'Sharpe':>7} {'Sortino':>8} {'MDD':>7} {'IC':>7} {'H1':>5} {'H2':>5} {'AvgHold':>8}"
    print(hdr)
    print(f"  {'─'*88}")
    for r in results:
        h1, h2 = r["half1_sharpe"], r["half2_sharpe"]
        consistency = "✓" if h1 > 0 and h2 > 0 else ("⚠" if h1 * h2 > 0 else "✗")
        print(f"  {r['label']:22s} "
              f"{r['score']:>7.3f} "
              f"{r['port_total_ret']:>+8.3f} "
              f"{r['port_sharpe']:>+7.3f} "
              f"{r['port_sortino']:>+8.3f} "
              f"{r['port_mdd']:>7.3f} "
              f"{r['ic']:>7.4f} "
              f"{h1:>+5.2f} {h2:>+5.2f} "
              f"{r['avg_hold_h']:>7.1f}h {consistency}")
    print()

    # ── 5. 品种级详情 ───────────────────────────────────────────────
    print(f"{'─'*72}")
    print(f"  品种级详情")
    print(f"{'─'*72}")
    for r in results:
        print(f"\n  [{r['label']}]  {r['readable']}")
        for sym, d in r["per_sym"].items():
            sig = "✓" if d["total_ret"] > 0 else "✗"
            print(f"    {sym:8s}: TotRet={d['total_ret']:+.3f}  "
                  f"Sharpe={d['sharpe']:+.3f}  Sortino={d['sortino']:+.3f}  "
                  f"MDD={d['mdd']:.3f}  Trades={d['n_trades']}  "
                  f"AvgHold={d['avg_hold']:.0f}h  {sig}")

    # ── 6. 过拟合诊断 ───────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  过拟合与稳健性诊断")
    print(f"{'='*72}")
    for r in results:
        issues = []
        h1, h2 = r["half1_sharpe"], r["half2_sharpe"]
        if h1 > 0 and h2 < 0:
            issues.append(f"前半 Sharpe={h1:.2f} > 0，后半={h2:.2f} < 0 → 过拟合嫌疑")
        if h1 < 0 and h2 > 0:
            issues.append(f"前半={h1:.2f} < 0，后半={h2:.2f} > 0 → 近期才生效，需警惕")
        if r["port_mdd"] > 0.5:
            issues.append(f"最大回撤过大: {r['port_mdd']:.3f}")
        if r["avg_hold_h"] < 2:
            issues.append(f"持仓太短({r['avg_hold_h']:.1f}h)，点差侵蚀严重")
        if r["avg_hold_h"] > 500:
            issues.append(f"持仓极长({r['avg_hold_h']:.0f}h)，交易数极少，统计不可靠")
        if r["n_pos_syms"] < r["n_syms"]:
            issues.append(f"仅 {r['n_pos_syms']}/{r['n_syms']} 品种盈利，跨品种一致性差")
        if abs(r["ic"]) < 0.005:
            issues.append(f"IC≈0 ({r['ic']:.4f})，预测力存疑")
        # 前后半段 Sortino 一致性
        s1, s2 = r["half1_sortino"], r["half2_sortino"]
        if s1 > 0 and s2 > 0 and abs(s1 - s2) > max(s1, s2) * 0.7:
            issues.append(f"前后半 Sortino 差异大: {s1:.2f} vs {s2:.2f}，稳定性一般")

        if issues:
            print(f"\n  [{r['label']}]")
            for iss in issues:
                print(f"    ⚠ {iss}")
        else:
            print(f"\n  [{r['label']}]  ✓ 无明显问题")

    # ── 7. 资金曲线图 ───────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  生成图表")
    print(f"{'='*72}")
    times_np = times_all[0].numpy() if times_all is not None else None
    plot_equity_curves(results, times_np, OUTPUT_DIR)
    plot_per_symbol(results, OUTPUT_DIR)

    # ── 8. JSON 报告 ────────────────────────────────────────────────
    report = {
        "symbols": FOREX_SYMS,
        "T_bars": T,
        "factors": [],
    }
    for r in results:
        report["factors"].append({
            "label":         r["label"],
            "formula":       r["formula"],
            "readable":      r["readable"],
            "train_score":   round(r["score"], 4),
            "port_total_ret": round(r["port_total_ret"], 4),
            "port_sharpe":    round(r["port_sharpe"], 4),
            "port_sortino":   round(r["port_sortino"], 4),
            "port_mdd":       round(r["port_mdd"], 4),
            "ic":             round(r["ic"], 5),
            "half1_sharpe":   round(r["half1_sharpe"], 4),
            "half2_sharpe":   round(r["half2_sharpe"], 4),
            "half1_sortino":  round(r["half1_sortino"], 4),
            "half2_sortino":  round(r["half2_sortino"], 4),
            "avg_hold_h":     round(r["avg_hold_h"], 2),
            "per_sym": {
                sym: {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in d.items() if k != "pnl" and k != "cum"}
                for sym, d in r["per_sym"].items()
            },
        })
    rp = f"{OUTPUT_DIR}/forex_factors_report.json"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON 报告已保存 → {rp}")
    print("完成。\n")


if __name__ == "__main__":
    main()
