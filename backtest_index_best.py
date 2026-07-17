"""
backtest_index_best.py — 对 index 组最优因子做完整回测验证

用法：
    python backtest_index_best.py --offline

加载 strategies/best_index.json 中的因子，在 index 组 5 个品种全量历史数据上回测，
输出：回测摘要、品种级详情、前后半段一致性诊断、资金曲线图。
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
from model_core.vm import StackVM
from model_core.features import MT5FeatureEngineer
from strategy_manager.signal import compute_target_positions_stateless

_H1_PER_YEAR = 6240
OUTPUT_DIR = "backtest_output"
INDEX_SYMS = Config.SYMBOL_GROUPS["index"]  # ["US30.cash", "US100.cash", "US500.cash", "US2000.cash", "JP225.cash"]

# 指数品种点差成本更高
COST_RATE_INDEX = 0.0003


def decode(toks):
    names = FORMULA_VOCAB.token_names
    return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}" for t in toks)


def calc_sharpe(pnl):
    m, s = pnl.mean(), pnl.std()
    return float(m / (s + 1e-10) * math.sqrt(_H1_PER_YEAR))


def calc_sortino(pnl):
    m = pnl.mean()
    down = pnl[pnl < 0]
    ds = down.std() if len(down) > 0 else 1e-10
    ds = max(ds, abs(m), 1e-10)
    return float(np.clip(m / ds * math.sqrt(_H1_PER_YEAR), -20, 20))


def calc_mdd(cum_pnl):
    peak = np.maximum.accumulate(cum_pnl)
    return float((peak - cum_pnl).max())


def calc_ic(factor, target_ret):
    """时序 IC：factor[t] vs target_ret[t+1]，逐品种再取均值。"""
    N, T = factor.shape
    ic_list = []
    for n in range(N):
        x = factor[n, :-1]
        y = target_ret[n, 1:]
        xm = x - x.mean()
        ym = y - y.mean()
        sx = np.sqrt((xm**2).mean())
        sy = np.sqrt((ym**2).mean())
        if sx < 1e-6 or sy < 1e-6:
            continue
        ic = (xm * ym).mean() / (sx * sy + 1e-8)
        ic_list.append(float(ic))
    return float(np.mean(ic_list)) if ic_list else 0.0


def calc_annual_return(total_ret, T):
    """年化收益率：total_ret / (T / H1_PER_YEAR)"""
    years = T / _H1_PER_YEAR
    return float(total_ret / years) if years > 0 else 0.0


def calc_calmar(total_ret, mdd, T):
    """Calmar = 年化收益 / MDD"""
    ann = calc_annual_return(total_ret, T)
    return float(ann / mdd) if mdd > 1e-8 else 0.0


def backtest_one(formula, feat, raw_dict, target_ret, cost_rate=COST_RATE_INDEX):
    """对一个公式跑完整回测，返回统计结果。"""
    vm = StackVM()
    factor = vm.execute(formula, feat)  # [N, T]
    if factor is None:
        return None

    N, T = factor.shape
    factor_np = factor.detach().numpy()
    target_np = target_ret.detach().numpy()

    # 持仓
    pos = compute_target_positions_stateless(factor)  # [N, T]
    pos_np = pos.detach().numpy()

    # 换手
    prev = np.zeros_like(pos_np)
    prev[:, 1:] = pos_np[:, :-1]
    turnover = np.abs(pos_np - prev)

    pnl = pos_np * target_np - turnover * cost_rate  # [N, T]

    # 分品种统计
    per_sym = {}
    for i, sym in enumerate(INDEX_SYMS[:N]):
        p = pnl[i]
        cum = np.cumsum(p)
        n_trades = int((np.abs(np.diff(pos_np[i])) > 0.5).sum())
        per_sym[sym] = {
            "pnl":       p,
            "cum":       cum,
            "total_ret": float(cum[-1]),
            "ann_ret":   calc_annual_return(float(cum[-1]), T),
            "sharpe":    calc_sharpe(p),
            "sortino":   calc_sortino(p),
            "mdd":       calc_mdd(cum),
            "n_trades":  n_trades,
            "avg_hold":  float(T / max(1, n_trades)) if n_trades > 0 else float(T),
            "ic":        calc_ic(factor_np[i:i+1], target_np[i:i+1]),
        }

    # 组合（等权）
    port_pnl = pnl.mean(axis=0)
    port_cum = np.cumsum(port_pnl)

    # IC
    ic = calc_ic(factor_np, target_np)

    # 分段分析：前50% vs 后50%（检查过拟合）
    split = T // 2
    p1 = port_pnl[:split]
    p2 = port_pnl[split:]

    port_total_ret = float(port_cum[-1])
    port_mdd = calc_mdd(port_cum)
    port_ann_ret = calc_annual_return(port_total_ret, T)
    port_calmar = calc_calmar(port_total_ret, port_mdd, T)

    return {
        "formula":        formula,
        "readable":       decode(formula),
        "per_sym":        per_sym,
        "port_total_ret": port_total_ret,
        "port_ann_ret":   port_ann_ret,
        "port_sharpe":    calc_sharpe(port_pnl),
        "port_sortino":   calc_sortino(port_pnl),
        "port_mdd":       port_mdd,
        "port_calmar":    port_calmar,
        "ic":             ic,
        "n_pos_syms":     sum(1 for d in per_sym.values() if d["total_ret"] > 0),
        "n_syms":         N,
        # 稳定性：前后半段
        "half1_sharpe":   calc_sharpe(p1),
        "half2_sharpe":   calc_sharpe(p2),
        "half1_sortino":  calc_sortino(p1),
        "half2_sortino":  calc_sortino(p2),
        "half1_ret":      float(np.cumsum(p1)[-1]),
        "half2_ret":      float(np.cumsum(p2)[-1]),
        # 换手
        "avg_turnover":   float(turnover.mean()),
        "avg_hold_h":     float(np.mean([d["avg_hold"] for d in per_sym.values()])),
        # FTMO 相关
        "max_consec_loss": _max_consecutive_loss(port_pnl),
    }


def _max_consecutive_loss(pnl):
    """最大连续亏损 bar 数"""
    max_streak = 0
    current = 0
    for p in pnl:
        if p < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def plot_equity_curves(result, times_arr, output_dir):
    """绘制资金曲线 + 回撤 + 各品种详情"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 12), dpi=110)
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 2], hspace=0.15)

    ax_eq = fig.add_subplot(gs[0])
    ax_dd = fig.add_subplot(gs[1], sharex=ax_eq)
    ax_sym = fig.add_subplot(gs[2], sharex=ax_eq)

    T = len(result["port_pnl"]) if "port_pnl" in result else len(result["per_sym"][INDEX_SYMS[0]]["pnl"])
    x = np.arange(T)

    # 组合资金曲线
    port_cum = np.cumsum(result.get("port_pnl", np.zeros(T)))
    ax_eq.plot(x, port_cum, linewidth=2.0, color="#1565c0",
               label=f"Portfolio  (AnnRet={result['port_ann_ret']:+.4f}, Sortino={result['port_sortino']:+.2f}, MDD={result['port_mdd']:.3f})")
    ax_eq.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_eq.set_ylabel("Cumulative Log Return", fontsize=10)
    ax_eq.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax_eq.grid(alpha=0.25)
    ax_eq.set_title(f"Index 组最优因子回测  |  {' + '.join(INDEX_SYMS)} 等权组合\n"
                    f"公式: {result['readable']}", fontsize=11, pad=8)

    # 回撤
    peak = np.maximum.accumulate(port_cum)
    dd = port_cum - peak
    ax_dd.fill_between(x, dd, 0, alpha=0.4, color="#1565c0", label="Drawdown")
    ax_dd.axhline(0, color="gray", linewidth=0.5)
    ax_dd.set_ylabel("Drawdown", fontsize=9)
    ax_dd.grid(alpha=0.2)
    ax_dd.legend(loc="lower left", fontsize=8)

    # 各品种资金曲线
    colors = ["#e65100", "#00897b", "#6a1b9a", "#b71c1c", "#26418f"]
    for i, sym in enumerate(INDEX_SYMS):
        if sym in result["per_sym"]:
            cum = result["per_sym"][sym]["cum"]
            c = colors[i % len(colors)]
            ax_sym.plot(x, cum, linewidth=1.2, color=c, alpha=0.8,
                        label=f"{sym}  (Ret={result['per_sym'][sym]['total_ret']:+.3f})")
    ax_sym.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_sym.set_ylabel("Per-Symbol Cum Return", fontsize=9)
    ax_sym.set_xlabel("Bar Index (H1)", fontsize=9)
    ax_sym.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.85)
    ax_sym.grid(alpha=0.2)

    plt.tight_layout()
    plot_path = Path(output_dir) / "index_best_backtest.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    return str(plot_path)


def main():
    offline = "--offline" in sys.argv

    # ── 加载最优因子
    strategy_path = Path("strategies/best_index.json")
    if not strategy_path.exists():
        print(f"❌ 策略文件不存在: {strategy_path}")
        return
    data = json.load(open(strategy_path))
    if data.get("vocab_version", "unknown") != VOCAB_VERSION:
        print(f"❌ 词表版本不匹配: {data.get('vocab_version')} != {VOCAB_VERSION}")
        return

    formula = data["formula"]
    best_score = data.get("best_score", 0.0)
    print(f"\n{'='*70}")
    print(f"  Index 组最优因子回测验证")
    print(f"{'='*70}")
    print(f"  公式 tokens : {formula}")
    print(f"  可读形式    : {decode(formula)}")
    print(f"  训练评分    : {best_score:.4f}")
    print(f"  词表版本    : {VOCAB_VERSION}")
    print(f"  品种        : {INDEX_SYMS}")
    print(f"  成本率      : {COST_RATE_INDEX}")
    print(f"  offline     : {offline}")
    print(f"{'='*70}\n")

    # ── 加载数据
    print("加载数据...")
    with MT5DataFetcher(offline=offline) as fetcher:
        mgr = MT5DataManager(fetcher)
        mgr.load()
        raw_dict   = mgr.raw_dict
        syms       = mgr.symbols
        T          = raw_dict["open"].shape[1]
        print(f"  全部品种: {syms}")
        print(f"  T={T} bars (H1)")

        # 取 index 组品种
        idx_idx = [syms.index(s) for s in INDEX_SYMS if s in syms]
        if len(idx_idx) < len(INDEX_SYMS):
            missing = set(INDEX_SYMS) - set(syms)
            print(f"  [WARN] 缺失品种: {missing}")
        idx_raw  = {k: v[idx_idx] for k, v in raw_dict.items()}
        idx_feat = MT5FeatureEngineer.compute_features(idx_raw)
        target_ret = mgr.target_ret[idx_idx]
        idx_syms_loaded = [syms[i] for i in idx_idx]
        print(f"  Index 品种: {idx_syms_loaded}  feat={idx_feat.shape}\n")

        # ── 执行回测
        result = backtest_one(formula, idx_feat, idx_raw, target_ret)
        if result is None:
            print("❌ 因子执行失败！")
            return

        # 补充 port_pnl 用于画图
        port_pnl = np.mean([result["per_sym"][s]["pnl"] for s in idx_syms_loaded if s in result["per_sym"]], axis=0)
        result["port_pnl"] = port_pnl

    # ── 打印回测摘要
    print(f"\n{'='*70}")
    print(f"  回测结果摘要（{T} bars H1，index 组 {len(idx_syms_loaded)} 品种）")
    print(f"{'='*70}")
    print(f"  公式: {result['readable']}")
    print(f"  ─────────────────────────────────────────")
    print(f"  组合总收益:   {result['port_total_ret']:+.4f}")
    print(f"  年化收益:     {result['port_ann_ret']:+.4f}")
    print(f"  Sharpe:       {result['port_sharpe']:+.4f}")
    print(f"  Sortino:      {result['port_sortino']:+.4f}")
    print(f"  Max DD:       {result['port_mdd']:.4f}")
    print(f"  Calmar:       {result['port_calmar']:+.4f}")
    print(f"  IC:           {result['ic']:.5f}")
    print(f"  平均换手:     {result['avg_turnover']:.6f}")
    print(f"  平均持仓:     {result['avg_hold_h']:.1f}h")
    print(f"  最大连续亏损: {result['max_consec_loss']} bars")
    print(f"  盈利品种:     {result['n_pos_syms']}/{result['n_syms']}")

    # ── 前后半段一致性
    print(f"\n  ── 前后半段一致性 ─────────────────────")
    print(f"  前半段: Sharpe={result['half1_sharpe']:+.4f}  Sortino={result['half1_sortino']:+.4f}  Ret={result['half1_ret']:+.4f}")
    print(f"  后半段: Sharpe={result['half2_sharpe']:+.4f}  Sortino={result['half2_sortino']:+.4f}  Ret={result['half2_ret']:+.4f}")
    h1, h2 = result["half1_sharpe"], result["half2_sharpe"]
    if h1 > 0 and h2 > 0:
        print(f"  [OK] 前后半段均为正，一致性良好")
    elif h1 * h2 > 0:
        print(f"  [WARN] 前后半段同号但均为负，因子方向可能反了")
    else:
        print(f"  [FAIL] 前后半段符号相反，过拟合嫌疑！")

    # ── 品种级详情
    print(f"\n  ── 品种级详情 ─────────────────────────")
    for sym, d in result["per_sym"].items():
        sig = "[OK]" if d["total_ret"] > 0 else "[X]"
        print(f"  {sym:12s}: TotRet={d['total_ret']:+.4f}  AnnRet={d['ann_ret']:+.4f}  "
              f"Sharpe={d['sharpe']:+.4f}  Sortino={d['sortino']:+.4f}  "
              f"MDD={d['mdd']:.4f}  IC={d['ic']:.5f}  "
              f"Trades={d['n_trades']}  AvgHold={d['avg_hold']:.0f}h  {sig}")

    # ── FTMO 评估
    print(f"\n{'='*70}")
    print(f"  FTMO 适配性评估")
    print(f"{'='*70}")
    ftmo_issues = []
    if result["port_mdd"] > 0.10:
        ftmo_issues.append(f"[WARN] MDD={result['port_mdd']:.4f} > 10% FTMO Max Loss 上限")
    else:
        print(f"  [OK] MDD={result['port_mdd']:.4f} < 10%，符合 FTMO Max Loss 约束")
    if result["port_ann_ret"] < 0:
        ftmo_issues.append(f"[WARN] 年化收益为负，不满足 FTMO 盈利要求")
    else:
        print(f"  [OK] 年化收益={result['port_ann_ret']:+.4f}，方向正确")
    if result["max_consec_loss"] > 500:
        ftmo_issues.append(f"[WARN] 最大连续亏损 {result['max_consec_loss']} bars 过长，可能触发 FTMO 每日亏损限制")
    if result["avg_hold_h"] < 2:
        ftmo_issues.append(f"[WARN] 平均持仓 {result['avg_hold_h']:.1f}h 过短，交易成本侵蚀大")
    if result["n_pos_syms"] < result["n_syms"] // 2:
        ftmo_issues.append(f"[WARN] 仅 {result['n_pos_syms']}/{result['n_syms']} 品种盈利，分散度不足")

    if ftmo_issues:
        print(f"\n  问题:")
        for iss in ftmo_issues:
            print(f"    {iss}")
    else:
        print(f"\n  [OK] 未发现明显 FTMO 违规风险")

    # ── 画图
    print(f"\n  绘制资金曲线图...")
    plot_path = plot_equity_curves(result, None, OUTPUT_DIR)
    print(f"  图表已保存 → {plot_path}")

    # ── 保存报告
    report = {
        "formula":        formula,
        "readable":       result["readable"],
        "best_score":     best_score,
        "vocab_version":  VOCAB_VERSION,
        "symbols":        idx_syms_loaded,
        "T":              T,
        "cost_rate":      COST_RATE_INDEX,
        "port_total_ret": round(result["port_total_ret"], 6),
        "port_ann_ret":   round(result["port_ann_ret"], 6),
        "port_sharpe":    round(result["port_sharpe"], 4),
        "port_sortino":   round(result["port_sortino"], 4),
        "port_mdd":       round(result["port_mdd"], 6),
        "port_calmar":    round(result["port_calmar"], 4),
        "ic":             round(result["ic"], 6),
        "avg_turnover":   round(result["avg_turnover"], 8),
        "avg_hold_h":     round(result["avg_hold_h"], 1),
        "max_consec_loss": result["max_consec_loss"],
        "n_pos_syms":     result["n_pos_syms"],
        "n_syms":         result["n_syms"],
        "half1_sharpe":   round(result["half1_sharpe"], 4),
        "half2_sharpe":   round(result["half2_sharpe"], 4),
        "half1_ret":      round(result["half1_ret"], 6),
        "half2_ret":      round(result["half2_ret"], 6),
        "per_sym":        {
            sym: {
                "total_ret": round(d["total_ret"], 6),
                "ann_ret":   round(d["ann_ret"], 6),
                "sharpe":    round(d["sharpe"], 4),
                "sortino":   round(d["sortino"], 4),
                "mdd":       round(d["mdd"], 6),
                "ic":        round(d["ic"], 6),
                "n_trades":  d["n_trades"],
                "avg_hold":  round(d["avg_hold"], 1),
            }
            for sym, d in result["per_sym"].items()
        },
    }
    report_path = Path(OUTPUT_DIR) / "index_best_backtest.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  报告已保存 → {report_path}")

    print(f"\n{'='*70}")
    print(f"  回测完成")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
