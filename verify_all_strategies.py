"""
verify_all_strategies.py — 极度严谨的独立回测验证

不复用训练框架的 reward 函数，从第一性原理出发：
1. 加载数据 → 计算 target_ret = log(open[t+2]/open[t+1])
2. 执行公式 → 得到 factor[t]
3. 计算仓位 = tanh(factor[t])，应用 MIN_TRADE_EXPOSURE
4. PnL = pos[t] * target_ret[t] - |pos[t]-pos[t-1]| * cost
5. 严格统计：年化、Sharpe、Sortino、MDD、胜率、多空比、前后一致性
6. Walk-forward 4 折（不复用 engine 的实现）
7. 多 cost 场景压力测试
8. Beta 中性检验

验证标准：
- 有效：年化 > 2%, Sharpe > 0.5, MDD < 10% (FTMO), 前后一致, 多空均衡
- 可疑：任一维度不达标
- 无效：年化 < 0 或 Sharpe < 0 或 MDD > 20% 或 严重单边
"""
import sys
import json
import math
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.fetcher import MT5DataFetcher
from data_pipeline.data_manager import MT5DataManager
from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB
from strategy_manager.signal import compute_target_positions_stateless

# ── 常量 ──────────────────────────────────────────────────────────────
PERIODS_PER_YEAR = 6240  # H1
COST_RATE = 0.0001       # 单边成本
MIN_EXPOSURE = 0.05

GROUPS = {
    "forex":      ["EURUSD", "USDJPY"],
    "metals_comm": ["XAUUSD", "AAVUSD", "COCOA.c"],
    "index":       ["US30.cash", "US100.cash", "US500.cash"],
}

# ── 策略文件 ──────────────────────────────────────────────────────────
STRATEGIES = {
    "forex_v1":       {"file": "strategies/archive/best_forex_20250705_pre_refactor.json", "group": "forex"},
    "forex_v2":       {"file": "strategies/best_forex.json",                               "group": "forex"},
    "index_v1":       {"file": "strategies/archive/best_index_20250705_pre_refactor.json", "group": "index"},
    "index_v2":       {"file": "strategies/best_index.json",                               "group": "index"},
    "metals_comm_v1": {"file": "strategies/archive/best_metals_comm_20250705_pre_refactor.json", "group": "metals_comm"},
    "metals_comm_v2": {"file": "strategies/best_metals_comm.json",                         "group": "metals_comm"},
}


def load_strategy(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compute_factor(formula: list[int], feat_tensor: torch.Tensor, vm: StackVM) -> torch.Tensor:
    """执行公式得到因子值 [N, T]"""
    return vm.run(formula, feat_tensor)


def independent_backtest(factor: torch.Tensor, target_ret: torch.Tensor,
                         cost_rate: float = COST_RATE) -> dict:
    """完全独立的回测，不复用训练框架的任何评分函数。"""
    N, T = factor.shape

    # 仓位 = tanh(factor)，应用最小暴露门槛
    pos = torch.tanh(factor)
    pos = torch.where(pos.abs() >= MIN_EXPOSURE, pos, torch.zeros_like(pos))

    # 换手
    prev_pos = torch.zeros_like(pos)
    prev_pos[:, 1:] = pos[:, :-1]
    turnover = (pos - prev_pos).abs()

    # PnL
    pnl = pos * target_ret - turnover * cost_rate  # [N, T]

    # 组合（等权）
    port_pnl = pnl.mean(dim=0)  # [T]

    # ── 基础统计 ──────────────────────────────────────────────────────
    total_return = port_pnl.sum().item()
    ann_return = port_pnl.mean().item() * PERIODS_PER_YEAR

    # Sharpe
    mean_pnl = port_pnl.mean().item()
    std_pnl = port_pnl.std(unbiased=False).item()
    sharpe = (mean_pnl / (std_pnl + 1e-8)) * math.sqrt(PERIODS_PER_YEAR) if std_pnl > 1e-8 else 0.0

    # Sortino
    downside = port_pnl[port_pnl < 0]
    if downside.numel() > 0:
        ds_std = downside.std(unbiased=False).item()
    else:
        ds_std = 0.0
    full_std = max(std_pnl, 1e-8)
    ds_floor = max(full_std * 0.2, 1e-8)
    ds_std_clamped = max(ds_std, ds_floor)
    sortino = (mean_pnl / (ds_std_clamped + 1e-8)) * math.sqrt(PERIODS_PER_YEAR) if ds_std_clamped > 1e-8 else 0.0

    # MDD
    cum = torch.cumsum(port_pnl, dim=0)
    peak = torch.cummax(cum, dim=0).values
    drawdown = (peak - cum).max().item()
    mdd_pct = drawdown  # 绝对值

    # Calmar
    calmar = ann_return / (mdd_pct + 1e-8) if mdd_pct > 1e-8 else 0.0

    # 胜率
    win_rate = (port_pnl > 0).float().mean().item()

    # 多空比
    long_ratio = (pos > 0.05).float().mean().item()
    short_ratio = (pos < -0.05).float().mean().item()
    flat_ratio = 1.0 - long_ratio - short_ratio
    max_side = max(long_ratio, short_ratio)

    # 换手率
    avg_turnover = turnover.mean().item()

    # 交易次数（仓位方向变化）
    trade_count = 0
    for n in range(N):
        pos_n = pos[n].tolist()
        prev_dir = 0
        for p in pos_n:
            cur_dir = 1 if p > 0.05 else (-1 if p < -0.05 else 0)
            if cur_dir != 0 and cur_dir != prev_dir and prev_dir != 0:
                trade_count += 1
            if cur_dir != 0:
                prev_dir = cur_dir

    # ── 前后一致性 ────────────────────────────────────────────────────
    half = T // 2
    h1_pnl = port_pnl[:half]
    h2_pnl = port_pnl[half:]

    h1_ann = h1_pnl.mean().item() * PERIODS_PER_YEAR
    h2_ann = h2_pnl.mean().item() * PERIODS_PER_YEAR

    h1_std = h1_pnl.std(unbiased=False).item()
    h2_std = h2_pnl.std(unbiased=False).item()
    h1_sharpe = (h1_pnl.mean().item() / (h1_std + 1e-8)) * math.sqrt(PERIODS_PER_YEAR) if h1_std > 1e-8 else 0.0
    h2_sharpe = (h2_pnl.mean().item() / (h2_std + 1e-8)) * math.sqrt(PERIODS_PER_YEAR) if h2_std > 1e-8 else 0.0

    # ── 品种级 ────────────────────────────────────────────────────────
    per_symbol = []
    for n in range(N):
        sym_pnl = pnl[n]
        sym_ann = sym_pnl.mean().item() * PERIODS_PER_YEAR
        sym_std = sym_pnl.std(unbiased=False).item()
        sym_sharpe = (sym_pnl.mean().item() / (sym_std + 1e-8)) * math.sqrt(PERIODS_PER_YEAR) if sym_std > 1e-8 else 0.0
        sym_mdd = 0.0
        sym_cum = torch.cumsum(sym_pnl, dim=0)
        sym_peak = torch.cummax(sym_cum, dim=0).values
        sym_mdd = (sym_peak - sym_cum).max().item()
        per_symbol.append({
            "ann_ret": sym_ann,
            "sharpe": sym_sharpe,
            "mdd": sym_mdd,
            "long_ratio": (pos[n] > 0.05).float().mean().item(),
            "short_ratio": (pos[n] < -0.05).float().mean().item(),
        })

    # ── Walk-Forward 4 折（独立实现）──────────────────────────────────
    n_folds = 4
    fold_size = T // n_folds
    wf_results = []
    for k in range(1, n_folds):
        train_end = fold_size * k
        val_start = train_end
        val_end = min(train_end + fold_size, T)
        if val_start >= T or val_end <= val_start:
            continue
        val_pnl = port_pnl[val_start:val_end]
        val_ann = val_pnl.mean().item() * PERIODS_PER_YEAR
        val_std = val_pnl.std(unbiased=False).item()
        val_sharpe = (val_pnl.mean().item() / (val_std + 1e-8)) * math.sqrt(PERIODS_PER_YEAR) if val_std > 1e-8 else 0.0
        val_cum = torch.cumsum(val_pnl, dim=0)
        val_peak = torch.cummax(val_cum, dim=0).values
        val_mdd = (val_peak - val_cum).max().item()
        wf_results.append({
            "fold": k,
            "val_start": val_start,
            "val_end": val_end,
            "val_bars": val_end - val_start,
            "ann_ret": val_ann,
            "sharpe": val_sharpe,
            "mdd": val_mdd,
        })

    # ── 成本压力测试 ──────────────────────────────────────────────────
    cost_stress = {}
    for mult in [1.0, 2.0, 3.0, 5.0]:
        stressed_pnl = pos * target_ret - turnover * cost_rate * mult
        stressed_port = stressed_pnl.mean(dim=0)
        stressed_ann = stressed_port.mean().item() * PERIODS_PER_YEAR
        stressed_std = stressed_port.std(unbiased=False).item()
        stressed_sharpe = (stressed_port.mean().item() / (stressed_std + 1e-8)) * math.sqrt(PERIODS_PER_YEAR) if stressed_std > 1e-8 else 0.0
        cost_stress[f"{mult}x"] = {
            "ann_ret": stressed_ann,
            "sharpe": stressed_sharpe,
            "profitable": stressed_ann > 0,
        }

    # ── 判定 ──────────────────────────────────────────────────────────
    verdict = "VALID"
    issues = []

    if ann_return < 0.02:
        verdict = "INVALID"
        issues.append(f"年化收益 {ann_return*100:.2f}% < 2%")
    elif ann_return < 0:
        verdict = "INVALID"
        issues.append(f"年化收益为负 {ann_return*100:.2f}%")

    if sharpe < 0.5:
        if verdict == "VALID":
            verdict = "SUSPICIOUS"
        issues.append(f"Sharpe {sharpe:.3f} < 0.5")

    if mdd_pct > 0.10:
        if verdict == "VALID":
            verdict = "SUSPICIOUS"
        issues.append(f"MDD {mdd_pct*100:.2f}% > 10% (FTMO limit)")
        if mdd_pct > 0.20:
            verdict = "INVALID"
            issues.append(f"MDD {mdd_pct*100:.2f}% > 20% (critical)")

    if max_side > 0.85:
        if verdict == "VALID":
            verdict = "SUSPICIOUS"
        issues.append(f"单边占比 {max_side*100:.1f}% > 85% (beta factor)")

    # 前后一致性
    if h1_ann * h2_ann < 0:
        if verdict == "VALID":
            verdict = "SUSPICIOUS"
        issues.append(f"前后半段年化收益符号相反: H1={h1_ann*100:.2f}%, H2={h2_ann*100:.2f}%")

    # Walk-forward 一致性
    wf_positive = sum(1 for w in wf_results if w["ann_ret"] > 0)
    if wf_results and wf_positive < len(wf_results) * 0.5:
        if verdict == "VALID":
            verdict = "SUSPICIOUS"
        issues.append(f"Walk-forward {wf_positive}/{len(wf_results)} 折为正")

    # 成本压力
    if not cost_stress.get("2.0x", {}).get("profitable", False):
        if verdict == "VALID":
            verdict = "SUSPICIOUS"
        issues.append("2x 成本下亏损")

    return {
        "verdict": verdict,
        "issues": issues,
        "stats": {
            "ann_return": ann_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "mdd": mdd_pct,
            "win_rate": win_rate,
            "total_return": total_return,
            "avg_turnover": avg_turnover,
            "trade_count": trade_count,
        },
        "long_short": {
            "long_ratio": long_ratio,
            "short_ratio": short_ratio,
            "flat_ratio": flat_ratio,
            "max_side": max_side,
            "beta_neutral": max_side <= 0.85,
        },
        "consistency": {
            "h1_ann": h1_ann,
            "h2_ann": h2_ann,
            "h1_sharpe": h1_sharpe,
            "h2_sharpe": h2_sharpe,
            "same_sign": h1_ann * h2_ann > 0,
        },
        "per_symbol": per_symbol,
        "walk_forward": wf_results,
        "cost_stress": cost_stress,
    }


def main():
    vm = StackVM()
    results = {}

    with MT5DataFetcher(offline=True) as fetcher:
        for sname, sinfo in STRATEGIES.items():
            sfile = sinfo["file"]
            group = sinfo["group"]
            symbols = GROUPS.get(group, [])

            if not Path(sfile).exists():
                print(f"\n[{sname}] SKIP: {sfile} not found")
                continue

            strat = load_strategy(sfile)
            formula = strat.get("formula")
            saved_score = strat.get("best_score", "?")

            print(f"\n{'='*70}")
            print(f"  [{sname}] group={group} symbols={symbols}")
            print(f"  file={sfile}")
            print(f"  formula={formula}")
            print(f"  saved best_score={saved_score}")
            decoded = []
            for tid in formula:
                decoded.append(FORMULA_VOCAB.token_names[tid])
            print(f"  decoded: {' -> '.join(decoded)}")
            print(f"{'='*70}")

            # 加载数据（临时覆盖 Config.SYMBOLS）
            original_symbols = Config.SYMBOLS[:]
            Config.SYMBOLS = symbols
            try:
                mgr = MT5DataManager(fetcher)
                mgr.load()
            finally:
                Config.SYMBOLS = original_symbols
            feat = mgr.feat_tensor
            target_ret = mgr.target_ret
            N, T = target_ret.shape
            print(f"  Data: N={N} symbols, T={T} bars ({T/PERIODS_PER_YEAR:.2f} years)")

            # 执行公式
            try:
                factor = vm.execute(formula, feat)
            except Exception as e:
                print(f"  ERROR executing formula: {e}")
                results[sname] = {"error": str(e)}
                continue

            # 检查 factor 是否有 NaN/Inf
            if torch.isnan(factor).any() or torch.isinf(factor).any():
                print(f"  ERROR: factor contains NaN/Inf")
                nan_count = torch.isnan(factor).sum().item()
                inf_count = torch.isinf(factor).sum().item()
                print(f"    NaN={nan_count}, Inf={inf_count}")
                results[sname] = {"error": f"NaN={nan_count}, Inf={inf_count}"}
                continue

            print(f"  Factor stats: mean={factor.mean():.4f}, std={factor.std():.4f}, "
                  f"min={factor.min():.4f}, max={factor.max():.4f}")

            # 独立回测
            bt = independent_backtest(factor, target_ret)

            # 打印结果
            s = bt["stats"]
            ls = bt["long_short"]
            cs = bt["consistency"]

            print(f"\n  ── 核心统计 ──")
            print(f"  年化收益:   {s['ann_return']*100:>8.2f}%")
            print(f"  Sharpe:     {s['sharpe']:>8.3f}")
            print(f"  Sortino:    {s['sortino']:>8.3f}")
            print(f"  Calmar:     {s['calmar']:>8.3f}")
            print(f"  MDD:        {s['mdd']*100:>8.2f}%")
            print(f"  胜率:       {s['win_rate']*100:>8.1f}%")
            print(f"  换手率:     {s['avg_turnover']:>8.4f}")
            print(f"  交易次数:   {s['trade_count']:>8d}")

            print(f"\n  ── 多空均衡 ──")
            print(f"  多: {ls['long_ratio']*100:.1f}%  空: {ls['short_ratio']*100:.1f}%  "
                  f"平: {ls['flat_ratio']*100:.1f}%  最大单边: {ls['max_side']*100:.1f}%  "
                  f"Beta中性: {'是' if ls['beta_neutral'] else '否'}")

            print(f"\n  ── 前后一致性 ──")
            print(f"  H1 年化: {cs['h1_ann']*100:>8.2f}%  Sharpe: {cs['h1_sharpe']:.3f}")
            print(f"  H2 年化: {cs['h2_ann']*100:>8.2f}%  Sharpe: {cs['h2_sharpe']:.3f}")
            print(f"  同号: {'是' if cs['same_sign'] else '否'}")

            print(f"\n  ── Walk-Forward ({len(bt['walk_forward'])} 折) ──")
            for w in bt["walk_forward"]:
                print(f"  Fold {w['fold']}: bars={w['val_bars']}  "
                      f"年化={w['ann_ret']*100:>7.2f}%  Sharpe={w['sharpe']:.3f}  MDD={w['mdd']*100:.2f}%")
            wf_pos = sum(1 for w in bt["walk_forward"] if w["ann_ret"] > 0)
            print(f"  正收益折数: {wf_pos}/{len(bt['walk_forward'])}")

            print(f"\n  ── 成本压力测试 ──")
            for mult, cs_r in bt["cost_stress"].items():
                print(f"  {mult}: 年化={cs_r['ann_ret']*100:>7.2f}%  Sharpe={cs_r['sharpe']:.3f}  "
                      f"盈利={'是' if cs_r['profitable'] else '否'}")

            print(f"\n  ── 品种级 ──")
            for i, ps in enumerate(bt["per_symbol"]):
                sym_name = symbols[i] if i < len(symbols) else f"sym{i}"
                print(f"  {sym_name:12s}: 年化={ps['ann_ret']*100:>7.2f}%  "
                      f"Sharpe={ps['sharpe']:.3f}  MDD={ps['mdd']*100:.2f}%  "
                      f"L/S={ps['long_ratio']*100:.0f}/{ps['short_ratio']*100:.0f}")

            print(f"\n  ── 判定 ──")
            print(f"  结论: {bt['verdict']}")
            if bt["issues"]:
                for iss in bt["issues"]:
                    print(f"    ⚠ {iss}")
            else:
                print(f"    ✅ 全部检查通过")

            results[sname] = bt

    # ── 汇总表 ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print(f"  汇总")
    print(f"{'='*100}")
    print(f"{'策略':<20s} {'年化%':>8s} {'Sharpe':>8s} {'MDD%':>8s} {'多空比':>10s} {'H1/H2同号':>10s} {'WF正折':>8s} {'2x成本':>8s} {'判定':>10s}")
    print(f"{'-'*100}")
    for sname, r in results.items():
        if "error" in r:
            print(f"{sname:<20s} {'ERROR':>8s}")
            continue
        s = r["stats"]
        ls = r["long_short"]
        cs = r["consistency"]
        wf = r["walk_forward"]
        wf_pos = sum(1 for w in wf if w["ann_ret"] > 0)
        cs2x = r["cost_stress"]["2.0x"]["profitable"]
        ls_ratio = f"{ls['long_ratio']*100:.0f}/{ls['short_ratio']*100:.0f}"
        same = "是" if cs["same_sign"] else "否"
        profitable_2x = "是" if cs2x else "否"
        print(f"{sname:<20s} {s['ann_return']*100:>8.2f} {s['sharpe']:>8.3f} {s['mdd']*100:>8.2f} "
              f"{ls_ratio:>10s} {same:>10s} {wf_pos}/{len(wf):<5d} {profitable_2x:>8s} {r['verdict']:>10s}")

    # 保存完整结果
    output_path = "verification_results.json"
    serializable = {}
    for k, v in results.items():
        if "error" in v:
            serializable[k] = v
        else:
            serializable[k] = {
                "verdict": v["verdict"],
                "issues": v["issues"],
                "stats": v["stats"],
                "long_short": v["long_short"],
                "consistency": v["consistency"],
                "per_symbol": v["per_symbol"],
                "walk_forward": v["walk_forward"],
                "cost_stress": v["cost_stress"],
            }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
