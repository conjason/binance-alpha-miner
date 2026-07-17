"""
rigorous_backtest_audit.py — 对仓库内所有可用策略做独立严谨回测

不复用训练 reward；从第一性原理计算 PnL。
支持：全样本 / 前后半段 / Walk-Forward 4折 / 成本压力 / 单品种拆解。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
from model_core.vm import StackVM
from strategy_manager.signal import compute_target_positions_stateless

PERIODS_PER_YEAR = 6240
MIN_EXPOSURE = getattr(Config, "MIN_TRADE_EXPOSURE", 0.05)

# 品种组成本（单边 log）
COST_BY_GROUP = {
    "forex":          0.00015,
    "precious_metals": 0.00020,
    "index":          0.00030,
}

# 待测策略：name -> (path, group, symbols_override or None)
CANDIDATES: dict[str, dict] = {
    "precious_metals_current": {
        "path": "strategies/best_precious_metals.json",
        "group": "precious_metals",
        "symbols": None,
    },
    "precious_metals_v1": {
        "path": "strategies/precious_metals_v1.json",
        "group": "precious_metals",
        "symbols": None,
        "formula_key": "formula_tokens",
    },
    "index_current": {
        "path": "strategies/best_index.json",
        "group": "index",
        "symbols": None,
    },
}


def decode_formula(tokens: list[int]) -> str:
    names = FORMULA_VOCAB.token_names
    return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}" for t in tokens)


def load_candidate(path: str, formula_key: str = "formula") -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    data = json.load(p.open(encoding="utf-8"))
    ver = data.get("vocab_version")
    if ver and ver != VOCAB_VERSION:
        return {"error": f"vocab mismatch {ver} != {VOCAB_VERSION}"}
    formula = data.get(formula_key) or data.get("formula")
    if not formula:
        return {"error": "no formula"}
    return {
        "formula": [int(t) for t in formula],
        "best_score": data.get("best_score") or data.get("train_best_score"),
        "status": data.get("status", ""),
        "readable": data.get("formula_readable") or decode_formula(formula),
    }


def independent_backtest(
    factor: torch.Tensor,
    target_ret: torch.Tensor,
    cost_rate: float,
) -> dict:
    pos = compute_target_positions_stateless(factor)
    prev = torch.zeros_like(pos)
    prev[:, 1:] = pos[:, :-1]
    turnover = (pos - prev).abs()
    pnl = pos * target_ret - turnover * cost_rate
    port = pnl.mean(dim=0)
    T = int(port.shape[0])

    def _ann(x: torch.Tensor) -> float:
        return float(x.mean().item() * PERIODS_PER_YEAR)

    def _sharpe(x: torch.Tensor) -> float:
        m, s = x.mean().item(), x.std(unbiased=False).item()
        return float(m / (s + 1e-8) * math.sqrt(PERIODS_PER_YEAR)) if s > 1e-8 else 0.0

    def _sortino(x: torch.Tensor) -> float:
        m = x.mean().item()
        down = x[x < 0]
        ds = down.std(unbiased=False).item() if down.numel() else 0.0
        floor = max(x.std(unbiased=False).item() * 0.2, 1e-8)
        ds = max(ds, floor)
        return float(np.clip(m / ds * math.sqrt(PERIODS_PER_YEAR), -20, 20))

    def _mdd(x: torch.Tensor) -> float:
        cum = torch.cumsum(x, dim=0)
        peak = torch.cummax(cum, dim=0).values
        return float((peak - cum).max().item())

    half = T // 2
    h1, h2 = port[:half], port[half:]

    # Walk-forward 4 folds (val only)
    n_folds = 4
    fold_size = max(1, T // n_folds)
    wf = []
    for k in range(1, n_folds):
        vs = fold_size * k
        ve = min(vs + fold_size, T)
        if vs >= T:
            break
        vp = port[vs:ve]
        wf.append({
            "fold": k,
            "bars": ve - vs,
            "ann_ret": _ann(vp),
            "sharpe": _sharpe(vp),
            "mdd": _mdd(vp),
            "positive": _ann(vp) > 0,
        })

    cost_stress = {}
    for mult in (1.0, 2.0, 3.0, 5.0):
        sp = (pos * target_ret - turnover * cost_rate * mult).mean(dim=0)
        cost_stress[f"{mult}x"] = {
            "ann_ret": _ann(sp),
            "sharpe": _sharpe(sp),
            "profitable": _ann(sp) > 0,
        }

    long_r = (pos > MIN_EXPOSURE).float().mean().item()
    short_r = (pos < -MIN_EXPOSURE).float().mean().item()
    max_side = max(long_r, short_r)

    per_sym = []
    N = pnl.shape[0]
    for n in range(N):
        sp = pnl[n]
        per_sym.append({
            "total_ret": float(sp.sum().item()),
            "ann_ret": _ann(sp),
            "sharpe": _sharpe(sp),
            "sortino": _sortino(sp),
            "mdd": _mdd(sp),
            "long_pct": float((pos[n] > MIN_EXPOSURE).float().mean().item()),
            "short_pct": float((pos[n] < -MIN_EXPOSURE).float().mean().item()),
        })

    ann = _ann(port)
    sharpe = _sharpe(port)
    mdd = _mdd(port)

    verdict = "VALID"
    issues: list[str] = []
    if ann < 0.02:
        verdict = "INVALID"
        issues.append(f"年化 {ann*100:.2f}% < 2%")
    if sharpe < 0.5:
        if verdict == "VALID":
            verdict = "SUSPICIOUS"
        issues.append(f"Sharpe {sharpe:.3f} < 0.5")
    if mdd > 0.10:
        if verdict == "VALID":
            verdict = "SUSPICIOUS"
        issues.append(f"MDD {mdd*100:.2f}% > 10%")
        if mdd > 0.20:
            verdict = "INVALID"
    if max_side > 0.85:
        if verdict == "VALID":
            verdict = "SUSPICIOUS"
        issues.append(f"单边占比 {max_side*100:.1f}% > 85%")
    if _ann(h1) * _ann(h2) < 0:
        if verdict == "VALID":
            verdict = "SUSPICIOUS"
        issues.append(f"前后半段年化异号 H1={_ann(h1)*100:.2f}% H2={_ann(h2)*100:.2f}%")
    wf_pos = sum(1 for w in wf if w["positive"])
    if wf and wf_pos < len(wf) * 0.5:
        if verdict == "VALID":
            verdict = "SUSPICIOUS"
        issues.append(f"WF 仅 {wf_pos}/{len(wf)} 折为正")
    if not cost_stress["2.0x"]["profitable"]:
        if verdict == "VALID":
            verdict = "SUSPICIOUS"
        issues.append("2x 成本下亏损")

    return {
        "verdict": verdict,
        "issues": issues,
        "T": T,
        "years": T / PERIODS_PER_YEAR,
        "ann_ret": ann,
        "total_ret": float(port.sum().item()),
        "sharpe": sharpe,
        "sortino": _sortino(port),
        "mdd": mdd,
        "calmar": ann / (mdd + 1e-8),
        "long_pct": long_r,
        "short_pct": short_r,
        "max_side": max_side,
        "h1_ann": _ann(h1),
        "h2_ann": _ann(h2),
        "h1_sharpe": _sharpe(h1),
        "h2_sharpe": _sharpe(h2),
        "walk_forward": wf,
        "cost_stress": cost_stress,
        "per_symbol": per_sym,
        "n_pos_syms": sum(1 for p in per_sym if p["total_ret"] > 0),
        "n_syms": N,
    }


def run_one(name: str, meta: dict, fetcher: MT5DataFetcher) -> dict | None:
    loaded = load_candidate(meta["path"], meta.get("formula_key", "formula"))
    if loaded is None:
        print(f"\n[{name}] SKIP: file missing {meta['path']}")
        return None
    if "error" in loaded:
        print(f"\n[{name}] SKIP: {loaded['error']}")
        return None

    group = meta["group"]
    symbols = meta.get("symbols") or Config.SYMBOL_GROUPS[group]
    cost = COST_BY_GROUP.get(group, Config.COST_RATE)

    print(f"\n{'='*72}")
    print(f"  [{name}]  group={group}")
    print(f"  symbols={symbols}")
    print(f"  formula: {loaded['readable']}")
    print(f"  train_score={loaded.get('best_score')}  status={loaded.get('status') or '-'}")
    print(f"  cost_rate={cost}")
    print(f"{'='*72}")

    orig = Config.SYMBOLS[:]
    Config.SYMBOLS = list(symbols)
    try:
        mgr = MT5DataManager(fetcher)
        mgr.load()
        syms = mgr.symbols
        T = mgr.raw_dict["open"].shape[1]
        times = mgr.raw_dict.get("time")
        t0 = int(times[0, 0]) if times is not None else 0
        t1 = int(times[0, -1]) if times is not None else 0
        from datetime import datetime, timezone
        if t0 and t1:
            d0 = datetime.fromtimestamp(t0, tz=timezone.utc).strftime("%Y-%m-%d")
            d1 = datetime.fromtimestamp(t1, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"  loaded: {syms}  T={T} ({T/PERIODS_PER_YEAR:.2f}y)  {d0} .. {d1}")
        else:
            print(f"  loaded: {syms}  T={T} ({T/PERIODS_PER_YEAR:.2f}y)")

        vm = StackVM()
        factor = vm.execute(loaded["formula"], mgr.feat_tensor)
        if factor is None:
            print("  ERROR: formula execution failed")
            return {"error": "vm failed"}

        bt = independent_backtest(factor, mgr.target_ret, cost)
        bt["name"] = name
        bt["group"] = group
        bt["symbols"] = syms
        bt["formula"] = loaded["formula"]
        bt["readable"] = loaded["readable"]
        bt["train_score"] = loaded.get("best_score")

        print(f"\n  组合: ann={bt['ann_ret']*100:+.2f}%  Sharpe={bt['sharpe']:+.3f}  "
              f"Sortino={bt['sortino']:+.3f}  MDD={bt['mdd']*100:.2f}%  "
              f"Calmar={bt['calmar']:+.2f}")
        print(f"  多空: L={bt['long_pct']*100:.1f}% S={bt['short_pct']*100:.1f}% "
              f"max_side={bt['max_side']*100:.1f}%")
        print(f"  一致性: H1={bt['h1_ann']*100:+.2f}% H2={bt['h2_ann']*100:+.2f}%  "
              f"H1_S={bt['h1_sharpe']:+.2f} H2_S={bt['h2_sharpe']:+.2f}")
        wf_pos = sum(1 for w in bt["walk_forward"] if w["positive"])
        print(f"  WF: {wf_pos}/{len(bt['walk_forward'])} 折正收益")
        for mult, cs in bt["cost_stress"].items():
            print(f"    cost {mult}: ann={cs['ann_ret']*100:+.2f}% sharpe={cs['sharpe']:+.3f} "
                  f"ok={'Y' if cs['profitable'] else 'N'}")
        print(f"\n  品种级:")
        for i, sym in enumerate(syms):
            ps = bt["per_symbol"][i]
            tag = "OK" if ps["total_ret"] > 0 else "X"
            print(f"    {sym:14s}: ann={ps['ann_ret']*100:+7.2f}%  Sharpe={ps['sharpe']:+6.3f}  "
                  f"MDD={ps['mdd']*100:5.2f}%  L/S={ps['long_pct']*100:.0f}/{ps['short_pct']*100:.0f}  [{tag}]")
        print(f"\n  判定: {bt['verdict']}")
        for iss in bt["issues"]:
            print(f"    ! {iss}")
        if not bt["issues"]:
            print("    全部检查通过")

        # 单品种拆解（同一公式，N=1）
        print(f"\n  --- 单品种独立回测（同公式，逐品种加载）---")
        solo_results = []
        for sym in symbols:
            Config.SYMBOLS = [sym]
            try:
                sm = MT5DataManager(fetcher)
                sm.load()
                if sym not in sm.symbols:
                    continue
                f1 = vm.execute(loaded["formula"], sm.feat_tensor)
                if f1 is None:
                    continue
                sb = independent_backtest(f1, sm.target_ret, cost)
                solo_results.append({"symbol": sym, **sb})
                print(f"    {sym:14s} solo: ann={sb['ann_ret']*100:+7.2f}%  "
                      f"Sharpe={sb['sharpe']:+.3f}  MDD={sb['mdd']*100:.2f}%  "
                      f"verdict={sb['verdict']}")
            except Exception as e:
                print(f"    {sym:14s} solo ERROR: {e}")
        bt["solo"] = solo_results
        return bt
    finally:
        Config.SYMBOLS = orig


def main():
    offline = "--offline" in sys.argv
    mode = "offline" if offline else "MT5/live"
    print(f"\nRigorous Backtest Audit  |  vocab={VOCAB_VERSION}  |  mode={mode}\n")

    results: dict[str, dict] = {}
    with MT5DataFetcher(offline=offline) as fetcher:
        if not offline:
            try:
                fetcher.connect()
            except Exception as e:
                print(f"MT5 connect failed: {e}\nRetry with --offline if cache exists.")
                sys.exit(1)
        for name, meta in CANDIDATES.items():
            r = run_one(name, meta, fetcher)
            if r:
                results[name] = r

    print(f"\n\n{'='*88}")
    print("  汇总（组合等权）")
    print(f"{'='*88}")
    print(f"{'策略':<26} {'年化%':>8} {'Sharpe':>8} {'MDD%':>8} {'WF+':>6} {'2x成本':>8} {'盈利品种':>10} {'判定':>12}")
    print("-" * 88)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<26} ERROR")
            continue
        wf_pos = sum(1 for w in r["walk_forward"] if w["positive"])
        wf_n = len(r["walk_forward"])
        ok2x = "Y" if r["cost_stress"]["2.0x"]["profitable"] else "N"
        print(f"{name:<26} {r['ann_ret']*100:>8.2f} {r['sharpe']:>8.3f} {r['mdd']*100:>8.2f} "
              f"{wf_pos}/{wf_n:<4} {ok2x:>8} {r['n_pos_syms']}/{r['n_syms']:<8} {r['verdict']:>12}")

    # 单品种最优
    print(f"\n{'='*88}")
    print("  单品种拆解（同公式 solo）")
    print(f"{'='*88}")
    solo_rows = []
    for name, r in results.items():
        for s in r.get("solo", []):
            solo_rows.append((s["ann_ret"], name, s["symbol"], s))
    solo_rows.sort(reverse=True, key=lambda x: x[0])
    print(f"{'品种':<14} {'来源策略':<26} {'年化%':>8} {'Sharpe':>8} {'MDD%':>8} {'判定':>12}")
    print("-" * 88)
    for _, name, sym, s in solo_rows:
        print(f"{sym:<14} {name:<26} {s['ann_ret']*100:>8.2f} {s['sharpe']:>8.3f} "
              f"{s['mdd']*100:>8.2f} {s['verdict']:>12}")

    out = Path("backtest_output/rigorous_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    serial = {}
    for k, v in results.items():
        if "error" in v:
            serial[k] = v
            continue
        serial[k] = {key: val for key, val in v.items() if key != "solo" or True}
        if "solo" in v:
            serial[k]["solo"] = [
                {kk: vv for kk, vv in s.items() if kk not in ("walk_forward", "cost_stress", "per_symbol", "issues")}
                for s in v["solo"]
            ]
    out.write_text(json.dumps(serial, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n详细 JSON → {out}")


if __name__ == "__main__":
    main()
