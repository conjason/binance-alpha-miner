"""
scan_all_factors.py — 扫描 strategies/best_{symbol}.json，单品种 solo 回测（只看收益）

判定有效：年化收益 > 0（忽略 MDD / Sharpe / WF）
数据：D:\\K线数据 离线 H1
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.vocab import VOCAB_VERSION
from model_core.vm import StackVM
from strategy_manager.signal import compute_target_positions_stateless

PERIODS_PER_YEAR = 6240
COST = {
    "forex": 0.00015,
    "metals": 0.00020,
    "index": 0.00030,
}


def _cost(sym: str) -> float:
    if sym.startswith("XAU") or sym.startswith("XAG"):
        return COST["metals"]
    if ".cash" in sym or sym.startswith("US") or sym.startswith("JP"):
        return COST["index"]
    return COST["forex"]


def solo_backtest(formula: list[int], symbol: str) -> dict:
    with MT5DataFetcher(offline=True) as fetcher:
        orig = Config.SYMBOLS[:]
        Config.SYMBOLS = [symbol]
        try:
            mgr = MT5DataManager(fetcher)
            mgr.load()
            if symbol not in mgr.symbols:
                return {"error": "no data"}
            vm = StackVM()
            factor = vm.execute(formula, mgr.feat_tensor)
            if factor is None:
                return {"error": "vm failed"}
            pos = compute_target_positions_stateless(factor)
            prev = torch.zeros_like(pos)
            prev[:, 1:] = pos[:, :-1]
            turnover = (pos - prev).abs()
            cr = _cost(symbol)
            pnl = (pos * mgr.target_ret - turnover * cr).squeeze(0)
            T = int(pnl.shape[0])
            ann = float(pnl.mean().item() * PERIODS_PER_YEAR)
            total = float(pnl.sum().item())
            m = pnl.mean().item()
            s = pnl.std(unbiased=False).item()
            sharpe = float(m / (s + 1e-8) * math.sqrt(PERIODS_PER_YEAR))
            cum = torch.cumsum(pnl, dim=0)
            mdd = float((torch.cummax(cum, 0).values - cum).max().item())
            return {
                "T": T,
                "years": T / PERIODS_PER_YEAR,
                "ann_ret": ann,
                "total_ret": total,
                "sharpe": sharpe,
                "mdd": mdd,
                "valid": ann > 0,
            }
        finally:
            Config.SYMBOLS = orig


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
    if not sym or sym in ("index", "precious_metals", "forex", "metals_comm"):
        return None
    return {"symbol": sym, "formula": [int(t) for t in formula], "path": str(path)}


def main():
    strategies_dir = Path("strategies")
    files = sorted(strategies_dir.glob("best_*.json"))
    rows = []
    print(f"\nFactor Scan (returns-only) | vocab={VOCAB_VERSION} | offline\n")
    print(f"{'品种':<16} {'年化%':>8} {'Sharpe':>8} {'MDD%':>8} {'年数':>6} {'有效':>6}  文件")
    print("-" * 80)

    for path in files:
        loaded = load_best(path)
        if not loaded:
            continue
        sym = loaded["symbol"]
        bt = solo_backtest(loaded["formula"], sym)
        if "error" in bt:
            print(f"{sym:<16} ERROR: {bt['error']}")
            continue
        tag = "YES" if bt["valid"] else "NO"
        print(
            f"{sym:<16} {bt['ann_ret']*100:>8.2f} {bt['sharpe']:>8.3f} "
            f"{bt['mdd']*100:>8.2f} {bt['years']:>6.2f} {tag:>6}  {path.name}"
        )
        rows.append({"symbol": sym, "path": path.name, **bt})

    valid = [r for r in rows if r.get("valid")]
    print(f"\n有效因子（年化>0）: {len(valid)}/{len(rows)}")
    for r in sorted(valid, key=lambda x: -x["ann_ret"]):
        print(f"  {r['symbol']:<16} ann={r['ann_ret']*100:+.2f}%  {r['path']}")

    out = Path("backtest_output/factor_scan.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"all": rows, "valid": valid}, indent=2), encoding="utf-8")
    print(f"\nJSON → {out}")


if __name__ == "__main__":
    main()
