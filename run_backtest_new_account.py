"""
run_backtest_new_account.py — 用新账号品种名跑旧因子回测

新账号品种映射：
  EURUSDm  → EURUSD
  USDJPYm  → USDJPY
  XAUUSDm  → XAUUSD
  USTECm   → US100.cash
  US500m   → US500.cash
"""
import json, sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

import MetaTrader5 as mt5
import pandas as pd
import torch

from model_core.vm import StackVM
from model_core.features import MT5FeatureEngineer
from model_core.vocab import FORMULA_VOCAB
from strategy_manager.signal import compute_target_positions_stateless
from backtest_viz import BacktestEngine, BacktestChart

_H1_PER_YEAR = 6240

# ── 品种映射（旧名 → 新名）────────────────────────────────────────────
SYMBOL_MAP = {
    "EURUSDm": "EURUSD",
    "USDJPYm": "USDJPY",
    "XAUUSDm": "XAUUSD",
    "USTECm":  "US100.cash",
    "US500m":  "US500.cash",
}

# ── 要测试的因子（旧名 → formula tokens）────────────────────────────
STRATEGIES = {
    "v1_all":    {"formula": [6, 44, 3, 39, 4, 29, 34, 27],
                  "name": "v1 通用（均线差排名）",
                  "symbols": ["EURUSDm","USDJPYm","XAUUSDm","USTECm","US500m"]},
    "us500_v2":  {"formula": [5, 4, 28, 16, 23, 47, 18, 27],
                  "name": "US500 专属（ATR斜率/量价）",
                  "symbols": ["US500m"]},
    "xauusd_v1": {"formula": [6, 44, 3, 39, 4, 29, 34, 27],
                  "name": "XAU 专属（v1）",
                  "symbols": ["XAUUSDm"]},
}

def sharpe(p): m=p.mean(); s=p.std(); return float(m/s*math.sqrt(_H1_PER_YEAR)) if s>1e-10 else 0
def sortino(p):
    m=p.mean(); d=p[p<0]
    ds=max(d.std() if len(d)>0 else 1e-10, abs(m), 1e-10)
    return float(np.clip(m/ds*math.sqrt(_H1_PER_YEAR), -20, 20))
def mdd(c): return float((np.maximum.accumulate(c)-c).max()) if len(c)>0 else 0

def fetch(new_sym, bars=12000):
    r = mt5.copy_rates_from_pos(new_sym, 16385, 0, bars)
    if r is None or len(r) == 0:
        return None
    return pd.DataFrame(r)[["time","open","high","low","close","tick_volume"]]

def df_to_raw(df):
    t = torch.tensor(df[["open","high","low","close","tick_volume"]].values.T, dtype=torch.float32)
    raw = {
        "open":   t[0:1],
        "high":   t[1:2],
        "low":    t[2:3],
        "close":  t[3:4],
        "volume": t[4:5],
        "time":   torch.tensor(df["time"].values[None], dtype=torch.int64),
    }
    return raw

names = FORMULA_VOCAB.token_names

def main():
    mt5.initialize()
    print("\n获取实时点差...")
    costs = {}
    for old, new in SYMBOL_MAP.items():
        tick = mt5.symbol_info_tick(new)
        if tick and tick.ask > 0:
            mid = (tick.ask + tick.bid) / 2
            costs[old] = (tick.ask - tick.bid) / mid / 2
        else:
            costs[old] = 0.0001
        print(f"  {old}({new}): cost={costs[old]:.6f}")

    # 拉数据
    print("\n拉取 K 线数据...")
    raw_data = {}
    for old, new in SYMBOL_MAP.items():
        df = fetch(new)
        if df is None:
            print(f"  {new}: 无数据，跳过")
            continue
        raw_data[old] = df_to_raw(df)
        print(f"  {new}: {len(df)} bars")

    mt5.shutdown()

    vm = StackVM()
    all_results = []

    print(f"\n{'='*65}")
    print("  多因子回测（新账号数据，真实点差）")
    print(f"{'='*65}")
    header = f"{'品种':12s}{'策略':18s}{'PnL':>8}{'Sharpe':>8}{'Sortino':>8}{'MaxDD':>8}{'Trades':>7}{'WinRate':>8}"
    print(f"  {header}")
    print(f"  {'─'*75}")

    # 按策略组织：同一策略的品种合并算组合
    for strat_id, strat in STRATEGIES.items():
        formula   = strat["formula"]
        sym_list  = strat["symbols"]
        rd        = ' -> '.join(names[t] for t in formula)
        pnl_list  = []

        for old_sym in sym_list:
            if old_sym not in raw_data:
                continue
            new_sym = SYMBOL_MAP[old_sym]
            cost    = costs.get(old_sym, 0.0001)
            raw     = raw_data[old_sym]

            feat = MT5FeatureEngineer.compute_features(raw)  # [1,F,T]
            engine = BacktestEngine(formula=formula, cost_rate=cost)
            res    = engine.run(raw, feat, [old_sym])
            r      = res[0]

            pnl_list.append(r.pnl)
            all_results.append(r)

            # 单品种指标
            cum = r.cum_pnl
            print(f"  {old_sym:12s}{strat['name'][:17]:18s}"
                  f"{r.total_return:+8.3f}"
                  f"{sharpe(r.pnl):+8.3f}"
                  f"{sortino(r.pnl):+8.3f}"
                  f"{mdd(cum):8.3f}"
                  f"{r.n_trades:7d}"
                  f"{r.win_rate:8.1%}")

    # 等权组合（v1全品种）
    v1_syms   = [s for s in STRATEGIES["v1_all"]["symbols"] if s in raw_data]
    if v1_syms:
        pnls = []
        for s in v1_syms:
            raw   = raw_data[s]
            feat  = MT5FeatureEngineer.compute_features(raw)
            res   = BacktestEngine(formula=STRATEGIES["v1_all"]["formula"],
                                   cost_rate=costs.get(s, 0.0001)).run(raw, feat, [s])
            pnls.append(res[0].pnl)
        port = np.stack(pnls).mean(0)
        cum_p = np.cumsum(port)
        print(f"  {'─'*75}")
        print(f"  {'Portfolio(v1)':12s}{'均等权重':18s}"
              f"{cum_p[-1]:+8.3f}"
              f"{sharpe(port):+8.3f}"
              f"{sortino(port):+8.3f}"
              f"{mdd(cum_p):8.3f}")
        n_pos = sum(1 for s in v1_syms
                    for r in [BacktestEngine(STRATEGIES['v1_all']['formula'],
                                             costs.get(s,0.0001)).run(
                                             raw_data[s],
                                             MT5FeatureEngineer.compute_features(raw_data[s]),[s])[0]]
                    if r.total_return > 0)
        print(f"  正收益品种: {n_pos}/{len(v1_syms)}")

    print(f"\n{'='*65}")

    # 生成图表
    print("\n生成图表...")
    OUTPUT = "backtest_output_new_account"
    Path(OUTPUT).mkdir(exist_ok=True)
    chart = BacktestChart(max_bars=120)
    chart.plot_all(all_results, output_dir=OUTPUT)
    for r in all_results:
        chart.plot_all_trade_zooms(r, output_dir=OUTPUT, max_trades=8)
        print(f"  {r.symbol}: done")

    print(f"\n图表已保存 → {OUTPUT}/\n")


if __name__ == "__main__":
    main()
