"""run_v1_backtest.py — 用 v1 公式对所有品种回测（单公式对比）"""
import sys, json, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from backtest_viz import BacktestEngine
from model_core.features import MT5FeatureEngineer
from model_core.vocab import FORMULA_VOCAB

# v1 公式（策略库 strategy_v1_20260702.json）
V1_FORMULA = [6, 44, 3, 39, 4, 29, 34, 27]
names = FORMULA_VOCAB.token_names
readable = ' -> '.join(names[t] for t in V1_FORMULA)

_H1_PER_YEAR = 6240

def sharpe(p): m=p.mean(); s=p.std(); return float(m/s*math.sqrt(_H1_PER_YEAR)) if s>1e-10 else 0
def sortino(p):
    m=p.mean(); d=p[p<0]
    ds=d.std() if len(d)>0 else 1e-10; ds=max(ds,abs(m),1e-10)
    return float(np.clip(m/ds*math.sqrt(_H1_PER_YEAR),-20,20))
def mdd(c): return float((np.maximum.accumulate(c)-c).max())

# 真实点差
import MetaTrader5 as mt5
mt5.initialize()
costs = {}
for sym in Config.SYMBOLS:
    tick = mt5.symbol_info_tick(sym)
    if tick and tick.ask > 0:
        mid = (tick.ask+tick.bid)/2
        costs[sym] = (tick.ask-tick.bid)/mid/2
    else:
        costs[sym] = 0.0001
mt5.shutdown()

print(f"\nV1 公式（单公式，所有品种共用）")
print(f"  {readable}\n")

with MT5DataFetcher() as f:
    mgr = MT5DataManager(f)
    mgr.load()
    raw  = mgr.raw_dict
    syms = mgr.symbols
    feat = MT5FeatureEngineer.compute_features(raw)
    T    = feat.shape[2]

print(f"{'品种':12s} {'PnL':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Trades':>7} {'AvgH':>6}")
print('─'*65)

all_pnls = []
for i, sym in enumerate(syms):
    cr = costs.get(sym, 0.0001)
    engine = BacktestEngine(formula=V1_FORMULA, cost_rate=cr)
    res = engine.run({k: v[i:i+1] for k,v in raw.items()}, feat[i:i+1], [sym])
    r   = res[0]
    pnl = r.pnl
    cum = r.cum_pnl
    all_pnls.append(pnl)
    print(f"{sym:12s} {r.total_return:+8.3f} {sharpe(pnl):+8.3f} {sortino(pnl):+8.3f}"
          f" {mdd(cum):8.3f} {r.n_trades:7d} {r.avg_hold_bars:6.1f}h")

# 组合
port = np.stack(all_pnls).mean(0)
cum_p = np.cumsum(port)
print('─'*65)
print(f"{'Portfolio':12s} {cum_p[-1]:+8.3f} {sharpe(port):+8.3f} {sortino(port):+8.3f}"
      f" {mdd(cum_p):8.3f}")
print(f"\n  正收益品种: {sum(1 for p in all_pnls if np.cumsum(p)[-1]>0)}/{len(syms)}")
