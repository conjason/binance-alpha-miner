"""analyze_ckpt.py — read latest checkpoint and run backtest on best formula"""
import sys, torch, json
sys.path.insert(0, '.')

import glob, os
ckpts = sorted(glob.glob('checkpoints/ckpt_step_*.pt'))
latest = ckpts[-1]
print(f"Latest checkpoint: {latest}")

ckpt    = torch.load(latest, map_location='cpu', weights_only=True)
tokens  = ckpt['best_formula']
best_sc = ckpt['best_score']
h       = ckpt['training_history']
step    = ckpt['step']

from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
names = FORMULA_VOCAB.token_names
n     = len(h['step'])
vals  = h['val_score']
bests = h['best_score']
ents  = h['entropy']
ics   = h['ic_mean']
rews  = h['avg_reward']

formula_str = ' -> '.join(names[t] for t in tokens)
print(f"\n=== Training Progress ===")
print(f"  step={step}/300  VOCAB_VERSION={VOCAB_VERSION}")
print(f"  GlobalBest={best_sc:.4f}  LastVal={vals[-1]:+.4f}")
print(f"  LastEntropy={ents[-1]:.3f}  LastIC={ics[-1]:+.5f}")
print(f"  Formula: {formula_str}")
print(f"  Tokens:  {tokens}")
print()

# best_score jumps
print("=== BestScore Jumps ===")
prev = bests[0]
print(f"  step  0: {prev:.4f}")
for i in range(1, n):
    if bests[i] > prev + 0.05:
        print(f"  step{h['step'][i]:4d}: {prev:.4f} -> {bests[i]:.4f}  (+{bests[i]-prev:.4f})")
        prev = bests[i]

print()
print("=== Phase Stats (per 50 steps) ===")
for s, e in [(0,50),(50,100),(100,150),(150,n)]:
    sv  = [vals[i] for i in range(n) if s <= h['step'][i] < e]
    sr  = [rews[i] for i in range(n) if s <= h['step'][i] < e]
    se  = [ents[i] for i in range(n) if s <= h['step'][i] < e]
    sic = [ics[i]  for i in range(n) if s <= h['step'][i] < e]
    if sv:
        print(f"  {s:3d}-{e}: val={sum(sv)/len(sv):+.3f}  max={max(sv):+.3f}"
              f"  rew={sum(sr)/len(sr):+.3f}  H={sum(se)/len(se):.2f}  IC={sum(sic)/len(sic):+.5f}")

# Trade frequency analysis
print()
print("=== Trade Frequency (current best formula) ===")
from data_pipeline.fetcher import MT5DataFetcher
from data_pipeline.data_manager import MT5DataManager
from model_core.vm import StackVM

with MT5DataFetcher() as fetcher:
    mgr = MT5DataManager(fetcher)
    mgr.load()
    feat = mgr.feat_tensor
    syms = mgr.symbols

vm    = StackVM()
factor = vm.execute(tokens, feat)
signal = torch.tanh(factor)
pos    = torch.sign(signal)
T      = pos.shape[1]
print(f"  symbols={syms}  T={T} bars")

total = 0
for i, sym in enumerate(syms):
    p = pos[i]
    trades, runs, cl, cd, prev = 0, [], 0, 0, 0
    for v in p.tolist():
        vi = int(v)
        if vi != 0:
            if vi == cd: cl += 1
            else:
                if cl > 0: runs.append(cl)
                cd, cl = vi, 1
                if prev != vi: trades += 1
        else:
            if cl > 0: runs.append(cl)
            cd, cl = 0, 0
        prev = vi
    if cl > 0: runs.append(cl)
    avg_hold = sum(runs)/len(runs) if runs else 0
    zero_pct = (p==0).float().mean().item()*100
    total += trades
    print(f"  {sym}: {trades}笔  per100bar={trades/T*100:.1f}  avgHold={avg_hold:.1f}bar  zero={zero_pct:.0f}%")

avg = total/(T*len(syms))*100
print(f"\n  All: per100bar={avg:.1f}  ~{avg*24/100:.1f} trades/day")
