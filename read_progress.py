"""read_progress.py — 读取当前训练进度并做回测分析"""
import sys, torch
sys.path.insert(0, '.')

ckpt = torch.load('checkpoints/ckpt_step_0480.pt', map_location='cpu', weights_only=True)
h       = ckpt['training_history']
tokens  = ckpt['best_formula']
best_sc = ckpt['best_score']

from model_core.vocab import FORMULA_VOCAB
names = FORMULA_VOCAB.token_names
n = len(h['step'])

vals  = h['val_score']
bests = h['best_score']
rews  = h['avg_reward']
ents  = h['entropy']
ics   = h['ic_mean']

print(f"=== 当前训练进度 step={ckpt['step']} ===")
print(f"全局最优 BestScore : {best_sc:.4f}")
print(f"末步批次 ValScore  : {vals[-1]:+.4f}")
print(f"末步 Entropy       : {ents[-1]:.3f}")
print(f"末步 IC            : {ics[-1]:+.5f}")
print(f"最优公式 tokens    : {tokens}")
print(f"最优公式 解读      : {' -> '.join(names[t] for t in tokens)}")
print()

print("=== BestScore 跳变点 ===")
prev = bests[0]
print(f"  step  0: {prev:.4f}")
for i in range(1, n):
    if bests[i] > prev + 0.05:
        print(f"  step{h['step'][i]:4d}: {prev:.4f} -> {bests[i]:.4f}  (+{bests[i]-prev:.4f})")
        prev = bests[i]

print()
print("=== 分阶段统计 ===")
for s, e in [(0,100),(100,200),(200,300),(300,400),(400,481)]:
    sv  = [vals[i] for i in range(n) if s <= h['step'][i] < e]
    sr  = [rews[i] for i in range(n) if s <= h['step'][i] < e]
    se  = [ents[i] for i in range(n) if s <= h['step'][i] < e]
    sic = [ics[i]  for i in range(n) if s <= h['step'][i] < e]
    if sv:
        print(f"  {s:3d}-{e}: val_mean={sum(sv)/len(sv):+.3f}"
              f"  val_max={max(sv):+.3f}"
              f"  H={sum(se)/len(se):.2f}"
              f"  IC={sum(sic)/len(sic):+.5f}")

for i, v in enumerate(vals):
    if v > 0:
        print(f"\n首次 val>0: step {h['step'][i]} (val={v:+.4f})")
        break

# ── 交易频率分析 ──────────────────────────────────────────────────
print()
print("=== 最优公式交易频率 ===")
from data_pipeline.fetcher import MT5DataFetcher
from data_pipeline.data_manager import MT5DataManager
from model_core.vm import StackVM

with MT5DataFetcher() as fetcher:
    mgr = MT5DataManager(fetcher)
    mgr.load()
    feat = mgr.feat_tensor
    syms = mgr.symbols

vm = StackVM()
factor = vm.execute(tokens, feat)
pos    = torch.sign(torch.tanh(factor))
T      = pos.shape[1]
print(f"数据长度 T={T} bars (H1 交集对齐)")
print()

total_trades = 0
for i, sym in enumerate(syms):
    p = pos[i]
    trades, runs, cur_len, cur_dir, prev = 0, [], 0, 0, 0
    for v in p.tolist():
        vi = int(v)
        if vi != 0:
            if vi == cur_dir:
                cur_len += 1
            else:
                if cur_len > 0: runs.append(cur_len)
                cur_dir, cur_len = vi, 1
                if prev != vi: trades += 1
        else:
            if cur_len > 0: runs.append(cur_len)
            cur_dir, cur_len = 0, 0
        prev = vi
    if cur_len > 0: runs.append(cur_len)
    avg_hold = sum(runs) / len(runs) if runs else 0
    zero_pct = (p == 0).float().mean().item() * 100
    long_pct = (p == 1).float().mean().item() * 100
    total_trades += trades
    print(f"  {sym}: {trades}笔  每100bar={trades/T*100:.1f}笔  "
          f"均持仓={avg_hold:.1f}bar  空仓={zero_pct:.0f}%  "
          f"多{long_pct:.0f}%/空{100-zero_pct-long_pct:.0f}%")

avg100 = total_trades / (T * len(syms)) * 100
print(f"\n  三品种合计: 每100bar={avg100:.1f}笔  每天~{avg100*24/100:.1f}笔")
