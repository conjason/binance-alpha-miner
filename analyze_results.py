import json

with open(r'D:\cl\MT5_AlphaGPT\training_history.json') as f:
    hist = json.load(f)

with open(r'D:\cl\MT5_AlphaGPT\best_mt5_strategy.json') as f:
    best_strategy = json.load(f)

n = len(hist['step'])
print(f"Total steps: {n}")

# Final step
i = n - 1
print(f"\n=== Final Step {hist['step'][i]} ===")
print(f"  avg_reward  = {hist['avg_reward'][i]:.4f}")
print(f"  best_score  = {hist['best_score'][i]:.4f}")
print(f"  val_score   = {hist['val_score'][i]:.4f}")
print(f"  entropy     = {hist['entropy'][i]:.4f}")
print(f"  ic_mean     = {hist['ic_mean'][i]:.4f}")
print(f"  ic_stability= {hist['ic_stability'][i]:.4f}")
print(f"  sortino     = {hist['sortino'][i]:.4f}")

# Best val_score
best_i = max(range(n), key=lambda j: hist['val_score'][j])
print(f"\n=== Best ValScore @ Step {hist['step'][best_i]} ===")
print(f"  val_score   = {hist['val_score'][best_i]:.4f}")
print(f"  avg_reward  = {hist['avg_reward'][best_i]:.4f}")
print(f"  best_score  = {hist['best_score'][best_i]:.4f}")
print(f"  entropy     = {hist['entropy'][best_i]:.4f}")
print(f"  ic_mean     = {hist['ic_mean'][best_i]:.4f}")
print(f"  ic_stability= {hist['ic_stability'][best_i]:.4f}")
print(f"  sortino     = {hist['sortino'][best_i]:.4f}")

print(f"\n=== Best Strategy Tokens ===")
print(f"  {best_strategy}")

# Milestones
print(f"\n=== Milestones ===")
for s in [0, 25, 50, 75, 100, 125, 150, 175, 199]:
    if s < n:
        print(f"  Step {hist['step'][s]:3d}: avg_rw={hist['avg_reward'][s]:+.4f}  val={hist['val_score'][s]:+.4f}  ent={hist['entropy'][s]:.4f}  ic={hist['ic_mean'][s]:.4f}  sortino={hist['sortino'][s]:.4f}")

# Best val per phase
phases = [(0,50,"Early"), (50,100,"Mid-1"), (100,150,"Mid-2"), (150,200,"Late")]
print(f"\n=== Phase Analysis ===")
for start, end, name in phases:
    seg = range(start, min(end, n))
    if seg:
        bi = max(seg, key=lambda j: hist['val_score'][j])
        print(f"  {name:6s} (step {start}-{end}): BestVal={hist['val_score'][bi]:.4f} @ step {hist['step'][bi]}, AvgEnt={sum(hist['entropy'][s] for s in seg)/len(list(seg)):.4f}")
