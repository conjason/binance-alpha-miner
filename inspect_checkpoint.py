import torch, glob

ckpt_files = sorted(glob.glob('checkpoints/ckpt_index_step_*.pt'))
if not ckpt_files:
    print('No checkpoints')
else:
    latest = ckpt_files[-1]
    state = torch.load(latest, map_location='cpu')
    print(f'Checkpoint: {latest}')
    print(f"step: {state.get('step', '?')}")
    print(f"best_score: {state.get('best_score', float('nan')):.4f}")
    print(f"formula: {state.get('best_formula', [])}")
    print(f"restarts: {state.get('restart_count', '?')}")
    print(f"elite_pool_size: {len(state.get('elite_pool', []))}")
