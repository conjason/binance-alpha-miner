import json, sys, glob
sys.path.insert(0,'.')
from model_core.vocab import FORMULA_VOCAB
names = FORMULA_VOCAB.token_names

for f in sorted(glob.glob('strategies/best_*.json')):
    d = json.load(open(f))
    sym = d.get('symbol', f)
    tok = d['formula']
    rd  = ' -> '.join(names[t] for t in tok)
    sc  = d.get('best_score', 'N/A')
    print(f"{sym}  score={sc:.3f}")
    print(f"  {rd}")
    print()
