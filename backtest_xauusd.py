"""补跑 XAUUSD 单品种回测（仓库无 best_XAUUSD.json）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scan_all_factors import solo_backtest

CANDIDATES = [
    ("precious_metals_v1", "strategies/precious_metals_v1.json", "formula_tokens"),
    ("best_precious_metals", "strategies/best_precious_metals.json", "formula"),
    ("XAG_formula_on_XAU", "strategies/best_XAGUSD.json", "formula"),
]

print("\nXAUUSD solo backtest (returns-only)\n")
print(f"{'strategy':<28} {'ann%':>8} {'sharpe':>8} {'valid':>6}")
print("-" * 56)
results = []
for name, path, key in CANDIDATES:
    data = json.load(open(path, encoding="utf-8"))
    formula = data.get(key) or data.get("formula") or data.get("formula_tokens")
    bt = solo_backtest([int(t) for t in formula], "XAUUSD")
    tag = "YES" if bt.get("valid") else "NO"
    print(f"{name:<28} {bt['ann_ret']*100:>8.2f} {bt['sharpe']:>8.3f} {tag:>6}")
    results.append({"name": name, "symbol": "XAUUSD", **bt})

out = Path("backtest_output/xauusd_scan.json")
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\n-> {out}")
