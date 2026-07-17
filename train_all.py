"""
train_all.py — 批量单品种训练

按 TRAINABLE_SYMBOLS 顺序逐一训练所有品种。
用法: python train_all.py --offline
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.fetcher import MT5DataFetcher
from train_single import train_single


def main():
    offline = "--offline" in sys.argv
    t0_total = time.time()
    symbols = Config.TRAINABLE_SYMBOLS
    results = {}

    print(f"\n{'='*60}")
    print(f"  批量单品种训练 — {len(symbols)} 个品种")
    print(f"  顺序: {symbols}")
    print(f"{'='*60}")

    with MT5DataFetcher(offline=offline) as fetcher:
        for symbol in symbols:
            t0 = time.time()
            eng = train_single(fetcher, symbol, offline)
            elapsed = time.time() - t0
            if eng:
                results[symbol] = {
                    "score": eng.best_score,
                    "formula": eng._decode_formula(eng.best_formula) if eng.best_formula else "N/A",
                    "time_h": elapsed / 3600,
                }
            else:
                results[symbol] = {"score": -1, "formula": "FAILED", "time_h": 0}

    # 汇总
    print(f"\n{'='*60}")
    print(f"  批量训练完成 总耗时 {(time.time()-t0_total)/3600:.2f}h")
    print(f"{'='*60}")
    print(f"{'Symbol':<14s} {'Score':>8s}  {'Formula'}")
    print('-' * 60)
    for sym, r in results.items():
        print(f"{sym:<14s} {r['score']:>8.4f}  {r['formula']}")


if __name__ == "__main__":
    main()
