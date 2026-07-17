"""
train_precious_metals.py — 贵金属策略训练（XAUUSD + XAGUSD）
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.fetcher import MT5DataFetcher
from model_core.config import ModelConfig
from main import train_group

def main():
    offline = "--offline" in sys.argv
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  AlphaGPT 训练 — precious_metals 组 (贵金属)")
    print(f"{'='*60}")
    print(f"  品种: {Config.SYMBOL_GROUPS['precious_metals']}")
    print(f"  奖励模式: {ModelConfig.REWARD_MODE}")
    print(f"  训练步数: {ModelConfig.TRAIN_STEPS}")
    print(f"  offline={offline}")
    print(f"{'='*60}")

    with MT5DataFetcher(offline=offline) as fetcher:
        gsyms = Config.SYMBOL_GROUPS["precious_metals"]
        eng = train_group(fetcher, "precious_metals", gsyms, offline)
        if eng is not None:
            print(f"\n<<< [precious_metals] 完成: score={eng.best_score:.4f}")
            print(f"    {eng._decode_formula(eng.best_formula)}")
        else:
            print("\n<<< [precious_metals] 失败")

    elapsed = time.time() - t0
    print(f"\n耗时 {elapsed/3600:.2f}h")

if __name__ == "__main__":
    main()
