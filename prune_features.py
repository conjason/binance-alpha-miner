"""
prune_features.py — 特征剪枝：从 65 个特征中筛出有区分度、低冗余的子集

背景：
    特征库扩展到 65 个后，vocab=131，8-token 搜索空间暴增到 ~8.67×10^16。
    其中大量特征高度相关（多周期均线/动量/通道位置类），是噪声维度。

做法（用 model_core.evaluator）：
    1. 离线加载全部品种数据，直接从 _FEATURE_DEFS 计算全部 65 个特征
    2. score_all：对每个特征算 IC / RankIC / 互信息，跨特征秩归一聚合 importance
    3. prune：保守双条件相关性剪枝（corr>阈值 且 分差>margin 才剪）
    4. 额外按 importance 取 Top-K，去掉近零 IC 的弱特征
    5. 写出 active_features.json（features.py 启动时读取，只注册白名单特征）

用法：
    python prune_features.py                 # 默认 corr_threshold=0.85, top_k=28
    python prune_features.py --top-k 25      # 自定义保留数量
    python prune_features.py --corr 0.9      # 自定义相关性阈值
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data_pipeline.data_manager import MT5DataManager
from data_pipeline.fetcher import MT5DataFetcher
from model_core.features import _FEATURE_DEFS
from model_core.evaluator import score_all, prune

OUTPUT = Path(__file__).parent / "active_features.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=28,
                    help="剪枝后额外按 importance 截断保留的最大特征数")
    ap.add_argument("--corr", type=float, default=0.85,
                    help="相关性剪枝阈值（越低剪得越狠）")
    ap.add_argument("--margin", type=float, default=0.01,
                    help="剪枝分差阈值（势均力敌时保留）")
    args = ap.parse_args()

    print(f"{'='*62}")
    print(f"  特征剪枝  corr_threshold={args.corr}  top_k={args.top_k}")
    print(f"{'='*62}")

    # ── 1. 离线加载数据 ───────────────────────────────────────────────
    print("加载数据（离线缓存）...")
    with MT5DataFetcher(offline=True) as fetcher:
        mgr = MT5DataManager(fetcher)
        mgr.load()
        raw_dict = mgr.raw_dict
        target   = mgr.target_ret          # [N, T]
        syms     = mgr.symbols
    N, T = target.shape
    print(f"  品种={syms}  N={N}  T={T}")

    # ── 2. 直接从 _FEATURE_DEFS 计算全部 65 个特征（绕过白名单）───────
    print(f"\n计算全部 {len(_FEATURE_DEFS)} 个特征...")
    candidates: dict[str, torch.Tensor] = {}
    categories: dict[str, str] = {}
    for name, category, compute in _FEATURE_DEFS:
        try:
            series = compute(raw_dict)                 # [N, T]
            candidates[name] = torch.nan_to_num(series, nan=0.0,
                                                posinf=0.0, neginf=0.0)
            categories[name] = category
        except Exception as e:
            print(f"  [跳过] {name}: 计算失败 {e}")

    # ── 3. 打分 ───────────────────────────────────────────────────────
    print(f"\n对 {len(candidates)} 个特征打分（IC / RankIC / MI）...")
    scores = score_all(candidates, target, categories=categories)
    scores_sorted = sorted(scores, key=lambda s: (
        -s.importance_score if s.importance_score == s.importance_score else 1,
    ))

    print(f"\n  {'特征':22s}{'类别':14s}{'IC':>8}{'RankIC':>8}{'MI':>8}{'importance':>12}")
    print(f"  {'-'*70}")
    for s in scores_sorted:
        imp = s.importance_score
        imp_str = f"{imp:.4f}" if imp == imp and imp > -1e30 else "  退化"
        print(f"  {s.candidate:22s}{s.category:14s}"
              f"{s.ic:+8.4f}{s.rank_ic:+8.4f}{s.mi:8.4f}{imp_str:>12}")

    # ── 4. 相关性剪枝 ─────────────────────────────────────────────────
    print(f"\n相关性剪枝（corr>{args.corr} 且分差>{args.margin}）...")
    rows = prune(scores, candidates, corr_threshold=args.corr, margin=args.margin)
    retained = [r for r in rows if r.retention_status == "retained"]
    pruned   = [r for r in rows if r.retention_status == "pruned"]
    print(f"  相关性剪枝后保留 {len(retained)} 个，剪掉 {len(pruned)} 个")
    for r in pruned:
        print(f"    ✗ {r.candidate:22s} (favor of {r.pruned_in_favor_of})")

    # ── 5. 按 importance 取 Top-K（去掉弱特征）────────────────────────
    retained_sorted = sorted(
        retained,
        key=lambda r: (r.importance_score if r.importance_score == r.importance_score
                       and r.importance_score > -1e30 else -1e30),
        reverse=True,
    )
    final = retained_sorted[:args.top_k]
    final_names_set = {r.candidate for r in final}

    # 保持 _FEATURE_DEFS 原始顺序输出
    ordered_names = [name for name, _, _ in _FEATURE_DEFS if name in final_names_set]

    print(f"\n{'='*62}")
    print(f"  最终保留 {len(ordered_names)} 个特征（vocab 从 65 特征降至 {len(ordered_names)}）")
    print(f"{'='*62}")
    for n in ordered_names:
        print(f"  ✓ {n}")

    # ── 6. 写出 active_features.json ─────────────────────────────────
    payload = {
        "active_features": ordered_names,
        "meta": {
            "source": "prune_features.py",
            "corr_threshold": args.corr,
            "top_k": args.top_k,
            "n_original": len(_FEATURE_DEFS),
            "n_retained": len(ordered_names),
            "symbols": syms,
            "bars": T,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"\n已写出 → {OUTPUT}")
    print("重新 import model_core.features 时将只注册白名单特征。\n")


if __name__ == "__main__":
    main()
