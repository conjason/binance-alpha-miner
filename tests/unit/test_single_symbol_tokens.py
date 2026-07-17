"""单品种训练不得采样只对多品种截面有意义的 token。"""

import torch

from model_core.engine import AlphaEngine
from model_core.vocab import FORMULA_VOCAB


def test_single_symbol_engine_blocks_cross_sectional_tokens():
    engine = AlphaEngine(
        data_manager=None,
        target_symbol="BTCUSDT",
        use_lord_regularization=False,
    )
    blocked_names = set(engine.blocked_token_names)
    assert {
        "REL_RET5", "REL_RET20", "REL_VOL",
        "CS_RANK_RET5", "CS_ZSCORE_RET20",
        "CS_RANK", "CS_SCALE", "CS_NEUTRALIZE",
    } <= blocked_names

    mask = engine.sampler.valid_mask(
        stack_depth=1,
        step_idx=4,
        total_steps=8,
        device=torch.device("cpu"),
    )
    for name in blocked_names:
        tid = FORMULA_VOCAB.token_names.index(name)
        assert not bool(mask[tid]), name


def test_multi_symbol_engine_keeps_cross_sectional_tokens():
    engine = AlphaEngine(
        data_manager=None,
        target_symbol=None,
        use_lord_regularization=False,
    )
    assert engine.blocked_token_names == ()
    assert engine.sampler.blocked_token_ids == set()
