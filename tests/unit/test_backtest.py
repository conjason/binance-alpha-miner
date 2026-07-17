"""
tests/unit/test_backtest.py — MT5Backtest 单元测试

验证 COST_RATE 默认值、evaluate() 返回类型、
已知 PnL 序列的 Sortino 计算结果、换手率惩罚以及 80/20 分割点。

Requirements: 5.2, 5.3
"""
import math
import pytest
import torch

from model_core.backtest import MT5Backtest


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: COST_RATE 默认值
# ─────────────────────────────────────────────────────────────────────────────

def test_default_cost_rate():
    """MT5Backtest 默认 cost_rate 应为 0.0001（forex/metals 点差+佣金）。
    Requirements: 5.2
    """
    bt = MT5Backtest()
    assert bt.cost_rate == 0.0001


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: evaluate() 返回正确类型和形状
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluate_return_types():
    """evaluate() 应返回 (scalar Tensor, float)。
    Requirements: 5.2
    """
    bt = MT5Backtest()
    T = 100
    N = 2
    factors = torch.randn(N, T)
    target_ret = torch.randn(N, T) * 0.01

    score, oos = bt.evaluate(factors, {}, target_ret)

    assert isinstance(score, torch.Tensor), "score 应为 torch.Tensor"
    assert score.shape == torch.Size([]), "score 应为标量（零维张量）"
    assert isinstance(oos, float), "oos 应为 float"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: 简单已知 PnL 序列 — 全部做多 + 固定正收益
# ─────────────────────────────────────────────────────────────────────────────

def test_simple_known_pnl():
    """全部做多且每期收益固定为 0.01 时，score 应为有限标量，不为 NaN/Inf。

    逻辑推导：
    - factors >> 0  →  tanh(factors) → +1  →  position = +1
    - target_ret = 0.01（每期）
    - 仅第 1 期有换手（从 0 → +1），后续换手为 0
    - pnl[0] = 1 * 0.01 - 1 * cost_rate ≈ 0.01 - 0.0001
    - pnl[t>0] = 1 * 0.01 - 0 = 0.01
    - 所有 pnl > 0，Sortino 分子 > 0，无下行收益 → 分母用 eps 代替
    - 期望 score 为大正数，且为有限值

    Requirements: 5.2, 5.3
    """
    bt = MT5Backtest()
    N, T = 1, 100

    # 极大正因子 → sign(tanh(10)) = +1
    factors = torch.full((N, T), 10.0)
    target_ret = torch.full((N, T), 0.01)

    score, oos = bt.evaluate(factors, {}, target_ret)

    assert torch.isfinite(score), f"score 应为有限值，实际为 {score.item()}"
    assert score.item() > 0, f"全部正收益时 score 应 > 0，实际为 {score.item()}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: 高换手率惩罚
# ─────────────────────────────────────────────────────────────────────────────

def test_high_turnover_penalty():
    """交替方向（高换手）的评分应严格低于恒定方向（低换手）的评分。

    构造：
    - high_turnover: factors 在 +10/-10 之间交替 → 每期换手 = 2
    - low_turnover:  factors 全为 +10           → 换手仅在首期

    两者使用相同的正 target_ret，high_turnover 因惩罚 (score -= 1.0) 而得分更低。

    Requirements: 5.3
    """
    bt = MT5Backtest()
    N, T = 1, 100
    target_ret = torch.full((N, T), 0.005)

    # 交替因子 → position 在 +1/-1 间切换 → turnover.mean() >> 0.5
    alternating = torch.tensor(
        [10.0 if i % 2 == 0 else -10.0 for i in range(T)]
    ).unsqueeze(0)  # [1, T]

    # 恒定正因子 → position 恒为 +1 → turnover.mean() ≈ 0
    constant = torch.full((N, T), 10.0)

    score_high, _ = bt.evaluate(alternating, {}, target_ret)
    score_low, _ = bt.evaluate(constant, {}, target_ret)

    assert score_high.item() < score_low.item(), (
        f"高换手评分 ({score_high.item():.4f}) 应 < 低换手评分 ({score_low.item():.4f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: 80/20 分割点验证
# ─────────────────────────────────────────────────────────────────────────────

def test_split_point_100():
    """T=100 时，in-sample 分割点应为 80。
    Requirements: 5.4 (通过检查 OOS 收益来验证分割点)
    """
    # 前 80 期收益为 +0.1，后 20 期收益为 -0.1
    # 若分割正确，score 基于前 80 期（正收益），oos 基于后 20 期（负收益）
    bt = MT5Backtest()
    T = 100
    N = 1

    factors = torch.full((N, T), 10.0)  # position = +1 everywhere

    target_ret = torch.zeros(N, T)
    target_ret[:, :80] = 0.1   # in-sample 高收益
    target_ret[:, 80:] = -0.1  # out-of-sample 负收益

    score, oos = bt.evaluate(factors, {}, target_ret)

    # in-sample 全正 → score > 0
    assert score.item() > 0, f"IS score 应 > 0（前80期正收益），实际为 {score.item()}"
    # oos 全负 → oos < 0
    assert oos < 0, f"OOS 均值应 < 0（后20期负收益），实际为 {oos}"


def test_split_point_50():
    """T=50 时，in-sample 分割点应为 40。
    Requirements: 5.4
    """
    bt = MT5Backtest()
    T = 50
    N = 1

    factors = torch.full((N, T), 10.0)  # position = +1 everywhere

    target_ret = torch.zeros(N, T)
    target_ret[:, :40] = 0.1   # in-sample 高收益（前 40 期）
    target_ret[:, 40:] = -0.1  # out-of-sample 负收益（后 10 期）

    score, oos = bt.evaluate(factors, {}, target_ret)

    assert score.item() > 0, f"IS score 应 > 0（前40期正收益），实际为 {score.item()}"
    assert oos < 0, f"OOS 均值应 < 0（后10期负收益），实际为 {oos}"


def test_split_point_exact_count():
    """直接验证分割计数：in-sample = floor(T*0.8)，oos = T - split。
    Requirements: 5.4
    """
    for T in [10, 50, 100, 123, 200]:
        expected_split = math.floor(T * 0.8)
        expected_oos = T - expected_split

        bt = MT5Backtest()
        N = 1

        # 设计特殊 target_ret：前 split 期 = +1，后 oos 期 = -1
        factors = torch.full((N, T), 10.0)
        target_ret = torch.ones(N, T)
        target_ret[:, expected_split:] = -1.0

        score, oos = bt.evaluate(factors, {}, target_ret)

        # IS 全正 → score > 0；OOS 全负 → oos < 0
        assert score.item() > 0, (
            f"T={T}: IS score 应 > 0，期望分割={expected_split}，实际 {score.item()}"
        )
        assert oos < 0, (
            f"T={T}: OOS 均值应 < 0，期望 OOS={expected_oos} 期，实际 {oos}"
        )
