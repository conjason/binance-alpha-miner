"""
单元测试：factor_pool 因子去相关池

测试 _update_factor_pool 与 _apply_corr_penalty 的核心行为。
无需 data_manager，直接实例化 AlphaEngine(data_manager=None)。

需求：T3.1~T3.7
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from model_core.engine import AlphaEngine
from model_core.config import ModelConfig


# ── 公共 fixture ──────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    """无 data_manager 的 AlphaEngine 实例，供所有测试复用。"""
    return AlphaEngine(data_manager=None)


def _make_factor() -> torch.Tensor:
    """返回形状 [5, 50] 的随机因子张量。"""
    return torch.randn(5, 50)


# ── TestUpdateFactorPool ──────────────────────────────────────────────────

class TestUpdateFactorPool:
    """测试 _update_factor_pool 的池容量上限与 Top-K 正确性（T3.1, T3.2）。"""

    def test_pool_size_does_not_exceed_top_k(self, engine):
        """插入 FACTOR_TOP_K + 5 个因子后，池大小恰好等于 FACTOR_TOP_K（T3.1）。"""
        n_insert = ModelConfig.FACTOR_TOP_K + 5
        for i in range(n_insert):
            score = float(i)          # 分数单调递增，确保每个都被考虑
            engine._update_factor_pool(score, _make_factor())

        assert len(engine.factor_pool) == ModelConfig.FACTOR_TOP_K

    def test_pool_contains_top_k_scores(self, engine):
        """池中保留的是历史最高 K 个分数；被丢弃的分数均低于池内最小分（T3.2）。"""
        n_insert = ModelConfig.FACTOR_TOP_K + 5
        all_scores = []
        for i in range(n_insert):
            score = float(i)
            all_scores.append(score)
            engine._update_factor_pool(score, _make_factor())

        # 池中分数（堆元素第 0 项）
        pool_scores = sorted(s for s, _cnt, _ in engine.factor_pool)
        # 历史最高 K 个分数
        expected_top_k = sorted(all_scores)[-ModelConfig.FACTOR_TOP_K:]

        assert pool_scores == expected_top_k

    def test_low_score_rejected_when_pool_full(self, engine):
        """池已满时，分数不高于堆顶的因子不被入池（T3.2 严格大于规则）。"""
        # 先填满池，分数为 10.0 ~ 10.0+K-1
        for i in range(ModelConfig.FACTOR_TOP_K):
            engine._update_factor_pool(10.0 + i, _make_factor())

        min_pool_score_before = engine.factor_pool[0][0]   # 堆顶（最小分）

        # 尝试插入低于堆顶的分数
        engine._update_factor_pool(min_pool_score_before - 1.0, _make_factor())

        # 池大小不变，堆顶不变
        assert len(engine.factor_pool) == ModelConfig.FACTOR_TOP_K
        assert engine.factor_pool[0][0] == min_pool_score_before

    def test_equal_score_not_replacing_pool_entry(self, engine):
        """分数等于堆顶时，不替换（严格大于才替换，T3.2）。"""
        for i in range(ModelConfig.FACTOR_TOP_K):
            engine._update_factor_pool(5.0 + i, _make_factor())

        heap_top_before = engine.factor_pool[0][0]
        engine._update_factor_pool(heap_top_before, _make_factor())   # 等于堆顶，不替换

        assert len(engine.factor_pool) == ModelConfig.FACTOR_TOP_K
        assert engine.factor_pool[0][0] == heap_top_before

    def test_factor_stored_on_cpu(self, engine):
        """因子张量以 CPU tensor 存储，节省 VRAM（T3.1）。"""
        engine._update_factor_pool(1.0, _make_factor())
        _, _cnt, stored = engine.factor_pool[0]
        assert stored.device.type == "cpu"


# ── TestApplyCorrPenalty ──────────────────────────────────────────────────

class TestApplyCorrPenalty:
    """测试 _apply_corr_penalty 的各种场景（T3.3~T3.7）。"""

    def test_penalty_applied_for_identical_factors(self, engine):
        """两个完全相同的因子相关系数为 1.0，超过阈值 0.7，reward 应乘以 CORR_PENALTY（T3.3, T3.4）。"""
        factor = torch.randn(5, 50)
        # 将该因子（相同张量）放入池中，corr = 1.0
        engine._update_factor_pool(1.0, factor)

        reward = torch.tensor(2.0)
        penalized = engine._apply_corr_penalty(reward, factor)

        expected = reward * ModelConfig.CORR_PENALTY
        assert torch.isclose(penalized, expected), (
            f"期望 {expected.item():.4f}，得到 {penalized.item():.4f}"
        )

    def test_penalty_applied_only_once_for_multiple_correlated_pool_entries(self, engine):
        """多个池因子均与候选因子高相关时，惩罚最多施加一次（T3.4）。"""
        factor = torch.randn(5, 50)
        # 放入 3 个与 factor 完全相同的因子（使用不同分数避免堆比较 tensor）
        for i in range(3):
            engine._update_factor_pool(1.0 + i * 0.1, factor.clone())

        reward = torch.tensor(4.0)
        penalized = engine._apply_corr_penalty(reward, factor)

        # 只惩罚一次：4.0 * 0.5 = 2.0，而非 4.0 * 0.5^3 = 0.5
        expected = reward * ModelConfig.CORR_PENALTY
        assert torch.isclose(penalized, expected), (
            f"惩罚应仅施加一次，期望 {expected.item():.4f}，得到 {penalized.item():.4f}"
        )

    def test_no_penalty_for_uncorrelated_factor(self, engine):
        """与池中因子相关系数低于阈值时，reward 不变（T3.3, T3.4）。"""
        pool_factor = torch.zeros(5, 50)
        pool_factor[0, 0] = 1.0   # 近似常数，corr 极低
        # 构造与 pool_factor 完全无关的因子
        candidate = torch.randn(5, 50)
        # 强制让 candidate 与 pool_factor 正交（减去投影）
        pf_flat = pool_factor.reshape(-1).float()
        c_flat = candidate.reshape(-1).float()
        proj = (c_flat @ pf_flat) / (pf_flat @ pf_flat + 1e-8) * pf_flat
        c_ortho = c_flat - proj
        # 正交化后 std 可能足够大
        if c_ortho.std() > 1e-4:
            engine._update_factor_pool(1.0, pool_factor)
            reward = torch.tensor(3.0)
            # 计算实际相关系数
            f_c = c_ortho - c_ortho.mean()
            p_c = pf_flat - pf_flat.mean()
            corr = (f_c @ p_c) / (f_c.norm() * p_c.norm() + 1e-8)
            if corr.abs() <= ModelConfig.CORR_THRESHOLD:
                candidate_tensor = c_ortho.reshape(5, 50)
                result = engine._apply_corr_penalty(reward, candidate_tensor)
                assert torch.isclose(result, reward), (
                    f"低相关因子不应被惩罚，期望 {reward.item():.4f}，得到 {result.item():.4f}"
                )

    def test_constant_factor_skips_penalty(self, engine):
        """std < 1e-4 的常数因子跳过惩罚，reward 不变（T3.7）。"""
        # 先向池中插入一个正常因子（corr 会是 1.0 若常数，但应在 std 检测前返回）
        pool_factor = torch.randn(5, 50)
        engine._update_factor_pool(1.0, pool_factor)

        # 常数因子：全为同一值
        constant_factor = torch.full((5, 50), 3.14)
        reward = torch.tensor(2.5)
        result = engine._apply_corr_penalty(reward, constant_factor)

        assert torch.isclose(result, reward), (
            f"常数因子（std < 1e-4）应跳过惩罚，期望 {reward.item():.4f}，得到 {result.item():.4f}"
        )

    def test_empty_pool_returns_reward_unchanged(self, engine):
        """池为空时直接返回原始 reward，无任何修改（T3.6）。"""
        assert len(engine.factor_pool) == 0, "初始池应为空"

        reward = torch.tensor(1.5)
        factor = torch.randn(5, 50)
        result = engine._apply_corr_penalty(reward, factor)

        assert torch.isclose(result, reward), (
            f"空池时 reward 应不变，期望 {reward.item():.4f}，得到 {result.item():.4f}"
        )

    def test_corr_penalty_value_is_config_value(self, engine):
        """惩罚系数严格等于 ModelConfig.CORR_PENALTY（T3.5）。"""
        factor = torch.randn(5, 50)
        engine._update_factor_pool(1.0, factor)

        reward = torch.tensor(1.0)
        result = engine._apply_corr_penalty(reward, factor)

        assert float(result) == pytest.approx(ModelConfig.CORR_PENALTY), (
            f"惩罚后值应为 CORR_PENALTY={ModelConfig.CORR_PENALTY}，得到 {float(result):.6f}"
        )

    def test_reward_is_tensor_after_penalty(self, engine):
        """惩罚后返回值仍为 tensor，类型未改变。"""
        factor = torch.randn(5, 50)
        engine._update_factor_pool(1.0, factor)

        reward = torch.tensor(2.0)
        result = engine._apply_corr_penalty(reward, factor)

        assert isinstance(result, torch.Tensor)
