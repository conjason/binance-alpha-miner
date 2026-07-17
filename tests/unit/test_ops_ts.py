"""
单元测试：时序算子（TS_MEAN / TS_STD / TS_RANK / TS_CORR_10）及新增趋势算子

验证：
- 输出形状均为 [N, T]
- TS_RANK_5/10/20 值域 ∈ [0, 1)
- TS_CORR_10 在常数输入时输出 0
- 所有算子对边界值（全零、极大值 1e8）无 NaN / Inf
- len(OPS_CONFIG) == 28（原 22 + 新增 6 个趋势/动量算子）

需求：F2.1~F2.6
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from model_core.ops import OPS_CONFIG, _ts_mean, _ts_std, _ts_rank, _ts_corr_10

# ── 常量 ────────────────────────────────────────────────────────────────────────
N, T = 4, 30

# 原始 10 个时序算子（索引 12~21），不含新增的趋势算子
TS_OPS = OPS_CONFIG[12:22]


# ── 辅助：随机正态输入 ───────────────────────────────────────────────────────────
def rand_input() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(N, T)


# ── 1. OPS_CONFIG 长度验证 ───────────────────────────────────────────────────────
class TestOpsConfigLength:
    def test_ops_config_length_equals_22(self):
        """OPS_CONFIG 共 28 个算子（原 12 基础 + 10 时序 + 6 趋势/动量）"""
        assert len(OPS_CONFIG) == 28, (
            f"OPS_CONFIG 长度应为 28，实际为 {len(OPS_CONFIG)}"
        )

    def test_new_ops_count_equals_10(self):
        """时序算子（索引 12-21）共 10 个"""
        assert len(TS_OPS) == 10

    def test_ts_op_names(self):
        """验证 10 个新算子名称正确"""
        expected_names = [
            "TS_MEAN_5", "TS_MEAN_10", "TS_MEAN_20",
            "TS_STD_5",  "TS_STD_10",  "TS_STD_20",
            "TS_RANK_5", "TS_RANK_10", "TS_RANK_20",
            "TS_CORR_10",
        ]
        actual_names = [name for name, _, _ in TS_OPS]
        assert actual_names == expected_names


# ── 2. 输出形状 [N, T] ──────────────────────────────────────────────────────────
class TestOutputShape:
    """全部 10 个新算子：输入 [N, T] → 输出 [N, T]"""

    def _invoke(self, name, fn, arity):
        x = rand_input()
        if arity == 1:
            return fn(x)
        else:  # arity == 2（TS_CORR_10）
            y = rand_input() + 0.1  # 略加偏移，避免 x==y 完全相关
            return fn(x, y)

    @pytest.mark.parametrize("name,fn,arity", TS_OPS, ids=[n for n, _, _ in TS_OPS])
    def test_output_shape(self, name, fn, arity):
        out = self._invoke(name, fn, arity)
        assert out.shape == (N, T), (
            f"{name}: 期望形状 ({N}, {T})，实际 {tuple(out.shape)}"
        )


# ── 3. TS_RANK 值域 ∈ [0, 1) ────────────────────────────────────────────────────
class TestTsRankRange:
    """TS_RANK_5/10/20 输出所有值满足 0 ≤ v < 1"""

    @pytest.mark.parametrize("d", [5, 10, 20])
    def test_rank_lower_bound(self, d):
        x = rand_input()
        out = _ts_rank(x, d)
        assert (out >= 0.0).all(), f"_ts_rank(x, {d}) 存在负值"

    @pytest.mark.parametrize("d", [5, 10, 20])
    def test_rank_upper_bound_strict(self, d):
        x = rand_input()
        out = _ts_rank(x, d)
        assert (out < 1.0).all(), (
            f"_ts_rank(x, {d}) 存在 ≥ 1.0 的值（最大值={out.max().item():.6f}）"
        )

    @pytest.mark.parametrize("name,fn,arity", [
        (n, f, a) for n, f, a in TS_OPS if n.startswith("TS_RANK")
    ], ids=["TS_RANK_5", "TS_RANK_10", "TS_RANK_20"])
    def test_rank_range_via_ops_config(self, name, fn, arity):
        """通过 OPS_CONFIG 中的 lambda 验证同一约束"""
        x = rand_input()
        out = fn(x)
        assert (out >= 0.0).all() and (out < 1.0).all(), (
            f"{name} 值域越界：min={out.min().item():.6f}, max={out.max().item():.6f}"
        )


# ── 4. TS_CORR_10 常数输入输出 0 ────────────────────────────────────────────────
class TestTsCorr10Constant:
    """当 x 或 y 在整个序列中为常数时，TS_CORR_10 在窗口完全填满后（t >= 9）输出应为 0。

    注意：_ts_corr_10 使用 d=10 的因果滑动窗口，左补零填充。
    在前 9 个时间步（t=0..8），窗口内混有零值和真实常数，std 不为零，
    因此 mask (std < 1e-6) 不生效，输出可能非零。
    从第 10 步（t=9）起，窗口完全由真实常数填满，std=0 触发 mask，输出为 0。
    """

    # d=10，满窗口需要 10 个真实值，从 t=9 开始（0-indexed）
    _FULL_WINDOW_START = 9  # 即 T 维度索引 9 起（第 10 个时间步）

    def test_constant_x_outputs_zero_after_warmup(self):
        """x 为常数时，满窗口之后（t >= 9）输出应全为 0"""
        x = torch.ones(N, T) * 3.14
        y = rand_input()
        out = _ts_corr_10(x, y)
        warmed = out[:, self._FULL_WINDOW_START:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6), (
            f"x 为常数时 t>={self._FULL_WINDOW_START} 输出应为 0，"
            f"实际最大绝对值={warmed.abs().max().item():.2e}"
        )

    def test_constant_y_outputs_zero_after_warmup(self):
        """y 为常数时，满窗口之后（t >= 9）输出应全为 0"""
        x = rand_input()
        y = torch.ones(N, T) * -2.71
        out = _ts_corr_10(x, y)
        warmed = out[:, self._FULL_WINDOW_START:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6), (
            f"y 为常数时 t>={self._FULL_WINDOW_START} 输出应为 0，"
            f"实际最大绝对值={warmed.abs().max().item():.2e}"
        )

    def test_both_constant_outputs_zero_after_warmup(self):
        """x 和 y 均为常数时，满窗口之后（t >= 9）输出应全为 0"""
        x = torch.full((N, T), 1.0)
        y = torch.full((N, T), 2.0)
        out = _ts_corr_10(x, y)
        warmed = out[:, self._FULL_WINDOW_START:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6)

    def test_corr_10_via_ops_config_constant_after_warmup(self):
        """通过 OPS_CONFIG 中的 TS_CORR_10 条目验证常数输入的满窗口行为"""
        _, fn, _ = next((name, fn, a) for name, fn, a in TS_OPS if name == "TS_CORR_10")
        x = torch.ones(N, T)
        y = rand_input()
        out = fn(x, y)
        warmed = out[:, self._FULL_WINDOW_START:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6)

    def test_long_constant_input_fully_zero(self):
        """使用更长序列（T=50）确保满窗口大部分为零"""
        T_long = 50
        x = torch.full((N, T_long), 7.0)
        y = torch.randn(N, T_long)
        out = _ts_corr_10(x, y)
        # 从 t=9 起所有位置应为 0
        warmed = out[:, self._FULL_WINDOW_START:]
        assert torch.allclose(warmed, torch.zeros_like(warmed), atol=1e-6), (
            f"长序列中 x 为常数时，满窗口部分应全为 0，"
            f"实际最大绝对值={warmed.abs().max().item():.2e}"
        )


# ── 5. 无 NaN / Inf（边界值测试）───────────────────────────────────────────────
class TestNoNanInf:
    """全部 10 个算子对边界输入（全零、极大值 1e8）不产生 NaN / Inf"""

    @staticmethod
    def _check(name, out):
        assert not torch.isnan(out).any(), f"{name}: 输出包含 NaN"
        assert not torch.isinf(out).any(), f"{name}: 输出包含 Inf"

    def _invoke(self, fn, arity, x):
        if arity == 1:
            return fn(x)
        else:
            return fn(x, x.clone())  # TS_CORR_10：x==y → 常数窗口 → 输出 0（已 mask）

    @pytest.mark.parametrize("name,fn,arity", TS_OPS, ids=[n for n, _, _ in TS_OPS])
    def test_zero_input(self, name, fn, arity):
        x = torch.zeros(N, T)
        out = self._invoke(fn, arity, x)
        self._check(name, out)

    @pytest.mark.parametrize("name,fn,arity", TS_OPS, ids=[n for n, _, _ in TS_OPS])
    def test_large_input(self, name, fn, arity):
        x = torch.full((N, T), 1e8)
        out = self._invoke(fn, arity, x)
        self._check(name, out)

    @pytest.mark.parametrize("name,fn,arity", TS_OPS, ids=[n for n, _, _ in TS_OPS])
    def test_random_input(self, name, fn, arity):
        """随机正态输入亦无 NaN / Inf"""
        x = rand_input()
        out = self._invoke(fn, arity, x)
        self._check(name, out)


# ── 6. 辅助函数单元测试 ─────────────────────────────────────────────────────────
class TestHelperFunctions:
    """直接测试 _ts_mean / _ts_std / _ts_rank / _ts_corr_10 的基本行为"""

    def test_ts_mean_shape(self):
        x = rand_input()
        assert _ts_mean(x, 5).shape == (N, T)
        assert _ts_mean(x, 10).shape == (N, T)
        assert _ts_mean(x, 20).shape == (N, T)

    def test_ts_std_non_negative(self):
        """滑动标准差加了 1e-6 下界，输出应 ≥ 1e-6"""
        x = rand_input()
        for d in (5, 10, 20):
            out = _ts_std(x, d)
            assert (out >= 1e-7).all(), f"_ts_std(x, {d}) 存在 < 1e-7 的值"

    def test_ts_std_constant_near_eps(self):
        """全常数输入的标准差：窗口填满（t >= d-1）后应约等于 1e-6（仅来自下界偏移）。

        _ts_rolling 对长度 d 的窗口左补 d-1 个零，所以：
        - t=0..d-2：窗口包含零和真实常数，std > 0
        - t=d-1 起：窗口全为真实常数，std ≈ 0，加上 1e-6 下界后 ≈ 1e-6
        """
        x = torch.ones(N, T) * 5.0
        for d in (5, 10, 20):
            out = _ts_std(x, d)
            # 只检查满窗口部分（t >= d-1）
            warmed = out[:, d - 1:]
            assert torch.allclose(warmed, torch.full_like(warmed, 1e-6), atol=1e-7), (
                f"_ts_std 常数输入（满窗口 t>={d-1}）应接近 1e-6，实际最大={warmed.max().item():.2e}"
            )

    def test_ts_corr_10_range(self):
        """正常随机输入时，相关系数值域应在 [-1, 1]"""
        torch.manual_seed(0)
        x = torch.randn(N, T)
        y = torch.randn(N, T)
        out = _ts_corr_10(x, y)
        assert (out >= -1.0 - 1e-5).all() and (out <= 1.0 + 1e-5).all(), (
            f"TS_CORR_10 值域越界：min={out.min().item():.6f}, max={out.max().item():.6f}"
        )
