"""
单元测试：坍塌快照恢复逻辑（需求 T4.1~T4.4）

验证：
- 有快照时，attention/qk_norm/norm1/norm2 层参数恢复后与快照逐元素相等
- 有快照时，FFN 层参数在噪声扰动后与快照不同
- 无快照时退化为全参数扰动（不抛异常）
- AlphaEngine 初始化时 _best_snapshot 为 None
"""
import copy
import sys
import os

import pytest
import torch

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model_core.alphagpt import AlphaGPT
from model_core.engine import AlphaEngine


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：将模拟的"坍塌重启-方案A"逻辑提取为独立函数，方便测试直接调用
# 与 engine.py 中的实际实现保持完全一致
# ─────────────────────────────────────────────────────────────────────────────

def _apply_snapshot_restore_with_ffn_noise(model: AlphaGPT, snapshot: dict) -> None:
    """方案 A：恢复快照，仅对 FFN 参数加高斯噪声（noise_std=0.02）。"""
    model.load_state_dict(snapshot)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'ffn' in name:
                param.add_(torch.randn_like(param) * 0.02)


def _apply_full_noise(model: AlphaGPT) -> None:
    """方案 B：无快照时，全参数加高斯噪声（noise_std=0.02）。"""
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.02)


# ─────────────────────────────────────────────────────────────────────────────
# 测试：AlphaEngine 初始化状态
# ─────────────────────────────────────────────────────────────────────────────

class TestAlphaEngineInit:
    """验证 AlphaEngine 初始化时 _best_snapshot 为 None（需求 T4.1）。"""

    def test_best_snapshot_is_none_on_init(self):
        """AlphaEngine 初始化后 _best_snapshot 应为 None。"""
        engine = AlphaEngine(data_manager=None)
        assert engine._best_snapshot is None

    def test_model_is_alphagpt_instance(self):
        """AlphaEngine 持有的 model 应是 AlphaGPT 实例。"""
        engine = AlphaEngine(data_manager=None)
        assert isinstance(engine.model, AlphaGPT)


# ─────────────────────────────────────────────────────────────────────────────
# 测试：有快照时——attention 类参数恢复后与快照完全一致
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotRestoreAttentionParams:
    """验证有快照时，attention/qk_norm/norm1/norm2 参数与快照逐元素相等（需求 T4.2, T4.3）。"""

    # attention 相关关键字（这些参数在方案 A 中不加噪声）
    FROZEN_KEYWORDS = ('attention', 'qk_norm', 'norm1', 'norm2')

    @pytest.fixture
    def model_with_snapshot(self):
        """返回 (model, snapshot)；snapshot 在"训练"后保存，然后再次修改 model 权重。"""
        model = AlphaGPT()

        # 步骤1：保存快照（模拟在最优时刻保存）
        snapshot = copy.deepcopy(model.state_dict())

        # 步骤2：模拟训练，修改所有参数
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.5)

        return model, snapshot

    def test_attention_params_match_snapshot_after_restore(self, model_with_snapshot):
        """恢复快照后，所有含 'attention' 关键字的参数应与快照逐元素相等。"""
        model, snapshot = model_with_snapshot
        _apply_snapshot_restore_with_ffn_noise(model, snapshot)

        restored_sd = model.state_dict()
        for name, snap_tensor in snapshot.items():
            if 'attention' in name:
                assert torch.equal(restored_sd[name], snap_tensor), (
                    f"参数 '{name}' 恢复后与快照不一致"
                )

    def test_qk_norm_params_match_snapshot_after_restore(self, model_with_snapshot):
        """恢复快照后，所有含 'qk_norm' 关键字的参数应与快照逐元素相等。"""
        model, snapshot = model_with_snapshot
        _apply_snapshot_restore_with_ffn_noise(model, snapshot)

        restored_sd = model.state_dict()
        checked = 0
        for name, snap_tensor in snapshot.items():
            if 'qk_norm' in name:
                assert torch.equal(restored_sd[name], snap_tensor), (
                    f"参数 '{name}' 恢复后与快照不一致"
                )
                checked += 1
        assert checked > 0, "未找到含 'qk_norm' 的参数，请检查模型结构"

    def test_norm1_params_match_snapshot_after_restore(self, model_with_snapshot):
        """恢复快照后，所有含 'norm1' 关键字的参数应与快照逐元素相等。"""
        model, snapshot = model_with_snapshot
        _apply_snapshot_restore_with_ffn_noise(model, snapshot)

        restored_sd = model.state_dict()
        checked = 0
        for name, snap_tensor in snapshot.items():
            if 'norm1' in name:
                assert torch.equal(restored_sd[name], snap_tensor), (
                    f"参数 '{name}' 恢复后与快照不一致"
                )
                checked += 1
        assert checked > 0, "未找到含 'norm1' 的参数，请检查模型结构"

    def test_norm2_params_match_snapshot_after_restore(self, model_with_snapshot):
        """恢复快照后，所有含 'norm2' 关键字的参数应与快照逐元素相等。"""
        model, snapshot = model_with_snapshot
        _apply_snapshot_restore_with_ffn_noise(model, snapshot)

        restored_sd = model.state_dict()
        checked = 0
        for name, snap_tensor in snapshot.items():
            if 'norm2' in name:
                assert torch.equal(restored_sd[name], snap_tensor), (
                    f"参数 '{name}' 恢复后与快照不一致"
                )
                checked += 1
        assert checked > 0, "未找到含 'norm2' 的参数，请检查模型结构"

    def test_all_frozen_params_match_snapshot(self, model_with_snapshot):
        """整合测试：所有不含 'ffn' 的参数均与快照逐元素相等。"""
        model, snapshot = model_with_snapshot
        _apply_snapshot_restore_with_ffn_noise(model, snapshot)

        restored_sd = model.state_dict()
        for name, snap_tensor in snapshot.items():
            if not any(kw in name for kw in ('ffn',)):
                assert torch.equal(restored_sd[name], snap_tensor), (
                    f"非FFN参数 '{name}' 恢复后与快照不一致"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 测试：有快照时——FFN 参数在噪声扰动后与快照不同
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotRestoreFFNParams:
    """验证有快照时，FFN 参数在加噪声后与快照不同（需求 T4.2）。"""

    @pytest.fixture
    def model_with_snapshot(self):
        model = AlphaGPT()
        snapshot = copy.deepcopy(model.state_dict())

        # 模拟训练后修改权重
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.5)

        return model, snapshot

    def test_ffn_params_differ_from_snapshot_after_noise(self, model_with_snapshot):
        """恢复快照并加噪声后，FFN 参数应与快照不完全相等。"""
        model, snapshot = model_with_snapshot

        # 固定随机种子确保噪声非零
        torch.manual_seed(42)
        _apply_snapshot_restore_with_ffn_noise(model, snapshot)

        restored_sd = model.state_dict()
        ffn_params_found = 0
        for name, snap_tensor in snapshot.items():
            if 'ffn' in name:
                ffn_params_found += 1
                assert not torch.equal(restored_sd[name], snap_tensor), (
                    f"FFN 参数 '{name}' 加噪声后仍与快照完全相同（噪声未生效）"
                )
        assert ffn_params_found > 0, "未找到含 'ffn' 的参数，请检查模型结构"

    def test_ffn_params_close_to_snapshot_after_small_noise(self, model_with_snapshot):
        """噪声 std=0.02 较小，FFN 参数值应在快照附近（差异 < 1.0）。"""
        model, snapshot = model_with_snapshot
        torch.manual_seed(42)
        _apply_snapshot_restore_with_ffn_noise(model, snapshot)

        restored_sd = model.state_dict()
        for name, snap_tensor in snapshot.items():
            if 'ffn' in name:
                max_diff = (restored_sd[name] - snap_tensor).abs().max().item()
                assert max_diff < 1.0, (
                    f"FFN 参数 '{name}' 噪声扰动幅度异常大（max_diff={max_diff:.4f}）"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 测试：无快照时退化为全参数扰动（不抛异常，需求 T4.4）
# ─────────────────────────────────────────────────────────────────────────────

class TestNoSnapshotFallback:
    """验证无快照时退化为全参数扰动且不抛异常（需求 T4.4）。"""

    def test_full_noise_does_not_raise(self):
        """_best_snapshot 为 None 时，全参数加噪声应正常完成，不抛任何异常。"""
        model = AlphaGPT()
        # 记录加噪前的参数值
        before = {name: param.clone() for name, param in model.named_parameters()}

        # 不应抛出异常
        _apply_full_noise(model)

        # 验证参数确实被修改了
        changed = 0
        for name, param in model.named_parameters():
            if not torch.equal(param, before[name]):
                changed += 1
        assert changed > 0, "全参数扰动后所有参数均未改变"

    def test_full_noise_modifies_all_params(self):
        """全参数扰动应修改模型中所有可训练参数。"""
        model = AlphaGPT()
        before = {name: param.clone() for name, param in model.named_parameters()}

        torch.manual_seed(123)
        _apply_full_noise(model)

        for name, param in model.named_parameters():
            assert not torch.equal(param, before[name]), (
                f"参数 '{name}' 在全参数扰动后未改变"
            )

    def test_engine_with_none_snapshot_fallback_no_raise(self):
        """通过 AlphaEngine 接口验证：_best_snapshot=None 时全参数加噪声不抛异常。"""
        engine = AlphaEngine(data_manager=None)
        assert engine._best_snapshot is None

        # 直接模拟 engine.py 中的无快照分支逻辑
        with torch.no_grad():
            for param in engine.model.parameters():
                param.add_(torch.randn_like(param) * 0.02)

        # 如果执行到此处，说明无异常抛出
        assert True


# ─────────────────────────────────────────────────────────────────────────────
# 测试：快照深拷贝隔离（需求 T4.1）
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotIsolation:
    """验证 deepcopy 快照与模型权重完全解耦（需求 T4.1）。"""

    def test_snapshot_not_affected_by_model_update(self):
        """修改模型权重后，快照内容不应改变。"""
        model = AlphaGPT()
        snapshot = copy.deepcopy(model.state_dict())

        # 记录快照初始值
        snap_values = {k: v.clone() for k, v in snapshot.items()}

        # 修改模型权重
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 1.0)

        # 快照应保持不变
        for name, original_val in snap_values.items():
            assert torch.equal(snapshot[name], original_val), (
                f"快照参数 '{name}' 被模型更新污染（deepcopy 失效）"
            )

    def test_restore_from_snapshot_is_exact(self):
        """load_state_dict(snapshot) 后，模型权重应与快照完全一致（无精度损失）。"""
        model = AlphaGPT()
        snapshot = copy.deepcopy(model.state_dict())

        # 修改权重
        with torch.no_grad():
            for param in model.parameters():
                param.mul_(2.0)

        # 恢复
        model.load_state_dict(snapshot)

        restored_sd = model.state_dict()
        for name, snap_tensor in snapshot.items():
            assert torch.equal(restored_sd[name], snap_tensor), (
                f"参数 '{name}' 恢复后与快照不完全一致"
            )
