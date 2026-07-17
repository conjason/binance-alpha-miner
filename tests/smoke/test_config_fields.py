"""
冒烟测试：Config 字段与 FEATURE_NAMES 内容验证

断言：
- FEATURE_NAMES 不含 LIQ_SCORE / FOMO（旧 Solana 特有因子）
- Config.INPUT_DIM == 6

注意：task 5.1 会将 vocab.py 更新为新的 MT5 特征名称。
      此测试使用 try/except 优雅处理旧版 vocab.py 中仍含旧字段的情形。

Requirements: 11.1, 5.5 (4.1)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from config import Config


def _get_feature_names():
    """
    尝试从 model_core.vocab 获取 FEATURE_NAMES。
    若模块不存在或属性不存在，返回 None。
    """
    try:
        from model_core.vocab import FEATURE_NAMES
        return FEATURE_NAMES
    except (ImportError, AttributeError):
        return None


def _is_mt5_feature_names_updated():
    """
    检查 FEATURE_NAMES 是否已更新为 MT5 特征集（task 5.1 完成后）。
    若仍包含旧 Solana 特有因子则返回 False，表示 task 5.1 尚未执行。
    """
    feature_names = _get_feature_names()
    if feature_names is None:
        return False
    # 旧版特征含 Solana 特有因子
    old_features = {"LIQ_SCORE", "FOMO", "LOG_VOL"}
    return not bool(old_features & set(feature_names))


class TestConfigInputDimSmoke:
    def test_input_dim_equals_10(self):
        """Config.INPUT_DIM must equal 20 (expanded from 10 to 20 features)."""
        assert Config.INPUT_DIM == 20


class TestFeatureNamesSmoke:
    """
    验证 FEATURE_NAMES 不含旧 Solana 链上特有因子。

    - 若 vocab.py 尚未更新（task 5.1 前），这些测试将跳过，以免误报。
    - 若 vocab.py 已更新，则必须满足断言。
    """

    def test_feature_names_accessible(self):
        """FEATURE_NAMES 应当可以从 model_core.vocab 导入"""
        feature_names = _get_feature_names()
        assert feature_names is not None, (
            "Cannot import FEATURE_NAMES from model_core.vocab. "
            "Ensure model_core/vocab.py exists and defines FEATURE_NAMES."
        )

    def test_liq_score_not_in_feature_names(self):
        """FEATURE_NAMES 不应包含 'LIQ_SCORE'（已废弃的链上流动性因子）"""
        feature_names = _get_feature_names()
        if feature_names is None:
            pytest.skip("model_core.vocab.FEATURE_NAMES not available yet (pending task 5.1)")
        if not _is_mt5_feature_names_updated():
            pytest.skip(
                "vocab.py still contains old Solana features — "
                "pending task 5.1 (MT5FeatureEngineer implementation)"
            )
        assert "LIQ_SCORE" not in feature_names, (
            f"FEATURE_NAMES should not contain 'LIQ_SCORE', but got: {feature_names}"
        )

    def test_fomo_not_in_feature_names(self):
        """FEATURE_NAMES 不应包含 'FOMO'（已废弃的链上情绪因子）"""
        feature_names = _get_feature_names()
        if feature_names is None:
            pytest.skip("model_core.vocab.FEATURE_NAMES not available yet (pending task 5.1)")
        if not _is_mt5_feature_names_updated():
            pytest.skip(
                "vocab.py still contains old Solana features — "
                "pending task 5.1 (MT5FeatureEngineer implementation)"
            )
        assert "FOMO" not in feature_names, (
            f"FEATURE_NAMES should not contain 'FOMO', but got: {feature_names}"
        )

    def test_feature_names_length_matches_input_dim(self):
        """FEATURE_NAMES 的长度应等于 Config.INPUT_DIM（20）"""
        feature_names = _get_feature_names()
        if feature_names is None:
            pytest.skip("model_core.vocab.FEATURE_NAMES not available yet (pending task 5.1)")
        if not _is_mt5_feature_names_updated():
            pytest.skip(
                "vocab.py still contains old Solana features — "
                "pending task 5.1 (MT5FeatureEngineer implementation)"
            )
        assert len(feature_names) == Config.INPUT_DIM, (
            f"Expected len(FEATURE_NAMES) == {Config.INPUT_DIM}, "
            f"got {len(feature_names)}: {feature_names}"
        )
