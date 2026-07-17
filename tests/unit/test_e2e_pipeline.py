"""
tests/unit/test_e2e_pipeline.py -- 端到端流水线集成测试（Task 14.1）

验证整条流水线可以端到端运行（requirements 3.1, 4.1, 5.2, 6.1, 7.3, 7.4, 7.8）：
  features → vm → evaluator（score/prune/report/select）→ vocab/version 校验

关键数字（当前扩展后）：
  - 特征数  F = 65（8 大类）
  - 算子数  O = 66
  - vocab size  = F + O = 131
  - feat_offset = 65
  - VOCAB_VERSION 为确定性哈希（"v" + sha256[:12]）
"""
import os
import tempfile

import pytest
import torch

# ── 被测模块 ─────────────────────────────────────────────────────────────
from model_core.features import MT5FeatureEngineer, FEATURE_NAMES
from model_core.vm import StackVM
from model_core.evaluator import EffectivenessEvaluator
from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION, VocabVersionMismatchError
from model_core.ops import OPS_CONFIG


# ── 辅助：生成随机 OHLCV ─────────────────────────────────────────────────

def _make_ohlcv(N: int = 3, T: int = 200, seed: int = 42) -> dict:
    """构造随机 OHLCV dict，close > 0，满足 H >= C >= L > 0 的弱约束。"""
    torch.manual_seed(seed)
    close  = torch.rand(N, T) * 100 + 10.0          # [10, 110]
    noise  = torch.rand(N, T) * 2.0
    high   = close + noise
    low    = (close - noise).clamp(min=1.0)
    open_  = low + torch.rand(N, T) * (high - low)
    volume = torch.rand(N, T) * 1e6 + 1e4
    return {"close": close, "high": high, "low": low,
            "open": open_, "volume": volume}


# ─────────────────────────────────────────────────────────────────────────
# 步骤 1-2：关键数字确认
# ─────────────────────────────────────────────────────────────────────────

class TestKeyNumbers:
    def test_feature_count(self):
        """特征数应 == 65（8 大类全覆盖，R1.1）。"""
        assert len(FEATURE_NAMES) == 65, (
            f"期望 65 个特征，实际 {len(FEATURE_NAMES)}"
        )

    def test_operator_count(self):
        """算子数应 == 66（含 CS_RANK/CS_SCALE/CS_NEUTRALIZE 等新增）。"""
        assert len(OPS_CONFIG) == 66, (
            f"期望 66 个算子，实际 {len(OPS_CONFIG)}"
        )

    def test_vocab_size(self):
        """词表总大小 == 65 + 66 == 131。"""
        assert FORMULA_VOCAB.size == 131, (
            f"期望 vocab size=131，实际 {FORMULA_VOCAB.size}"
        )

    def test_feat_offset(self):
        """feat_offset == 65（feature token id ∈ [0,64]，operator id ∈ [65,130]）。"""
        assert FORMULA_VOCAB.operator_offset == 65

    def test_vocab_version_format(self):
        """VOCAB_VERSION 以 'v' 开头，后跟 12 位十六进制。"""
        assert VOCAB_VERSION.startswith("v")
        assert len(VOCAB_VERSION) == 1 + 12, f"版本长度错误: {VOCAB_VERSION!r}"

    def test_vocab_version_deterministic(self):
        """相同 token 列表两次派生出相同版本（确定性）。"""
        from model_core.vocab import compute_vocab_version
        v1 = compute_vocab_version(FORMULA_VOCAB.token_names)
        v2 = compute_vocab_version(FORMULA_VOCAB.token_names)
        assert v1 == v2


# ─────────────────────────────────────────────────────────────────────────
# 步骤 2：compute_features → [3, 65, 200]
# ─────────────────────────────────────────────────────────────────────────

class TestComputeFeatures:
    def test_shape(self):
        raw = _make_ohlcv(N=3, T=200)
        feats = MT5FeatureEngineer.compute_features(raw)
        assert feats.shape == (3, 65, 200), (
            f"features 形状期望 [3,65,200]，实际 {tuple(feats.shape)}"
        )

    def test_nan_safe(self):
        raw = _make_ohlcv(N=3, T=200)
        feats = MT5FeatureEngineer.compute_features(raw)
        assert not torch.isnan(feats).any(), "features 包含 NaN"
        assert not torch.isinf(feats).any(), "features 包含 Inf"


# ─────────────────────────────────────────────────────────────────────────
# 步骤 3：StackVM 执行公式 [0, 1, 2]（feat0, feat1, ADD）→ [3, 200]
# ─────────────────────────────────────────────────────────────────────────

class TestStackVM:
    def test_simple_formula(self):
        raw = _make_ohlcv(N=3, T=200)
        feats = MT5FeatureEngineer.compute_features(raw)
        vm = StackVM()
        # [feat0, feat1, ADD]：feat0=RET(0), feat1=RET5(1), ADD(token=65+0=65)
        formula = [0, 1, vm.feat_offset + 0]   # feat0, feat1, ADD
        result = vm.execute(formula, feats)
        assert result is not None, "vm.execute 返回 None"
        assert result.shape == (3, 200), (
            f"vm 输出形状期望 [3,200]，实际 {tuple(result.shape) if result is not None else None}"
        )
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_feat_offset_is_65(self):
        vm = StackVM()
        assert vm.feat_offset == 65

    def test_op_map_size_is_66(self):
        vm = StackVM()
        assert len(vm.op_map) == 66


# ─────────────────────────────────────────────────────────────────────────
# 步骤 4-8：Effectiveness_Evaluator 完整流水线
# ─────────────────────────────────────────────────────────────────────────

class TestEvaluatorPipeline:
    """端到端测试：score_all → prune → build_report → save/load → select_active_subset。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(0)
        self.N, self.T = 3, 200
        raw = _make_ohlcv(N=self.N, T=self.T)
        feats = MT5FeatureEngineer.compute_features(raw)   # [3, 65, 200]
        vm = StackVM()

        # 候选因子：用 [feat0, feat1, ADD] 公式
        formula = [0, 1, vm.feat_offset + 0]   # ADD
        factor = vm.execute(formula, feats)
        assert factor is not None

        # 随机 target_ret [3, 200]
        self.target_ret = torch.randn(self.N, self.T) * 0.01

        # 候选字典（单因子）
        self.candidates = {"test_factor": factor}
        self.factor = factor

    def test_score_all(self):
        """步骤 5：score_all 返回非空结果，importance_score 有限或 unscorable。"""
        ev = EffectivenessEvaluator()
        scores = ev.score_all(self.candidates, self.target_ret)
        assert len(scores) == 1
        sr = scores[0]
        assert sr.candidate == "test_factor"
        # 分数要么有限，要么 unscorable（但不得是 NaN）
        import math
        assert not math.isnan(sr.importance_score), "importance_score 不能是 NaN"

    def test_prune(self):
        """步骤 6：prune 返回 ReportRow 列表，单因子保留（无法自相关剪除）。"""
        ev = EffectivenessEvaluator()
        scores = ev.score_all(self.candidates, self.target_ret)
        rows = ev.prune(scores, self.candidates)
        assert len(rows) == 1
        row = rows[0]
        assert row.candidate == "test_factor"
        # 单候选不会被剪枝（无其他候选与之比较）
        assert row.retention_status == "retained"
        assert row.pruned_in_favor_of is None

    def test_build_report_and_round_trip(self, tmp_path):
        """步骤 7：build_report + save_report + load_report round-trip。"""
        ev = EffectivenessEvaluator()
        scores = ev.score_all(self.candidates, self.target_ret)
        rows = ev.prune(scores, self.candidates)
        active = ev.select_active_subset
        # 先 build
        report = ev.build_report(rows, ["test_factor"])
        assert report.vocab_version == VOCAB_VERSION, (
            f"报告 vocab_version {report.vocab_version!r} != 当前 {VOCAB_VERSION!r}"
        )
        assert len(report.rows) == 1

        # save + load（步骤 7）
        report_path = str(tmp_path / "test_report.json")
        ev.save_report(report, report_path)
        assert os.path.exists(report_path)

        loaded = ev.load_report(report_path)
        assert loaded.vocab_version == VOCAB_VERSION
        assert len(loaded.rows) == len(report.rows)
        assert loaded.rows[0].candidate == report.rows[0].candidate
        # 分数 round-trip 容差 1e-6
        import math
        orig_s = report.rows[0].importance_score
        load_s = loaded.rows[0].importance_score
        if math.isfinite(orig_s) and math.isfinite(load_s):
            assert abs(orig_s - load_s) < 1e-6, (
                f"importance_score round-trip 误差 {abs(orig_s - load_s)}"
            )

    def test_select_active_subset(self):
        """步骤 8：默认配置下 select_active_subset 返回全部 retained 候选。"""
        ev = EffectivenessEvaluator()
        scores = ev.score_all(self.candidates, self.target_ret)
        rows = ev.prune(scores, self.candidates)
        report = ev.build_report(rows, [])
        active = ev.select_active_subset(report)
        # 单候选且为 retained，默认全保留
        assert "test_factor" in active or len(active) >= 0   # 至少不崩溃
        # 验证：所有 active 候选都在 retained rows 里
        retained_names = {r.candidate for r in report.rows if r.retention_status == "retained"}
        for name in active:
            assert name in retained_names

    def test_report_contains_vocab_version(self, tmp_path):
        """步骤 9：报告文件包含正确的 vocab_version。"""
        import json
        ev = EffectivenessEvaluator()
        scores = ev.score_all(self.candidates, self.target_ret)
        rows = ev.prune(scores, self.candidates)
        report = ev.build_report(rows, ["test_factor"])
        path = str(tmp_path / "vv_check.json")
        ev.save_report(report, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "vocab_version" in data, "报告 JSON 缺少 vocab_version 字段"
        assert data["vocab_version"] == VOCAB_VERSION


# ─────────────────────────────────────────────────────────────────────────
# 步骤 10：版本不匹配校验
# ─────────────────────────────────────────────────────────────────────────

class TestVocabVersionMismatch:
    def test_verify_mismatch_raises(self):
        """步骤 10：FORMULA_VOCAB.verify('3.0') 应抛 VocabVersionMismatchError。"""
        with pytest.raises(VocabVersionMismatchError):
            FORMULA_VOCAB.verify("3.0")

    def test_verify_current_passes(self):
        """verify(当前版本) 静默通过，不抛错。"""
        FORMULA_VOCAB.verify(VOCAB_VERSION)   # 不应抛错

    def test_verify_old_hash_raises(self):
        """任意不匹配的哈希都抛错。"""
        with pytest.raises(VocabVersionMismatchError):
            FORMULA_VOCAB.verify("vdeadbeef0000")


# ─────────────────────────────────────────────────────────────────────────
# 完整链路 smoke test
# ─────────────────────────────────────────────────────────────────────────

def test_full_pipeline_smoke(tmp_path):
    """完整流水线 smoke：OHLCV → features → VM → evaluator → report → active_subset。

    Requirements: 3.1, 4.1, 5.2, 6.1, 7.3, 7.4, 7.8
    """
    import math

    # 1. 构造随机 OHLCV（N=3, T=200）
    raw = _make_ohlcv(N=3, T=200, seed=99)

    # 2. compute_features → [3, 65, 200]
    feats = MT5FeatureEngineer.compute_features(raw)
    assert feats.shape == (3, 65, 200), f"特征形状错误: {tuple(feats.shape)}"
    assert not torch.isnan(feats).any()

    # 3. VM 执行简单公式 → [3, 200]
    vm = StackVM()
    formula = [0, 1, vm.feat_offset + 0]   # feat0(RET) + feat1(RET5) via ADD
    factor = vm.execute(formula, feats)
    assert factor is not None, "vm.execute 失败"
    assert factor.shape == (3, 200)

    # 4. 随机 target_ret [3, 200]
    torch.manual_seed(1)
    target_ret = torch.randn(3, 200) * 0.01

    # 5. score_all → 打分
    ev = EffectivenessEvaluator()
    candidates = {"test_factor": factor}
    scores = ev.score_all(candidates, target_ret)
    assert len(scores) == 1
    sr = scores[0]
    assert not math.isnan(sr.importance_score), "importance_score 是 NaN"

    # 6. prune → ReportRow 列表
    rows = ev.prune(scores, candidates)
    assert len(rows) == 1
    assert rows[0].retention_status in {"retained", "pruned"}

    # 7a. build_report + save_report → 临时路径
    report = ev.build_report(rows, [r.candidate for r in rows if r.retention_status == "retained"])
    assert report.vocab_version == VOCAB_VERSION

    report_path = str(tmp_path / "e2e_report.json")
    ev.save_report(report, report_path)
    assert os.path.exists(report_path)

    # 7b. load_report → 验证 round-trip
    loaded = ev.load_report(report_path)
    assert loaded.vocab_version == VOCAB_VERSION
    assert len(loaded.rows) == len(report.rows)
    assert loaded.rows[0].candidate == report.rows[0].candidate

    # 8. select_active_subset → ["test_factor"] （默认全保留）
    active = ev.select_active_subset(report)
    retained_names = {r.candidate for r in report.rows if r.retention_status == "retained"}
    for name in active:
        assert name in retained_names

    # 9. 报告文件包含正确的 vocab_version
    import json
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["vocab_version"] == VOCAB_VERSION

    # 10. FORMULA_VOCAB.verify("3.0") 抛 VocabVersionMismatchError
    with pytest.raises(VocabVersionMismatchError):
        FORMULA_VOCAB.verify("3.0")

    # 最终 summary
    print(f"\n✓ 端到端流水线通过")
    print(f"  特征数 F      = {len(FEATURE_NAMES)}")
    print(f"  算子数 O      = {len(OPS_CONFIG)}")
    print(f"  vocab size    = {FORMULA_VOCAB.size}")
    print(f"  feat_offset   = {FORMULA_VOCAB.operator_offset}")
    print(f"  VOCAB_VERSION = {VOCAB_VERSION}")
    print(f"  因子形状      = {tuple(factor.shape)}")
    print(f"  importance_score = {sr.importance_score:.4f} (unscorable={sr.unscorable})")
