# Feature: mt5-alphagpt-refactor, Property 3: 特征张量形状不变量
# Feature: mt5-alphagpt-refactor, Property 4: 特征值有界性
"""
Property-based tests for model_core.features (MT5FeatureEngineer).

Property 3 Validates: Requirements 3.3, 4.2, 4.4
Property 4 Validates: Requirements 4.3, 4.5
"""

import math
import torch
import pytest
from hypothesis import given, settings, strategies as st, assume

from model_core.features import MT5FeatureEngineer


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_raw_dict(
    n: int,
    t: int,
    open_: torch.Tensor | None = None,
    high: torch.Tensor | None = None,
    low: torch.Tensor | None = None,
    close: torch.Tensor | None = None,
    volume: torch.Tensor | None = None,
) -> dict:
    """Build a valid raw_dict with sensible defaults if tensors not provided."""
    if close is None:
        close = torch.ones(n, t, dtype=torch.float32) * 100.0
    if open_ is None:
        open_ = close * 0.999
    if high is None:
        high = close * 1.001
    if low is None:
        low = close * 0.998
    if volume is None:
        volume = torch.ones(n, t, dtype=torch.float32) * 1000.0

    return {
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }


# ── Property 3: 特征张量形状不变量 ───────────────────────────────────────────
# Validates: Requirements 3.3, 4.2, 4.4


@settings(max_examples=100)
@given(
    n=st.integers(min_value=1, max_value=8),
    t=st.integers(min_value=20, max_value=200),
)
def test_property3_feature_tensor_shape(n: int, t: int):
    """
    For any N symbols and T timesteps, MT5FeatureEngineer.compute_features()
    must return a tensor of shape exactly [N, 10, T].

    Feature count was upgraded from 6 to 10 (added ATR, RVOL, RET20, AC1).

    Validates: Requirements 3.3, 4.2, 4.4
    """
    raw_dict = _make_raw_dict(n, t)
    features = MT5FeatureEngineer.compute_features(raw_dict)

    assert features.shape == (n, 20, t), (
        f"Expected shape ({n}, 20, {t}), got {tuple(features.shape)}"
    )


# ── Property 4: 特征值有界性 ──────────────────────────────────────────────────
# Feature: mt5-alphagpt-refactor, Property 4: 特征值有界性
# Validates: Requirements 4.3, 4.5


@st.composite
def extreme_ohlcv_strategy(draw):
    """
    Generate a single-symbol OHLCV raw_dict (N=1) with extreme values:
    - Prices close to zero (min_value=1e-10)
    - Very large prices (max_value=1e8)
    - Zero volume
    - All prices equal (high == low == open == close, zero range)
    - Normal prices with varying volume
    """
    t = draw(st.integers(min_value=20, max_value=150))
    n = draw(st.integers(min_value=1, max_value=4))

    # Choose an extreme scenario
    scenario = draw(st.sampled_from([
        "tiny_prices",
        "large_prices",
        "zero_volume",
        "zero_range",      # high == low == close == open
        "identical_closes",  # all closes the same
        "normal_extreme_mix",
    ]))

    if scenario == "tiny_prices":
        # Prices close to zero
        base = draw(st.floats(min_value=1e-10, max_value=1e-3,
                               allow_nan=False, allow_infinity=False))
        close = torch.full((n, t), base, dtype=torch.float32)
        open_ = close * draw(st.floats(min_value=0.99, max_value=1.01,
                                        allow_nan=False, allow_infinity=False))
        high  = close * 1.001
        low   = close * 0.999
        # clamp so high >= low >= 0
        high  = torch.clamp(high, min=float(base) * 0.5)
        low   = torch.clamp(low,  min=float(base) * 0.1)
        volume = torch.ones(n, t, dtype=torch.float32) * 100.0

    elif scenario == "large_prices":
        base = draw(st.floats(min_value=1e6, max_value=1e8,
                               allow_nan=False, allow_infinity=False))
        close = torch.full((n, t), base, dtype=torch.float32)
        open_ = close * 0.9995
        high  = close * 1.001
        low   = close * 0.999
        volume = torch.ones(n, t, dtype=torch.float32) * 1000.0

    elif scenario == "zero_volume":
        # Volume is exactly 0
        close  = torch.ones(n, t, dtype=torch.float32) * 1800.0
        open_  = close * 0.9999
        high   = close * 1.001
        low    = close * 0.999
        volume = torch.zeros(n, t, dtype=torch.float32)

    elif scenario == "zero_range":
        # high == low == close == open  (zero candle body and wick)
        base = draw(st.floats(min_value=1.0, max_value=5000.0,
                               allow_nan=False, allow_infinity=False))
        close  = torch.full((n, t), base, dtype=torch.float32)
        open_  = close.clone()
        high   = close.clone()
        low    = close.clone()
        volume = torch.ones(n, t, dtype=torch.float32) * 500.0

    elif scenario == "identical_closes":
        # All closes the same → log-return is always 0
        base = draw(st.floats(min_value=1.0, max_value=10000.0,
                               allow_nan=False, allow_infinity=False))
        close  = torch.full((n, t), base, dtype=torch.float32)
        open_  = close * 0.999
        high   = close * 1.001
        low    = close * 0.999
        volume = torch.ones(n, t, dtype=torch.float32) * 200.0

    else:  # normal_extreme_mix
        # Random valid OHLCV within a wide but finite range
        price_vals = draw(
            st.lists(
                st.floats(min_value=1e-5, max_value=1e7,
                           allow_nan=False, allow_infinity=False),
                min_size=n * t,
                max_size=n * t,
            )
        )
        close = torch.tensor(price_vals, dtype=torch.float32).reshape(n, t)
        open_ = close * 0.9990
        high  = close * 1.001
        low   = close * 0.999
        vol_vals = draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1e6,
                           allow_nan=False, allow_infinity=False),
                min_size=n * t,
                max_size=n * t,
            )
        )
        volume = torch.tensor(vol_vals, dtype=torch.float32).reshape(n, t)

    # Ensure high >= close >= low >= some positive floor to avoid division by zero
    # outside the epsilon guard in the implementation
    high   = torch.max(high, close)
    low    = torch.min(low,  close)

    return {
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }


@settings(max_examples=100)
@given(raw_dict=extreme_ohlcv_strategy())
def test_property4_feature_values_bounded_no_nan_inf(raw_dict: dict):
    """
    For any legal OHLCV input including extreme values (prices near zero,
    zero volume, zero high-low range, identical closes), all values in the
    output of MT5FeatureEngineer.compute_features() must:
      1. Be in the range [-5.0, 5.0]
      2. Contain no NaN values
      3. Contain no Inf values (positive or negative)

    Validates: Requirements 4.3, 4.5
    """
    features = MT5FeatureEngineer.compute_features(raw_dict)

    # ── Assert no NaN ─────────────────────────────────────────────────────
    assert not torch.isnan(features).any(), (
        f"Output contains NaN values. "
        f"NaN count: {torch.isnan(features).sum().item()}"
    )

    # ── Assert no Inf ─────────────────────────────────────────────────────
    assert not torch.isinf(features).any(), (
        f"Output contains Inf values. "
        f"Inf count: {torch.isinf(features).sum().item()}"
    )

    # ── Assert values in [-5.0, 5.0] ─────────────────────────────────────
    bound = 5.0
    min_val = features.min().item()
    max_val = features.max().item()

    assert min_val >= -bound, (
        f"Feature values go below -5.0: min = {min_val:.6f}"
    )
    assert max_val <= bound, (
        f"Feature values exceed 5.0: max = {max_val:.6f}"
    )
