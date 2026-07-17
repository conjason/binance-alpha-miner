"""
Property-based tests for model_core.features (MT5FeatureEngineer).

Tests in this file validate universal properties of compute_features
using Hypothesis for random input generation.

Property 1: compute_features Output Shape and NaN Safety Invariant
  Validates: Requirements F1.1, F1.2, F1.10

Property 2: PRESSURE and AC1 Value Range Constraint
  Validates: Requirements F4.1, F4.4
"""

import torch
from hypothesis import given, settings, strategies as st

from model_core.features import MT5FeatureEngineer


# ── Shared OHLCV generator ─────────────────────────────────────────────────────

def _make_ohlcv(N: int, T: int, seed_price: float = 100.0) -> dict:
    """
    Build a valid raw_dict with positive OHLCV tensors.

    close  = seed_price (constant, all positive)
    open   = close * 0.999
    high   = close * 1.001   (>= close)
    low    = close * 0.998   (<= close)
    volume = 1000.0 (positive constant)
    """
    close  = torch.full((N, T), seed_price, dtype=torch.float32)
    open_  = close * 0.999
    high   = close * 1.001
    low    = close * 0.998
    volume = torch.ones(N, T, dtype=torch.float32) * 1000.0
    return {
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }


@st.composite
def ohlcv_strategy(draw, min_T: int = 21) -> dict:
    """
    Composite strategy that generates valid OHLCV raw_dict for given N and T.
    Prices are random positive floats; high >= close >= low > 0.
    """
    N = draw(st.integers(min_value=1, max_value=10))
    T = draw(st.integers(min_value=min_T, max_value=200))

    base_price = draw(
        st.floats(min_value=1.0, max_value=10_000.0,
                  allow_nan=False, allow_infinity=False)
    )

    close  = torch.full((N, T), base_price, dtype=torch.float32)
    open_  = close * 0.999
    high   = close * 1.001
    low    = close * 0.998
    volume = torch.ones(N, T, dtype=torch.float32) * 1000.0

    return {
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }


# ── Property 1: Output Shape and NaN Safety Invariant ─────────────────────────
# Validates: Requirements F1.1, F1.2, F1.10


@given(
    N=st.integers(min_value=1, max_value=10),
    T=st.integers(min_value=21, max_value=200),
)
@settings(max_examples=50)
def test_compute_features_shape_and_nan_free(N: int, T: int):
    """
    Property 1: compute_features Output Shape and NaN Safety Invariant

    For any N ∈ [1, 10] and T ∈ [21, 200] with positive OHLCV inputs,
    compute_features must return a tensor of shape [N, 10, T] with no NaN/Inf.

    **Validates: Requirements F1.1, F1.2, F1.10**
    """
    raw_dict = _make_ohlcv(N, T)
    out = MT5FeatureEngineer.compute_features(raw_dict)

    # Shape invariant: 20 features (expanded from 10)
    assert out.shape == (N, 20, T), (
        f"Expected shape ({N}, 20, {T}), got {tuple(out.shape)}"
    )

    # NaN safety
    assert not torch.isnan(out).any(), (
        f"Output contains NaN. Count: {torch.isnan(out).sum().item()}"
    )

    # Inf safety
    assert not torch.isinf(out).any(), (
        f"Output contains Inf. Count: {torch.isinf(out).sum().item()}"
    )


# ── Property 2: PRESSURE and AC1 Value Range Constraint ───────────────────────
# Validates: Requirements F4.1, F4.4


@given(
    N=st.integers(min_value=1, max_value=10),
    T=st.integers(min_value=21, max_value=200),
)
@settings(max_examples=50)
def test_pressure_and_ac1_in_range(N: int, T: int):
    """
    Property 2: PRESSURE and AC1 Value Range Constraint

    For any valid OHLCV input, the PRESSURE feature (index 3) and AC1 feature
    (index 9) must have all values ∈ [-1.0, 1.0].

    PRESSURE is normalised via clamp(-1, 1) per F4.1.
    AC1 is normalised via clamp(-1, 1) per F4.4.

    **Validates: Requirements F4.1, F4.4**
    """
    raw_dict = _make_ohlcv(N, T)
    out = MT5FeatureEngineer.compute_features(raw_dict)

    pressure = out[:, 12, :]  # PRESSURE — index 12 in 20-feature vocab
    ac1      = out[:, 13, :]  # AC1      — index 13 in 20-feature vocab

    # PRESSURE ∈ [-1.0, 1.0]
    assert (pressure >= -1.0).all(), (
        f"PRESSURE has values below -1.0: min = {pressure.min().item():.6f}"
    )
    assert (pressure <= 1.0).all(), (
        f"PRESSURE has values above 1.0: max = {pressure.max().item():.6f}"
    )

    # AC1 ∈ [-1.0, 1.0]
    assert (ac1 >= -1.0).all(), (
        f"AC1 has values below -1.0: min = {ac1.min().item():.6f}"
    )
    assert (ac1 <= 1.0).all(), (
        f"AC1 has values above 1.0: max = {ac1.max().item():.6f}"
    )


# ── Property 3: ATR Non-Negativity ───────────────────────────────────────────
# Validates: Requirements F1.3


@given(
    N=st.integers(min_value=1, max_value=10),
    T=st.integers(min_value=21, max_value=200),
)
@settings(max_examples=50)
def test_atr_non_negative(N: int, T: int) -> None:
    """
    Property 3: ATR Non-Negativity

    For any valid OHLCV input (close > 0, high >= low),
    MT5FeatureEngineer._atr raw output (before log1p) must satisfy
    all values >= 0.

    **Validates: Requirements F1.3**
    """
    # Generate strictly positive prices with valid high >= close >= low
    close = torch.rand(N, T) + 1.0                                           # (1, 2]
    open_ = torch.rand(N, T) + 1.0
    high  = torch.maximum(close, open_) + torch.rand(N, T) * 0.5            # >= max(close, open)
    low   = (torch.minimum(close, open_) - torch.rand(N, T) * 0.5).clamp(min=1e-3)  # > 0

    atr_raw = MT5FeatureEngineer._atr(close, high, low)

    assert (atr_raw >= 0).all(), (
        f"ATR raw output contains negative values. "
        f"Min value: {atr_raw.min().item():.6e}, "
        f"Negative count: {(atr_raw < 0).sum().item()}"
    )


# ── Property 4: RVOL Positive Value Constraint ───────────────────────────────
# Validates: Requirements F1.4


@given(
    N=st.integers(min_value=1, max_value=10),
    T=st.integers(min_value=21, max_value=200),
)
@settings(max_examples=50)
def test_rvol_positive(N: int, T: int) -> None:
    """
    Property 4: RVOL Positive Value Constraint

    For any close sequence (close > 0),
    MT5FeatureEngineer._rvol output must satisfy all values >= 1e-9
    (the epsilon lower bound).

    **Validates: Requirements F1.4**
    """
    # Generate strictly positive close prices
    close = torch.rand(N, T) + 1.0  # (1, 2]

    rvol_raw = MT5FeatureEngineer._rvol(close)

    assert (rvol_raw >= 1e-9).all(), (
        f"RVOL output contains values below 1e-9. "
        f"Min value: {rvol_raw.min().item():.6e}, "
        f"Below-threshold count: {(rvol_raw < 1e-9).sum().item()}"
    )


# ── Property 5: RET20 Prefix Zero Padding ─────────────────────────────────────
# Validates: Requirements F1.5


@given(
    N=st.integers(min_value=1, max_value=10),
    T=st.integers(min_value=25, max_value=200),
)
@settings(max_examples=50)
def test_ret20_prefix_zero(N: int, T: int) -> None:
    """
    Property 5: RET20 Prefix Zero Padding

    For any close sequence (length >= 25), the first 20 positions of
    MT5FeatureEngineer._ret20 output must be exactly 0, and positions
    >= 20 must equal log(close[t] / (close[t-20] + 1e-9)).

    **Validates: Requirements F1.5**
    """
    # Generate strictly positive close prices
    close = torch.rand(N, T) + 1.0  # (1, 2]

    ret20 = MT5FeatureEngineer._ret20(close)

    # First 20 positions must be exactly 0
    assert (ret20[:, :20] == 0.0).all(), (
        f"RET20 first 20 positions are not all zero; "
        f"max abs = {ret20[:, :20].abs().max().item():.6f}"
    )

    # Positions >= 20 must equal log(close[t] / (close[t-20] + 1e-9))
    expected = torch.log(close[:, 20:] / (close[:, :-20] + 1e-9))
    assert torch.allclose(ret20[:, 20:], expected, atol=1e-6), (
        f"RET20 values at positions >= 20 differ from expected. "
        f"Max diff: {(ret20[:, 20:] - expected).abs().max().item():.6e}"
    )
