# Feature: mt5-alphagpt-refactor, Property 10: 手数计算后处理不变量
"""
Property-based tests for strategy_manager.risk (MT5RiskEngine).

Property 10 Validates: Requirements 9.3, 9.4

Property 10: 输出手数是 volume_step 的整数倍，且在 [volume_min, volume_max] 内

For any initial lot calculation input values and symbol specifications,
after MT5RiskEngine post-processing:
  - result >= volume_min
  - result <= volume_max
  - abs(result % volume_step) < 1e-9  (result is an integer multiple of volume_step)
"""

import math
import pytest
from unittest.mock import MagicMock, patch
from hypothesis import given, settings, assume, strategies as st

from strategy_manager.risk import MT5RiskEngine


# ── Hypothesis strategies ─────────────────────────────────────────────────────

# volume_step: typical MT5 values are 0.01 (forex) or 0.1 (metals)
# Use discrete multiples of 0.01 to match realistic MT5 symbol specs (e.g. 0.01, 0.1, 1.0)
volume_step_strategy = st.sampled_from([0.01, 0.1, 0.5, 1.0])

# volume_min: smallest allowed lot (e.g., 0.01 for most forex).
# Generated as a positive integer multiple of volume_step, via @given composition below.
# Here we just generate the multiplier (1–10); the test body derives volume_min.
volume_min_multiplier_strategy = st.integers(min_value=1, max_value=10)

# volume_max: largest allowed lot (e.g., 10–100 for retail accounts).
# Generated as a positive integer multiple of volume_step to ensure volume_max is
# always a clean multiple of volume_step (MT5 guarantees this in real symbol specs).
# Multiplier range 100–10000 gives 1.0–100.0 for step=0.01, or 100–10000 for step=1.0.
volume_max_multiplier_strategy = st.integers(min_value=100, max_value=10_000)

# Account equity: realistic range 1000–100000
equity_strategy = st.floats(
    min_value=1000.0,
    max_value=100_000.0,
    allow_nan=False,
    allow_infinity=False,
)

# Stop pips: 1–100 pips
stop_pips_strategy = st.floats(
    min_value=1.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
)

# Pip value (trade_tick_value): 0.1–10.0 per pip per lot
pip_value_strategy = st.floats(
    min_value=0.1,
    max_value=10.0,
    allow_nan=False,
    allow_infinity=False,
)


def _make_symbol_info(pip_value, volume_step, volume_min, volume_max):
    """Build a mock mt5.symbol_info() return value."""
    info = MagicMock()
    info.trade_tick_value = pip_value
    info.volume_step = volume_step
    info.volume_min = volume_min
    info.volume_max = volume_max
    return info


def _make_account_info(margin_free=999_999.0):
    """Build a mock mt5.account_info() return value with sufficient free margin."""
    info = MagicMock()
    info.margin_free = margin_free
    return info


# ── Property 10: 手数计算后处理不变量 ────────────────────────────────────────
# Validates: Requirements 9.3, 9.4


@settings(max_examples=100, deadline=None)
@given(
    equity=equity_strategy,
    stop_pips=stop_pips_strategy,
    pip_value=pip_value_strategy,
    volume_step=volume_step_strategy,
    volume_min_mult=volume_min_multiplier_strategy,
    volume_max_mult=volume_max_multiplier_strategy,
)
def test_property10_lot_postprocessing_invariants(
    equity: float,
    stop_pips: float,
    pip_value: float,
    volume_step: float,
    volume_min_mult: int,
    volume_max_mult: int,
):
    """
    For any equity/stop_pips/pip_value and symbol specs (volume_step, volume_min,
    volume_max), whenever MT5RiskEngine.calculate_lot() returns a non-zero result,
    the returned lot must satisfy:
      1. result >= volume_min
      2. result <= volume_max
      3. abs(result % volume_step) < 1e-9  (multiple of volume_step)

    Symbol specs are generated so that volume_min and volume_max are themselves
    integer multiples of volume_step, which matches the guarantee provided by real
    MT5 symbol specifications and makes the invariants coherent.

    Validates: Requirements 9.3 (round to volume_step), 9.4 (clamp to [min, max])
    """
    # Derive volume_min and volume_max as exact multiples of volume_step so the
    # clamp operation cannot produce a value that violates the multiple-of-step
    # invariant (which would happen if volume_max itself weren't a multiple of step).
    volume_min: float = round(volume_min_mult * volume_step, 10)
    volume_max: float = round(volume_max_mult * volume_step, 10)

    # Precondition: volume_min must be < volume_max for a valid symbol spec
    assume(volume_min < volume_max)

    engine = MT5RiskEngine()

    symbol_info_mock = _make_symbol_info(pip_value, volume_step, volume_min, volume_max)
    account_info_mock = _make_account_info(margin_free=999_999.0)

    with patch.object(engine, "_get_symbol_info", return_value=symbol_info_mock), \
         patch.object(engine, "_get_account_info", return_value=account_info_mock):

        result = engine.calculate_lot("XAUUSD", equity, stop_pips)

    # If the engine returned 0.0, the position was rejected (e.g., margin check);
    # invariants only apply to accepted lot sizes.
    if result == 0.0:
        return

    # Invariant 1 (Requirement 9.4): result must be >= volume_min
    assert result >= volume_min - 1e-9, (
        f"lot={result} is below volume_min={volume_min} "
        f"(volume_step={volume_step}, equity={equity}, stop_pips={stop_pips})"
    )

    # Invariant 2 (Requirement 9.4): result must be <= volume_max
    assert result <= volume_max + 1e-9, (
        f"lot={result} exceeds volume_max={volume_max} "
        f"(volume_step={volume_step}, equity={equity}, stop_pips={stop_pips})"
    )

    # Invariant 3 (Requirement 9.3): result must be an integer multiple of volume_step
    remainder = result % volume_step
    # Handle floating-point modulo wrap-around: remainder near volume_step is also ~0
    if remainder > volume_step / 2:
        remainder = volume_step - remainder
    assert remainder < 1e-9, (
        f"lot={result} is not a multiple of volume_step={volume_step} "
        f"(remainder={remainder}, equity={equity}, stop_pips={stop_pips})"
    )
