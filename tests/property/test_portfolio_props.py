# Feature: mt5-alphagpt-refactor, Property 11: 投资组合状态 round-trip 持久化
"""
Property-based tests for strategy_manager.portfolio (MT5PortfolioManager).

Property 11 Validates: Requirements 8.3

For any portfolio state containing a set of Position objects, calling
save_state() followed by load_state() on a fresh manager instance must
restore a state identical to the original across all 8 fields:
    symbol, ticket, entry_price, entry_time, lot_size,
    direction, highest_price, is_partial_closed
"""

import os
import tempfile

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite

from strategy_manager.portfolio import MT5PortfolioManager, Position


# ── Hypothesis strategies ─────────────────────────────────────────────────────

# Valid MT5 symbol strings (e.g. "XAUUSD", "EURUSD")
symbol_strategy = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    min_size=3,
    max_size=8,
)

# MT5 ticket numbers: non-negative integers
ticket_strategy = st.integers(min_value=1, max_value=10_000_000)

# Prices: realistic positive floats
price_strategy = st.floats(
    min_value=0.001,
    max_value=200_000.0,
    allow_nan=False,
    allow_infinity=False,
)

# Timestamps: Unix epoch seconds (2000–2100)
time_strategy = st.floats(
    min_value=946_684_800.0,   # 2000-01-01
    max_value=4_102_444_800.0, # 2100-01-01
    allow_nan=False,
    allow_infinity=False,
)

# Lot sizes: positive floats in realistic MT5 range
lot_strategy = st.floats(
    min_value=0.01,
    max_value=500.0,
    allow_nan=False,
    allow_infinity=False,
)

direction_strategy = st.sampled_from(["BUY", "SELL"])


@composite
def position_strategy(draw, symbol: str) -> Position:
    """Draw a random Position for a given symbol."""
    entry_price = draw(price_strategy)
    return Position(
        symbol=symbol,
        ticket=draw(ticket_strategy),
        entry_price=entry_price,
        entry_time=draw(time_strategy),
        lot_size=draw(lot_strategy),
        direction=draw(direction_strategy),
        highest_price=draw(price_strategy),
        lowest_price=draw(price_strategy),   # required field added in portfolio refactor
        is_partial_closed=draw(st.booleans()),
    )


@composite
def portfolio_strategy(draw) -> dict:
    """Draw a dict of 1–5 positions keyed by unique symbol strings."""
    n = draw(st.integers(min_value=1, max_value=5))
    # Generate n distinct symbols (uniquify via sets)
    symbols = draw(
        st.lists(
            symbol_strategy,
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    positions = {}
    for sym in symbols:
        pos = draw(position_strategy(sym))
        positions[sym] = pos
    return positions


# ── Property 11: save_state() + load_state() round-trip ──────────────────────
# Validates: Requirements 8.3


@settings(max_examples=100, deadline=None)
@given(original_positions=portfolio_strategy())
def test_property11_portfolio_roundtrip_persistence(original_positions: dict):
    """
    Property 11: save_state() + load_state() 恢复的状态与原状态所有字段完全相等

    For any set of Position objects injected into MT5PortfolioManager:
    1. Override the state file path with a temp file.
    2. Inject positions directly and call save_state().
    3. Create a fresh MT5PortfolioManager pointing to the same temp file.
    4. Call load_state() and assert all 8 fields match for every position.

    Validates: Requirements 8.3
    """
    # Use a real temp file so JSON I/O is exercised on the real filesystem.
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    try:
        # ── Save phase ────────────────────────────────────────────────────────
        saver = MT5PortfolioManager.__new__(MT5PortfolioManager)
        saver.positions = {}
        saver.state_file = tmp_path

        # Inject the generated positions and persist them
        saver.positions = dict(original_positions)  # shallow copy
        saver.save_state()

        # ── Load phase ────────────────────────────────────────────────────────
        loader = MT5PortfolioManager.__new__(MT5PortfolioManager)
        loader.positions = {}
        loader.state_file = tmp_path
        loader.load_state()

        # ── Assertions ────────────────────────────────────────────────────────
        assert set(loader.positions.keys()) == set(original_positions.keys()), (
            f"Loaded symbols {set(loader.positions.keys())} differ from "
            f"original {set(original_positions.keys())}"
        )

        for sym, orig_pos in original_positions.items():
            loaded_pos = loader.positions[sym]

            assert loaded_pos.symbol == orig_pos.symbol, (
                f"[{sym}] symbol mismatch: {loaded_pos.symbol!r} != {orig_pos.symbol!r}"
            )
            assert loaded_pos.ticket == orig_pos.ticket, (
                f"[{sym}] ticket mismatch: {loaded_pos.ticket} != {orig_pos.ticket}"
            )
            assert loaded_pos.entry_price == pytest.approx(orig_pos.entry_price, rel=1e-9), (
                f"[{sym}] entry_price mismatch: {loaded_pos.entry_price} != {orig_pos.entry_price}"
            )
            assert loaded_pos.entry_time == pytest.approx(orig_pos.entry_time, rel=1e-9), (
                f"[{sym}] entry_time mismatch: {loaded_pos.entry_time} != {orig_pos.entry_time}"
            )
            assert loaded_pos.lot_size == pytest.approx(orig_pos.lot_size, rel=1e-9), (
                f"[{sym}] lot_size mismatch: {loaded_pos.lot_size} != {orig_pos.lot_size}"
            )
            assert loaded_pos.direction == orig_pos.direction, (
                f"[{sym}] direction mismatch: {loaded_pos.direction!r} != {orig_pos.direction!r}"
            )
            assert loaded_pos.highest_price == pytest.approx(orig_pos.highest_price, rel=1e-9), (
                f"[{sym}] highest_price mismatch: {loaded_pos.highest_price} != {orig_pos.highest_price}"
            )
            assert loaded_pos.is_partial_closed == orig_pos.is_partial_closed, (
                f"[{sym}] is_partial_closed mismatch: "
                f"{loaded_pos.is_partial_closed} != {orig_pos.is_partial_closed}"
            )

    finally:
        # Clean up temp file regardless of test outcome
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
