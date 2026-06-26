"""
draft/inflation.py — live market inflation.

As the draft unfolds, compare the money still in the room to the value still on the
board:

    inflation = (sum of every manager's remaining budget)
                ---------------------------------------------------
                (board value of the players who will actually be bought)

The denominator is the top-N available players by value, where N = total remaining
open roster slots across the league (only that many more players get bought). Both
sides carry the mandatory $1-per-slot, so the factor starts ~1.0 and drifts:

  > 1.0  early BARGAINS (stars went UNDER value) -> money is still in the room
         chasing fewer players, so what's left costs MORE than the board (pay up).
  < 1.0  early OVERPAYS (stars went OVER value)  -> money has been drained from the
         room, so what's left goes for LESS than the board (bargains are coming).

(An overpay removes more money than board value, so numerator falls faster than
denominator -> factor drops below 1. This is the standard auction-inflation result.)
"""

from __future__ import annotations

from ffdrafter.utils import get_logger

logger = get_logger(__name__)


def inflation_factor(state, board) -> float:
    """Return the current market inflation multiplier (1.0 == on par with the board)."""
    avail = board[~board["name_key"].isin(state.drafted_keys())]
    slots_left = state.total_open_slots()
    if slots_left <= 0 or avail.empty:
        return 1.0
    pool = avail.nlargest(int(slots_left), "value")
    remaining_value = float(pool["value"].sum())
    remaining_money = float(state.total_remaining_money())
    if remaining_value <= 0:
        return 1.0
    return remaining_money / remaining_value


def add_inflated_value(board, factor: float):
    """Add an 'inflated_value' column = board value re-scaled by the inflation factor."""
    out = board.copy()
    out["inflated_value"] = (out["value"] * factor).round().clip(lower=1).astype(int)
    return out
