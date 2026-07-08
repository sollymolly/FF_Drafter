"""
draft/engine.py — live recommendations for the draft table.

Answers the two questions you actually ask during an auction:
  1. "What should I pay for player X?"   -> recommend_player(...)
  2. "What's the best value left?"       -> best_available(...)

Plus the supporting reads: each manager's budget/max-bid panel and how many
opponents can still afford a given price (leverage / affordability).

Everything is league-agnostic and board-agnostic: it works the same on the Phase-2
baseline board and the Phase-4 model board.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ffdrafter.draft.inflation import add_inflated_value, inflation_factor
from ffdrafter.utils import get_logger, normalize_name

logger = get_logger(__name__)

POS_ORDER = ("QB", "RB", "WR", "TE", "DST", "K")


def available(board, state):
    """Players still on the board (not yet sold)."""
    return board[~board["name_key"].isin(state.drafted_keys())].copy()


def manager_panel(state) -> pd.DataFrame:
    """Budget / slots / max-bid / open-position needs for every manager."""
    rows = []
    for m in state.managers:
        needs = state.position_needs(m)
        rows.append({
            "manager": m,
            "is_me": m == state.my_team,
            "budget_left": state.budget_remaining(m),
            "filled": state.filled_slots(m),
            "open_slots": state.open_slots(m),
            "max_bid": state.max_bid(m),
            "needs": ",".join(p for p in POS_ORDER if needs.get(p, 0) > 0),
        })
    return pd.DataFrame(rows)


def affordability(state, price: int, exclude_me: bool = True) -> int:
    """How many (opponent) managers can still bid at least `price`."""
    count = 0
    for m in state.managers:
        if exclude_me and m == state.my_team:
            continue
        if state.max_bid(m) >= price:
            count += 1
    return count


def last_in_tier(state, board, name_key: str) -> bool:
    """True if taking this player empties their (position, tier) among available players."""
    row = board[board["name_key"] == name_key]
    if row.empty:
        return False
    pos, tier = row.iloc[0]["position"], row.iloc[0]["tier"]
    av = available(board, state)
    same_tier = av[(av["position"] == pos) & (av["tier"] == tier)]
    return len(same_tier) <= 1


def best_available(state, board, n: int = 25, position: str | None = None,
                   factor: float | None = None) -> pd.DataFrame:
    """Top remaining players by inflation-adjusted value, annotated for the table."""
    if factor is None:
        factor = inflation_factor(state, board)
    av = available(board, state)
    if position and position != "ALL":
        av = av[av["position"] == position]
    av = add_inflated_value(av, factor)
    av = av.nlargest(n, "inflated_value")
    av["my_max_bid"] = state.max_bid(state.my_team)
    av["opp_can_afford"] = av["inflated_value"].apply(lambda p: affordability(state, int(p)))
    cols = ["name", "position", "team", "value", "inflated_value",
            "my_max_bid", "opp_can_afford", "tier", "aav", "adp"]
    return av[[c for c in cols if c in av.columns]].reset_index(drop=True)


def recommend_player(state, board, name: str, factor: float | None = None) -> dict | None:
    """Full recommendation for a single (nominated) player, or None if not found."""
    if factor is None:
        factor = inflation_factor(state, board)
    key = normalize_name(name)
    row = board[board["name_key"] == key]
    if row.empty:
        return None
    r = row.iloc[0]
    base = int(r["value"])
    inflated = max(1, round(base * factor))
    my_max = state.max_bid(state.my_team)
    return {
        "name": r["name"],
        "position": r["position"],
        "team": r["team"],
        "board_value": base,
        "inflated_value": inflated,
        "my_max_bid": my_max,
        "suggested_max": min(inflated, my_max),
        "opp_can_afford": affordability(state, inflated),
        "tier": int(r["tier"]) if "tier" in r and pd.notna(r["tier"]) else None,
        "last_in_tier": last_in_tier(state, board, key),
        "already_drafted": state.is_drafted(name),
        "narrative_reason": r.get("narrative_reason"),
        "is_rookie": bool(r.get("is_rookie")) if pd.notna(r.get("is_rookie")) else False,
    }


def nomination_board(state, board, n: int = 15, factor: float | None = None) -> pd.DataFrame:
    """
    Rank available players by how good they are to NOMINATE — the auction lever v1
    intentionally deferred. The play: throw out players OTHER managers still need and
    can pay for (draining their budgets on spots you've filled), while HOLDING the
    players you actually want until the room's money thins.

    For each available player we compute, per opponent:
      opp_need       — opponents with an open starter slot at that position;
      opp_demand     — of those, how many can also afford the (inflated) price — the
                       real bidders who will push it up;
    and flag `i_target` (you still need the spot and can afford him). Nominate score =
    price x opp_demand, zeroed for your own targets so they sink down the list.
    """
    if factor is None:
        factor = inflation_factor(state, board)
    av = available(board, state)
    if av.empty:
        return av
    av = add_inflated_value(av, factor)

    opps = [m for m in state.managers if m != state.my_team]
    opp_needs = {m: state.position_needs(m) for m in opps}
    opp_maxbid = {m: state.max_bid(m) for m in opps}
    my_needs = state.position_needs(state.my_team)
    my_max = state.max_bid(state.my_team)

    def demand(row) -> int:
        p, v = row["position"], int(row["inflated_value"])
        return sum(1 for m in opps if opp_needs[m].get(p, 0) > 0 and opp_maxbid[m] >= v)

    av["opp_need"] = av["position"].map(lambda p: sum(1 for m in opps if opp_needs[m].get(p, 0) > 0))
    av["opp_can_afford"] = av["inflated_value"].apply(lambda v: affordability(state, int(v)))
    av["opp_demand"] = av.apply(demand, axis=1)
    my_need = av["position"].map(lambda p: my_needs.get(p, 0) > 0)
    av["i_target"] = my_need & (av["inflated_value"] >= 3) & (av["inflated_value"] <= my_max)
    av["nominate_score"] = (av["inflated_value"] * av["opp_demand"]).where(~av["i_target"], 0)
    av["suggestion"] = np.where(
        av["i_target"], "HOLD — you want him",
        np.where(av["opp_demand"] > 0, "DRAIN — others need & can pay", "low leverage"))

    out = av.sort_values(["nominate_score", "inflated_value"], ascending=False)
    cols = ["name", "position", "team", "inflated_value", "opp_demand",
            "opp_need", "opp_can_afford", "suggestion", "nominate_score"]
    return out[cols].head(n).reset_index(drop=True)
