"""
draft/threat.py — per-opponent purchasing power and player-level threat.

WHY THIS EXISTS:
  inflation.py prices the ROOM: bargains anywhere leave money everywhere, so the
  whole board drifts up. It cannot say that the manager who just got Jahmyr Gibbs
  $25 under board is now shopping with house money — and that YOUR number for the
  next star he wants should reflect the risk of NOT getting that player. This
  module attributes banked value and discretionary money to specific managers,
  then prices the consequences for one nominated player: who will chase him,
  what he'll realistically close for, and how far past value it is rational for
  YOU to go rather than let him walk.

PER-MANAGER ECONOMICS (manager_profiles):
  banked_edge  Σ (board value − price paid) over their purchases — value locked in
               under market. Useful identity: remaining money + roster value =
               starting budget + banked_edge, so edge IS a team's power above par
               and doubles as the roster-strength ("strong lineup") signal.
  fill_cost    realistic cost to finish the roster: each open starter slot at the
               MEDIAN inflated price of that position's remaining startable pool
               (stars pull the mean; the median keeps this a baseline, not a wish
               list), open flex at the cheapest flex-eligible pool, $1 per bench.
  surplus      budget_remaining − fill_cost: discretionary star money. The formal
               version of "they got Gibbs under market, so they can afford Puka".
  excess       surplus above the room's median surplus. Early on EVERY team has
               big raw surplus, but they can't all overpay at once — that room
               level is inflation's job. Only the advantage is a targeted threat.
  threat_money excess + banked_edge above the room (each floored at 0) — the two
               distinct licenses to overpay: spare cash burning a hole (the
               hoarder who out-monies the endgame), and banked cushion they can
               give back while still finishing ahead (the Gibbs-bargain team,
               whose CASH surplus actually fell when they bought him).

PLAYER THREAT (assess_player):
  willingness  what each opponent would plausibly pay: managers with an open
               starter slot at the position go to inflated value + deploy_fraction
               × threat_money × star_factor (rich teams chase stars, not $8
               players); everyone else is bench-money at a deep discount. All
               capped by max_bid. A symmetric room therefore prices exactly at
               inflated value — threat only moves prices when someone is ahead.
  exp_price    auction mechanics: the winner pays $1 more than the SECOND-highest
               willingness in the room (we include ourselves at our own board
               price, since we'd enforce value). A lone rich bidder does NOT
               raise the closing price — someone has to push them.
  cost_to_win  top willingness + $1 — what YOU must pay to take him home today.
  premium      bounded uplift to YOUR walk-away price, from the two real costs of
               losing him: scarcity (last of his tier at a position you still
               need — insure a fraction of the drop to the next option) and
               rivalry (a small denial tax, only against a credible bidder whose
               banked edge towers over the room). Deliberately capped: blocking
               has a free-rider problem — you alone pay the overbid while every
               other team shares the benefit — so we never recommend more than a
               price you'd be content actually winning at.

All math runs off the live board's `value` column, so it behaves identically on
the baseline (ESPN) and model-blend boards, at any league size. Knobs: config.THREAT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from ffdrafter.utils import get_logger, normalize_name

logger = get_logger(__name__)


def _available(board, state):
    return board[~board["name_key"].isin(state.drafted_keys())]


def league_pool_prices(state, board, factor: float) -> dict:
    """
    Median inflated price of each position's remaining STARTABLE pool.

    Pool at position p = top-N available by value, N = the league's open starter
    demand at p (strict slots, plus open flex spread across RB/WR/TE by the same
    FLEX_SPLIT used for replacement levels). This is what one more starter at p
    should cost a manager who shops the middle of the pool, not the top of it.
    """
    avail = _available(board, state)
    flex_pos = config.LEAGUE.get("flex_positions", ["RB", "WR", "TE"])
    demand = {p: 0 for p in config.SCORABLE_POSITIONS}
    flex_total = 0
    for m in state.managers:
        needs, flex_open = state.strict_position_needs(m)
        for p, n in needs.items():
            demand[p] = demand.get(p, 0) + n
        flex_total += flex_open
    for p in flex_pos:
        demand[p] += int(round(flex_total * config.FLEX_SPLIT.get(p, 0.0)))

    prices = {}
    for p, d in demand.items():
        pool = avail[avail["position"] == p].nlargest(max(int(d), 1), "value")
        med = float(pool["value"].median()) if len(pool) else 1.0
        prices[p] = max(1.0, med * factor)
    return prices


def manager_profiles(state, board, factor: float | None = None) -> pd.DataFrame:
    """
    One purchasing-power row per manager: budget_left, max_bid, banked_edge,
    fill_cost, surplus, excess, power (= starting budget + banked_edge), and
    edge_vs_room (banked_edge minus the room average — the rivalry signal: how
    far this team's projected final value sits above everyone else's).
    """
    if factor is None:
        from ffdrafter.draft.inflation import inflation_factor
        factor = inflation_factor(state, board)

    value_by_key = dict(zip(board["name_key"], board["value"].astype(float)))
    pool_price = league_pool_prices(state, board, factor)
    flex_pos = config.LEAGUE.get("flex_positions", ["RB", "WR", "TE"])
    flex_price = min([pool_price.get(p, 1.0) for p in flex_pos] or [1.0])
    bench_cost = float(config.THREAT["bench_fill_cost"])

    rows = []
    for m in state.managers:
        # Board value of what they bought vs what they paid; off-board pickups
        # count at the $1 floor.
        edge = sum(value_by_key.get(s.name_key, 1.0) - s.price for s in state.sales_for(m))
        needs, flex_open = state.strict_position_needs(m)
        starters_open = sum(needs.values()) + flex_open
        bench_open = max(0, state.open_slots(m) - starters_open)
        fill = (sum(n * pool_price.get(p, 1.0) for p, n in needs.items())
                + flex_open * flex_price + bench_open * bench_cost)
        budget_left = state.budget_remaining(m)
        rows.append({
            "manager": m,
            "is_me": m == state.my_team,
            "budget_left": budget_left,
            "max_bid": state.max_bid(m),
            "banked_edge": int(round(edge)),
            "fill_cost": int(round(fill)),
            "surplus": int(round(budget_left - fill)),
            "power": int(round(state.budget + edge)),
        })
    df = pd.DataFrame(rows)
    # A room where everyone is broke makes $0 spare an advantage: excess is
    # measured against the median, wherever the median sits.
    df["excess"] = (df["surplus"] - df["surplus"].median()).clip(lower=0).round().astype(int)
    df["edge_vs_room"] = (df["banked_edge"] - df["banked_edge"].mean()).round().astype(int)
    df["threat_money"] = (df["excess"] + df["edge_vs_room"].clip(lower=0)).astype(int)
    return df


def assess_player(state, board, name: str, factor: float | None = None,
                  profiles: pd.DataFrame | None = None) -> dict | None:
    """
    Threat + pricing readout for one player (None if he's not on the board):
    likely bidders with their willingness, expected closing price, my cost to
    win today, and the capped premium that prices the risk of NOT getting him.
    """
    if factor is None:
        from ffdrafter.draft.inflation import inflation_factor
        factor = inflation_factor(state, board)
    key = normalize_name(name)
    row = board[board["name_key"] == key]
    if row.empty:
        return None
    r = row.iloc[0]
    pos = r["position"]
    v_infl = max(1, round(float(r["value"]) * factor))
    cfg = config.THREAT
    if profiles is None:
        profiles = manager_profiles(state, board, factor)

    avail = _available(board, state)
    v_max = float(avail["value"].max()) * factor if len(avail) else float(v_infl)
    star = float(np.clip(v_infl / max(v_max, 1.0), 0.0, 1.0))

    # --- who bids, and up to what number ---
    threats, pricing_bids = [], []
    for pr in profiles.itertuples():
        if pr.is_me:
            continue
        if state.position_needs(pr.manager).get(pos, 0) > 0:
            uplift = cfg["deploy_fraction"] * pr.threat_money * star
            w = int(min(pr.max_bid, round(v_infl + uplift)))
            if w > 0:
                threats.append({"manager": pr.manager, "willingness": w,
                                "surplus": int(pr.surplus),
                                "edge_vs_room": int(pr.edge_vs_room)})
                pricing_bids.append(w)
        else:
            # No open starter slot -> bench/flex hoarding money only.
            w = int(min(pr.max_bid, round(cfg["bench_bid_fraction"] * v_infl)))
            if w > 0:
                pricing_bids.append(w)
    threats.sort(key=lambda t: (-t["willingness"], -t["surplus"]))

    my_max = state.max_bid(state.my_team)
    cost_to_win = (max(pricing_bids) + 1) if pricing_bids else 1
    # Second-price logic, with us in the room enforcing our own board price.
    ws = sorted(pricing_bids + [min(v_infl, my_max)], reverse=True)
    exp_price = int(max(1, min(ws[0], ws[1] + 1))) if len(ws) > 1 else 1

    # --- my premium: the two real costs of NOT getting him ---
    i_need = state.position_needs(state.my_team).get(pos, 0) > 0
    same_tier = avail[(avail["position"] == pos) & (avail["tier"] == r["tier"])]
    is_last = len(same_tier) <= 1

    scarcity = 0
    if i_need and is_last:
        rest = avail[(avail["position"] == pos) & (avail["name_key"] != key)]
        nxt = float(rest["value"].max()) * factor if len(rest) else 0.0
        scarcity = int(round(cfg["scarcity_alpha"] * max(0.0, v_infl - nxt)))

    rivalry = 0
    top = threats[0] if threats else None
    if top and top["willingness"] >= v_infl and top["edge_vs_room"] > 0:
        rivalry = int(round(cfg["rivalry_beta"] * top["edge_vs_room"] * star))

    cap = int(round(cfg["premium_cap"] * v_infl))
    premium = int(min(scarcity + rivalry, cap))

    return {
        "expected_price": exp_price,
        "cost_to_win": int(cost_to_win),
        "threats": threats,          # full list, most threatening first; UI slices
        "top_threat": top["manager"] if top else None,
        "scarcity_premium": scarcity,
        "rivalry_premium": rivalry,
        "premium": premium,
        "premium_cap": cap,
        "suggested_max": int(min(my_max, v_infl + premium)),
        "last_in_tier": is_last,
    }
