"""
valuation/auction.py — turn values into an auction board (dollars + tiers).

Two builders produce the SAME board schema so the live tool never cares which one
ran:

  build_baseline_board(market_df)  → Phase 2: re-scale ESPN consensus AAV onto the
                                      user's league economics.
  build_model_board(proj_df)       → Phase 4: VOR → dollars from our projections.

DOLLAR MODEL (shared idea):
  Every roster spot must cost ≥ $1, so only money ABOVE that is "discretionary":
      discretionary = teams × (budget − roster_size)
  Spread that discretionary money across the draftable pool in proportion to each
  player's value share, then add the mandatory $1 base. Summed over the pool, the
  board's dollars ≈ total league money — and it rescales automatically with league
  size, which is why a 10-team and 12-team board differ correctly.
"""

from __future__ import annotations

import numpy as np

import config
from ffdrafter.utils import get_logger
from ffdrafter.valuation.vor import compute_vor

logger = get_logger(__name__)

BOARD_COLUMNS = [
    "name", "name_key", "position", "team",
    "value", "tier", "vor", "aav", "adp", "espn_id", "source",
]


def _scale_to_budget(values, league: dict):
    """
    Given a non-negative value signal (AAV or VOR) per player, return integer
    auction dollars: $1 base for everyone plus a share of discretionary money,
    calibrated so the draftable pool sums to ~total league money.
    """
    teams = league["teams"]
    budget = league["budget"]
    rsize = config.roster_size(league)
    n_pool = teams * rsize

    v = values.clip(lower=0)
    pool_sum = float(v.sort_values(ascending=False).head(n_pool).sum())
    discretionary = teams * (budget - rsize)
    per_unit = (discretionary / pool_sum) if pool_sum > 0 else 0.0

    dollars = (1 + v * per_unit).round().clip(lower=1).astype(int)
    logger.info("$%.2f per value-unit over top %d players (pool signal=%.0f)",
                per_unit, n_pool, pool_sum)
    return dollars


def assign_tiers(df, value_col: str = "value", players_per_tier: int = 6, max_tiers: int = 8):
    """
    Tier players WITHIN each position using natural breaks: split the sorted values
    at their largest gaps so each position gets ~ n/players_per_tier tiers (capped at
    max_tiers). A tier is a cluster of similar-value players; a tier boundary is a
    real value cliff -- which is what makes "last player in a tier" a useful signal.
    """
    out = df.copy()
    out["tier"] = 1
    for _, grp in out.groupby("position"):
        grp = grp.sort_values(value_col, ascending=False)
        vals = grp[value_col].to_numpy()
        n = len(vals)
        if n <= 1:
            continue
        target = min(max_tiers, max(1, round(n / players_per_tier)))
        k = min(target - 1, n - 1)          # number of tier boundaries to place
        if k <= 0:
            continue
        drops = vals[:-1] - vals[1:]         # value drop after each player
        boundaries = {int(i) for i in np.argsort(drops)[-k:]}  # k largest gaps
        tier = 1
        for pos_i, idx in enumerate(grp.index):
            out.at[idx, "tier"] = tier
            if pos_i in boundaries:
                tier += 1
    return out


def _finalize(df):
    """Ensure all board columns exist, ordered, sorted by value desc."""
    out = df.copy()
    for col in BOARD_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out.sort_values("value", ascending=False).reset_index(drop=True)
    return out[BOARD_COLUMNS]


def build_baseline_board(market_df, league: dict = config.LEAGUE):
    """Phase 2 board: ESPN consensus AAV re-scaled onto this league's economics."""
    df = market_df.copy()
    df["value"] = _scale_to_budget(df["aav"], league)
    df["vor"] = np.nan
    df = assign_tiers(df, "value")
    board = _finalize(df)
    logger.info("Baseline board: %d players, total value $%d (league money $%d)",
                len(board), int(board["value"].sum()), config.total_money(league))
    return board


def build_model_board(proj_df, league: dict = config.LEAGUE):
    """Phase 4 board: VOR → dollars from our own projections (same schema)."""
    df = compute_vor(proj_df, league)
    df["value"] = _scale_to_budget(df["vor"], league)
    df = assign_tiers(df, "value")
    return _finalize(df)
