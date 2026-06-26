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


def build_model_board(proj_df, baseline_board, league: dict = config.LEAGUE,
                      model_weight: float = 0.5, narrative_df=None):
    """
    Phase 4 board: blend our model with the market for a trust-anchored board.

    Skill positions (QB/RB/WR/TE) get model dollars from VOR, calibrated to the same
    total the market assigns those positions (so model$ and market$ are directly
    comparable); DST/K carry the market value. Final value is
        model_weight * model$ + (1 - model_weight) * market$.
    Anchoring on the already-complete, calibrated baseline board keeps every player,
    every column, and the league-money calibration intact, and prevents a single model
    quirk (e.g. an over-extrapolated backup) from distorting the board.
    """
    import pandas as pd

    skill = ("QB", "RB", "WR", "TE")
    p_in = proj_df[proj_df["position"].isin(skill)].copy()

    # Bounded narrative nudge: tilt projected points within a hard cap, keep the reason.
    p_in["narrative_mult"] = 1.0
    p_in["narrative_reason"] = None
    if narrative_df is not None and len(narrative_df):
        eint = pd.to_numeric(p_in["espn_id"], errors="coerce")
        mult_by_e = {int(e): float(m) for e, m in zip(narrative_df["espn_id"], narrative_df["narrative_mult"]) if pd.notna(e)}
        reason_by_e = {int(e): r for e, r in zip(narrative_df["espn_id"], narrative_df["narrative_reason"]) if pd.notna(e)}
        p_in["narrative_mult"] = eint.map(mult_by_e).fillna(1.0)
        p_in["narrative_reason"] = eint.map(reason_by_e)
        p_in["projected_pts"] = p_in["projected_pts"] * p_in["narrative_mult"]

    p = compute_vor(p_in, league)

    # VOR / projection lookups by espn_id (preferred) then name_key (fallback).
    p_e = p.dropna(subset=["espn_id"]).copy()
    vor_by_espn = {int(e): float(v) for e, v in zip(p_e["espn_id"], p_e["vor"])}
    vor_by_name, proj_by_name, rookie_by_name = {}, {}, {}
    reason_by_name, nmult_by_name = {}, {}
    for _, r in p.iterrows():
        nk = r["name_key"]
        if nk not in vor_by_name:
            vor_by_name[nk] = float(r["vor"])
            proj_by_name[nk] = float(r.get("projected_pts", float("nan")))
            rookie_by_name[nk] = bool(r.get("is_rookie", False))
            reason_by_name[nk] = r.get("narrative_reason")
            nmult_by_name[nk] = float(r.get("narrative_mult", 1.0))

    def lookup(row, by_espn, by_name, default):
        e = row.get("espn_id")
        if pd.notna(e):
            try:
                ei = int(e)
            except (TypeError, ValueError):
                ei = None
            if ei is not None and ei in by_espn:
                return by_espn[ei]
        return by_name.get(row.get("name_key"), default)

    b = baseline_board.copy()
    b["vor"] = b.apply(lambda r: lookup(r, vor_by_espn, vor_by_name, 0.0), axis=1)
    b["projected_pts"] = b.apply(lambda r: lookup(r, {}, proj_by_name, float("nan")), axis=1)
    b["is_rookie"] = b.apply(lambda r: lookup(r, {}, rookie_by_name, False), axis=1)
    b["narrative_reason"] = b.apply(lambda r: lookup(r, {}, reason_by_name, None), axis=1)
    b["narrative_mult"] = b.apply(lambda r: lookup(r, {}, nmult_by_name, 1.0), axis=1)

    # Model dollars for skill, calibrated to the market's skill-money total.
    skill_mask = b["position"].isin(skill)
    skill_money = float(b.loc[skill_mask, "value"].sum())
    n_skill = int(skill_mask.sum())
    vsum = float(b.loc[skill_mask, "vor"].clip(lower=0).sum())
    per = (skill_money - n_skill) / vsum if vsum > 0 else 0.0

    b["market_value"] = b["value"].astype(int)
    b["model_value"] = b["value"].astype(float)                 # default = market (DST/K)
    b.loc[skill_mask, "model_value"] = 1 + b.loc[skill_mask, "vor"].clip(lower=0) * per

    w = float(model_weight)
    b["value"] = (w * b["model_value"] + (1 - w) * b["market_value"]).round().clip(lower=1).astype(int)
    b = assign_tiers(b, "value")
    b["source"] = "model_blend"

    cols = BOARD_COLUMNS + ["projected_pts", "is_rookie", "model_value", "market_value",
                            "narrative_reason", "narrative_mult"]
    b = b.sort_values("value", ascending=False).reset_index(drop=True)
    for c in cols:
        if c not in b.columns:
            b[c] = np.nan
    return b[cols]
