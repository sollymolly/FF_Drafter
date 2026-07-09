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

    LEAGUE PRICE CURVE (config.PRICE_CURVE): the linear allocation preserves the
    signal's shape, and ESPN AAV's shape is too flat at the top for a competitive
    room. But the real correction is a PLATEAU, not a power curve: this league's
    top ~10 clear $80-90 (a compressed band — willingness saturates well below
    half a budget, so nobody pays $118 for the #1 either), the mid tier sells
    UNDER the flat board, and the tail goes $1-2. So when targets are set we pin
    the top `top_n` dollars to a rank-linear band (top1_target -> topn_target)
    and rescale every other player's above-$1 money so the pool still sums to
    league money: the mid tier sags and the tail compresses toward the floor —
    exactly the observed zero-sum reshape.
    """
    teams = league["teams"]
    budget = league["budget"]
    rsize = config.roster_size(league)
    n_pool = teams * rsize
    money = teams * budget

    v = values.clip(lower=0)
    pool_sum = float(v.sort_values(ascending=False).head(n_pool).sum())
    discretionary = teams * (budget - rsize)
    per_unit = (discretionary / pool_sum) if pool_sum > 0 else 0.0
    dollars = 1 + v * per_unit                      # linear market shape (floats)

    curve = getattr(config, "PRICE_CURVE", {}) or {}
    t_list, t1, tn = curve.get("targets"), curve.get("top1_target"), curve.get("topn_target")
    if t_list:                       # per-rank prices fitted from league history
        targets = np.asarray(t_list, dtype=float)
    elif t1 and tn:                  # two-point band when no history exists
        targets = np.linspace(float(t1), float(tn), num=int(curve.get("top_n", 10)))
    else:
        targets = None
    if targets is not None and len(dollars) > len(targets):
        top_n = len(targets)
        order = dollars.sort_values(ascending=False).index
        top_idx, rest_idx = order[:top_n], order[top_n:]
        # Money the plateau adds must come out of the rest of the pool, pro-rata
        # above the $1 floor ($30 players sag by dollars, $2 players barely move)
        # — EXCEPT that a hard pin would leave a cliff at rank top_n+1 (an $80
        # player next to a near-equal $35 one). The next `top_n` ranks instead
        # taper geometrically from the band's bottom down to wherever the sagged
        # curve naturally sits, and the sag rescales to fund the taper too.
        band_lo = float(targets[-1])
        rest_extra = (dollars.loc[rest_idx] - 1).clip(lower=0)   # rank order, above-floor $
        n_rest_pool = max(0, n_pool - top_n)
        rest_budget = money - float(targets.sum()) - n_rest_pool  # above-floor $ the rest may keep
        K = min(top_n, len(rest_idx))

        def rest_prices(s: float) -> pd.Series:
            base = 1 + rest_extra * s
            if K and band_lo > float(base.iloc[K - 1]):
                anchor = max(float(base.iloc[K - 1]), 1.0)
                phi = (anchor / band_lo) ** (1.0 / K)
                bridge = band_lo * phi ** np.arange(1, K + 1)
                base.iloc[:K] = np.maximum(base.iloc[:K].to_numpy(), bridge)
            return base

        pool_extra = float(rest_extra.iloc[:n_rest_pool].sum())
        s = (rest_budget / pool_extra) if pool_extra > 0 else 0.0
        s = max(0.0, s)
        for _ in range(8):   # fixed point: conserve pool money including the taper
            spent = float((rest_prices(s).iloc[:n_rest_pool] - 1).sum())
            if spent <= 0 or abs(spent - rest_budget) < 1:
                break
            s = max(0.0, s * rest_budget / spent)
        if s <= 0:
            logger.warning("Price-curve targets absorb the whole pool — "
                           "check top1/topn_target vs league money")
        priced = rest_prices(s)
        dollars.loc[rest_idx] = priced
        dollars.loc[top_idx] = targets
        logger.info("Price curve: top-%d pinned $%.0f→$%.0f, taper to $%.0f by rank %d, "
                    "rest scaled ×%.2f", top_n, float(targets[0]), band_lo,
                    float(priced.iloc[K - 1]) if K else band_lo, top_n + K, s)

    out = dollars.round().clip(lower=1).astype(int)
    logger.info("$%.2f per value-unit over top %d players (pool signal=%.0f)",
                per_unit, n_pool, pool_sum)
    return out


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
                      model_weight: float | dict = 0.5, narrative_df=None):
    """
    Phase 4 board: blend our model with the market for a trust-anchored board.

    Skill positions (QB/RB/WR/TE) get model dollars from VOR, calibrated to the same
    total the market assigns those positions (so model$ and market$ are directly
    comparable); DST/K carry the market value. Final value is
        w * model$ + (1 - w) * market$
    where w is model_weight: a single float for the whole board, or a per-position
    dict of learned weights (model/blend.py); positions missing from the dict get
    w=0 (pure market).
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

    if isinstance(model_weight, dict):
        w = b["position"].map(model_weight).fillna(0.0).astype(float)
    else:
        w = float(model_weight)
    b["model_weight"] = w
    b["value"] = (w * b["model_value"] + (1 - w) * b["market_value"]).round().clip(lower=1).astype(int)
    b = assign_tiers(b, "value")
    b["source"] = "model_blend"

    cols = BOARD_COLUMNS + ["projected_pts", "is_rookie", "model_value", "market_value",
                            "model_weight", "narrative_reason", "narrative_mult"]
    b = b.sort_values("value", ascending=False).reset_index(drop=True)
    for c in cols:
        if c not in b.columns:
            b[c] = np.nan
    return b[cols]
