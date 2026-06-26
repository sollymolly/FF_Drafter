"""
valuation/vor.py — replacement levels + Value Over Replacement (VOR).

VOR = projected_pts - replacement_pts, where replacement level per position is the
projected points of the last 'startable' player at that position. The replacement
RANK comes from config.positions(), which is derived from league size — so VOR
automatically reflects a 10-team vs 12-team league.

VOR is the foundation the auction-dollar conversion (auction.py) builds on. The
baseline board (Phase 2) does not need it; the model board (Phase 4) does.
"""

from __future__ import annotations

import config
from ffdrafter.utils import get_logger

logger = get_logger(__name__)


def replacement_levels(proj_df, league: dict = config.LEAGUE) -> dict[str, float]:
    """Projected points of the replacement-level player at each position."""
    pos_cfg = config.positions(league)
    levels: dict[str, float] = {}
    for pos, cfg in pos_cfg.items():
        pool = proj_df[proj_df["position"] == pos].nlargest(
            cfg["replacement_rank"], "projected_pts"
        )
        levels[pos] = float(pool["projected_pts"].iloc[-1]) if len(pool) else 0.0
    return levels


def compute_vor(proj_df, league: dict = config.LEAGUE):
    """
    Add replacement_pts, vor, and vor_rank columns to a projections DataFrame.

    Expects columns: position, projected_pts.
    """
    df = proj_df.copy()
    levels = replacement_levels(df, league)
    df["replacement_pts"] = df["position"].map(levels).fillna(0.0)
    df["vor"] = (df["projected_pts"] - df["replacement_pts"]).clip(lower=0)
    df["vor_rank"] = df["vor"].rank(ascending=False, method="min").astype("Int64")
    logger.info("Replacement levels: %s", {k: round(v, 1) for k, v in levels.items()})
    return df.sort_values("vor", ascending=False).reset_index(drop=True)
