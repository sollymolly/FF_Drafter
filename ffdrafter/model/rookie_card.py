"""
model/rookie_card.py — per-rookie deep-dive content for the draft-day app.

For each incoming rookie we assemble: draft capital + landing spot + college + the
projection band, and a set of historical COMPS — past rookies drafted at a similar
slot and position, with what they actually scored as rookies. That comp set is the
honest "is the hype real?" context: it shows the realistic range for this draft slot.

(College production via CollegeFootballData is a planned refinement once a key is set;
draft_class already carries each rookie's college name.)
"""

from __future__ import annotations

import pandas as pd

import config
from ffdrafter import store
from ffdrafter.data import nfl
from ffdrafter.features import rookie
from ffdrafter.utils import get_logger

logger = get_logger(__name__)


def _historical(season_df, ids):
    h = rookie.build_rookie_training(season_df, ids)
    return h[["name", "position", "draft_ovr", "season", "target_rookie"]].copy()


def comps(hist, position: str, draft_ovr, n: int = 5):
    """Closest historical rookies at the same position by draft slot."""
    d = hist[(hist["position"] == position) & hist["draft_ovr"].notna()].copy()
    if d.empty or pd.isna(draft_ovr):
        return d.head(0)
    d["dist"] = (d["draft_ovr"] - draft_ovr).abs()
    return d.nsmallest(n, "dist").sort_values("draft_ovr")


def build_rookie_cards(proj, season: int | None = None, force_refresh: bool = False):
    """Build + save the rookie deep-dive table for the incoming class."""
    season = season or config.LEAGUE["season"]
    season_df = nfl.season_stats(range(2018, config.LEAGUE["season"]), force_refresh=force_refresh)
    ids = nfl.player_ids(force_refresh=force_refresh)
    dclass = nfl.draft_class(season, force_refresh=force_refresh)
    college = dict(zip(dclass["name_key"], dclass["college"])) if "college" in dclass.columns else {}

    hist = _historical(season_df, ids)
    rookies = proj[proj["is_rookie"]].copy()

    rows = []
    for _, r in rookies.iterrows():
        cps = comps(hist, r["position"], r.get("draft_ovr"))
        comp_str = "; ".join(
            f"{c['name']} (#{int(c['draft_ovr'])}, {int(c['season'])}): {c['target_rookie']:.0f}pts"
            for _, c in cps.iterrows()
        )
        comp_avg = float(cps["target_rookie"].mean()) if len(cps) else float("nan")
        rows.append({
            "name": r["name"], "name_key": r["name_key"], "position": r["position"],
            "team": r["team"], "draft_ovr": r.get("draft_ovr"), "age": r.get("age"),
            "college": college.get(r["name_key"]),
            "projected_pts": r["projected_pts"], "floor": r.get("floor"), "ceiling": r.get("ceiling"),
            "comp_avg_pts": comp_avg, "comps": comp_str,
        })

    cards = pd.DataFrame(rows).sort_values("projected_pts", ascending=False).reset_index(drop=True)
    store.save_df(cards, config.PATHS["processed"] / f"rookie_cards_{season}.parquet")
    logger.info("Rookie cards built: %d rookies", len(cards))
    return cards
