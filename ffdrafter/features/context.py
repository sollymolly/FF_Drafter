"""
features/context.py — opportunity-change features (Phase C).

The market prices offseason roster churn; a box-score model is blind to it. For a
returning player we quantify how much opportunity is opening up or closing down on
his team between season N and N+1, using only what's known preseason N+1 (who is on
the roster) plus season-N production (how much volume those comings/goings represent):

  vacated_target_share / vacated_carry_share   — share of the team's season-N targets /
      carries held by teammates NOT on the roster in N+1 (opportunity freed up).
  incoming_target_share / incoming_carry_share — season-N volume of arriving VETERANS
      (relative to the same team's season-N totals).
  net_target_share / net_carry_share           — vacated minus incoming.
  incoming_rookie_threat                        — 261 minus the best (lowest) draft slot of
      a rookie added at the player's position for N+1 (0 if none); higher = tougher new
      competition. Built from the already-cached draft classes (no extra pull).

Roster membership comes from nflverse season rosters (a rostered-but-injured player
still counts), falling back to 'recorded stats' only if rosters are unavailable. For
the 2026 projection, N+1 membership comes from the current-team crosswalk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import PATHS
from ffdrafter import store
from ffdrafter.data import nfl
from ffdrafter.utils import get_logger

logger = get_logger(__name__)

CONTEXT_FEATURES = [
    "vacated_target_share", "vacated_carry_share",
    "incoming_target_share", "incoming_carry_share",
    "net_target_share", "net_carry_share",
    "incoming_rookie_threat",
]

_LAST_PICK = 261.0     # ~one past the last pick; undrafted/no-rookie -> ~0 threat


def _rookie_threat_map() -> dict:
    """(team, arrival_season, position) -> best (lowest) rookie draft slot, from the
    cached draft classes only (never triggers a pull)."""
    out: dict = {}
    for path in sorted(PATHS["processed"].glob("draft_*.parquet")):
        d = store.load_df(path)
        if d is None or d.empty or "draft_ovr" not in d.columns:
            continue
        d = d.dropna(subset=["team", "position", "season", "draft_ovr"])
        for (team, season, pos), g in d.groupby(["team", "season", "position"]):
            out[(team, int(season), pos)] = float(g["draft_ovr"].min())
    return out


def _next_membership(season_df, rosters, n, ids, project_base_season):
    """team -> set(player_id) for season n+1 (rosters preferred, stats fallback)."""
    if project_base_season is not None and n == project_base_season:
        m = (ids.dropna(subset=["gsis_id", "team"])[["gsis_id", "team"]]
             .rename(columns={"gsis_id": "player_id"}))
    else:
        m = None
        if rosters is not None:
            r = rosters[rosters["season"] == n + 1]
            if not r.empty:
                m = r[["player_id", "team"]]
        if m is None:
            m = season_df[season_df["season"] == n + 1][["player_id", "team"]]
    return m.groupby("team")["player_id"].apply(set).to_dict()


def build_context(season_df, ids, project_base_season: int | None = None, rosters=None) -> pd.DataFrame:
    """One row per (player_id, season N) with the N->N+1 opportunity-change features."""
    if "team" not in season_df.columns:
        logger.warning("season stats have no `team` column yet — opportunity features "
                       "inactive. Refresh season stats to enable them.")
        return pd.DataFrame(columns=["player_id", "season"] + CONTEXT_FEATURES)

    df = season_df.copy()
    df["team"] = df["team"].fillna("FA")
    for c in ("targets", "carries"):
        df[c] = df[c].fillna(0.0)

    if rosters is None:
        lo, hi = int(df["season"].min()), int(df["season"].max())
        rosters = nfl.rosters(range(lo, hi + 2))
    rk_map = _rookie_threat_map()

    team_tot = (df.groupby(["team", "season"])[["targets", "carries"]].sum()
                .rename(columns={"targets": "tt", "carries": "tc"}))
    prod = df.groupby(["player_id", "season"])[["targets", "carries"]].sum()

    rows = []
    for n in sorted(df["season"].unique()):
        next_by_team = _next_membership(df, rosters, n, ids, project_base_season)
        if not next_by_team:
            continue
        for team, grp in df[df["season"] == n].groupby("team"):
            if team in ("FA", ""):
                continue
            now = set(grp["player_id"])
            nxt = next_by_team.get(team, set())
            departed = now - nxt
            arrived = [(p, n) for p in (nxt - now) if (p, n) in prod.index]

            vac = grp.loc[grp["player_id"].isin(departed), ["targets", "carries"]].sum()
            inc_t = float(prod.loc[arrived, "targets"].sum()) if arrived else 0.0
            inc_c = float(prod.loc[arrived, "carries"].sum()) if arrived else 0.0
            tt = float(team_tot.loc[(team, n), "tt"]) if (team, n) in team_tot.index else 0.0
            tc = float(team_tot.loc[(team, n), "tc"]) if (team, n) in team_tot.index else 0.0
            vts, vcs = (vac["targets"] / tt if tt > 0 else 0.0), (vac["carries"] / tc if tc > 0 else 0.0)
            its, ics = (inc_t / tt if tt > 0 else 0.0), (inc_c / tc if tc > 0 else 0.0)

            pos_by_player = dict(zip(grp["player_id"], grp["position"]))
            for pid in now:
                slot = rk_map.get((team, n + 1, pos_by_player.get(pid)))
                threat = max(0.0, _LAST_PICK - slot) if slot is not None else 0.0
                rows.append((pid, n, vts, vcs, its, ics, threat))

    ctx = pd.DataFrame(rows, columns=["player_id", "season",
                                      "vacated_target_share", "vacated_carry_share",
                                      "incoming_target_share", "incoming_carry_share",
                                      "incoming_rookie_threat"])
    ctx["net_target_share"] = ctx["vacated_target_share"] - ctx["incoming_target_share"]
    ctx["net_carry_share"] = ctx["vacated_carry_share"] - ctx["incoming_carry_share"]
    logger.info("Context features: %d (player, season) rows (rosters=%s)",
                len(ctx), "yes" if rosters is not None else "stats-fallback")
    return ctx
