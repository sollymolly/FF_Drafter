"""
data/nfl.py — free NFL data via nflreadpy (nflverse), cached to parquet.

Pulls three things the projection model needs:
  season_stats(seasons) -> season-level skill-position production + opportunity
  player_ids()          -> gsis <-> espn <-> name crosswalk, age, draft capital
  draft_class(season)   -> drafted rookies with draft capital + landing team

Fantasy points are computed from raw components under config.SCORING, so PPR /
half-PPR / standard all work. nflreadpy returns polars frames; we convert to pandas.
"""

from __future__ import annotations

from config import PATHS, SCORING
from ffdrafter import store
from ffdrafter.utils import get_logger, normalize_name

logger = get_logger(__name__)

SKILL = ("QB", "RB", "WR", "TE")


def _to_pandas(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def fantasy_points(df, scoring: dict = SCORING):
    """Vectorized league fantasy points from season component columns."""
    def g(col):
        return df[col].fillna(0) if col in df.columns else 0

    fumbles_lost = g("sack_fumbles_lost") + g("rushing_fumbles_lost") + g("receiving_fumbles_lost")
    two_pt = g("passing_2pt_conversions") + g("rushing_2pt_conversions") + g("receiving_2pt_conversions")
    return (
        g("passing_yards") * scoring["passing_yd"]
        + g("passing_tds") * scoring["passing_td"]
        + g("passing_interceptions") * scoring["interception"]
        + g("rushing_yards") * scoring["rushing_yd"]
        + g("rushing_tds") * scoring["rushing_td"]
        + g("receiving_yards") * scoring["receiving_yd"]
        + g("receiving_tds") * scoring["receiving_td"]
        + g("receptions") * scoring["reception"]
        + fumbles_lost * scoring["fumble_lost"]
        + two_pt * scoring["two_pt"]
    )


def season_stats(seasons, force_refresh: bool = False):
    """Season-level skill-position stats with league fantasy points + opportunity."""
    import nflreadpy as nfl

    seasons = list(seasons)
    cache = PATHS["processed"] / f"nfl_season_{min(seasons)}_{max(seasons)}.parquet"
    if not force_refresh:
        cached = store.load_df(cache)
        if cached is not None:
            logger.info("Cached season stats: %d rows", len(cached))
            return cached

    logger.info("Pulling nflverse season stats %s..%s ...", min(seasons), max(seasons))
    raw = _to_pandas(nfl.load_player_stats(seasons=seasons, summary_level="reg"))
    raw = raw[raw["position"].isin(SKILL)].copy()
    raw["fantasy_points_league"] = fantasy_points(raw)

    keep = [
        "player_id", "player_display_name", "position", "season", "games",
        "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
        "carries", "rushing_yards", "rushing_tds",
        "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_air_yards",
        "target_share", "air_yards_share", "wopr", "racr",
        "passing_epa", "rushing_epa", "receiving_epa",
        "fantasy_points_ppr", "fantasy_points_league",
    ]
    out = raw[[c for c in keep if c in raw.columns]].rename(
        columns={"player_display_name": "name"})
    out["name_key"] = out["name"].map(normalize_name)
    store.save_df(out, cache)
    return out


def player_ids(force_refresh: bool = False):
    """gsis <-> espn <-> name crosswalk with age + draft capital."""
    import nflreadpy as nfl

    cache = PATHS["processed"] / "ff_playerids.parquet"
    if not force_refresh:
        cached = store.load_df(cache)
        if cached is not None:
            return cached

    ids = _to_pandas(nfl.load_ff_playerids())
    keep = ["gsis_id", "espn_id", "name", "merge_name", "position", "team",
            "birthdate", "age", "draft_year", "draft_round", "draft_pick", "draft_ovr", "college"]
    ids = ids[[c for c in keep if c in ids.columns]].copy()
    ids["name_key"] = ids["name"].map(normalize_name)
    store.save_df(ids, cache)
    return ids


def draft_class(season: int, force_refresh: bool = False):
    """Drafted skill-position rookies for a season with draft capital + landing team."""
    import nflreadpy as nfl

    cache = PATHS["processed"] / f"draft_{season}.parquet"
    if not force_refresh:
        cached = store.load_df(cache)
        if cached is not None:
            return cached

    dp = _to_pandas(nfl.load_draft_picks(seasons=[season]))
    dp = dp[dp["position"].isin(SKILL)].copy()
    dp = dp.rename(columns={"pfr_player_name": "name", "pick": "draft_ovr", "round": "draft_round"})
    keep = ["season", "draft_round", "draft_ovr", "team", "gsis_id", "cfb_player_id",
            "name", "position", "college", "age"]
    dp = dp[[c for c in keep if c in dp.columns]].copy()
    dp["name_key"] = dp["name"].map(normalize_name)
    store.save_df(dp, cache)
    return dp
