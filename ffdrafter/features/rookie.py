"""
features/rookie.py — feature engineering for incoming rookies (no NFL history).

For a rookie, draft capital is the single strongest free predictor of year-1
fantasy output — the NFL invests opportunity in players it drafts early. We learn
the historical relationship (draft slot + position -> rookie-season points) from
past classes and apply it to the new class.

This needs NO college data or API key. College production (via CollegeFootballData)
is a planned refinement that would sharpen these once a key is configured.
"""

from __future__ import annotations

import numpy as np

from ffdrafter.utils import get_logger

logger = get_logger(__name__)

ROOKIE_FEATURES = ["draft_ovr", "log_draft_ovr", "draft_round", "age"]


def _draft_lookup(ids):
    """gsis_id -> (draft_year, draft_ovr, draft_round) from the crosswalk."""
    cols = ["gsis_id", "draft_year", "draft_ovr", "draft_round"]
    d = ids[[c for c in cols if c in ids.columns]].dropna(subset=["gsis_id"]).copy()
    return d


def build_rookie_training(season_df, ids):
    """
    Historical rookie seasons: a player's stat line in the season equal to their
    draft year. Features = draft capital (+ age); target = that season's points.
    """
    import pandas as pd

    draft = _draft_lookup(ids)
    df = season_df.merge(draft, left_on="player_id", right_on="gsis_id", how="inner")
    # A rookie season = the season that equals the player's draft year.
    rookies = df[df["season"] == df["draft_year"]].copy()

    by = ids.dropna(subset=["gsis_id"]).copy()
    by["birth_year"] = pd.to_datetime(by["birthdate"], errors="coerce").dt.year
    birth = dict(zip(by["gsis_id"], by["birth_year"]))
    rookies["age"] = rookies["season"] - rookies["player_id"].map(birth)

    rookies["draft_ovr"] = rookies["draft_ovr"].fillna(260)  # undrafted ~ after last pick
    rookies["log_draft_ovr"] = np.log(rookies["draft_ovr"].clip(lower=1))
    rookies["target_rookie"] = rookies["fantasy_points_league"]

    logger.info("Rookie training rows: %d (drafted skill players with a rookie season)",
                len(rookies))
    return rookies


def build_rookie_features(draft_class_df, ids):
    """Feature rows for an incoming draft class (e.g. 2026)."""
    df = draft_class_df.copy()
    df["draft_ovr"] = df["draft_ovr"].fillna(260)
    df["log_draft_ovr"] = np.log(df["draft_ovr"].clip(lower=1))
    if "age" not in df.columns:
        df["age"] = np.nan
    return df
