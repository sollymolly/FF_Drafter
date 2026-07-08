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
SKILL = ("QB", "RB", "WR", "TE")


def build_rookie_training(season_df, ids):
    """
    Every drafted skill player's ROOKIE season — including the ones who never played
    (target_rookie = 0).

    SURVIVORSHIP FIX: this used to start from the stat lines and inner-join draft
    capital, which silently dropped every drafted rookie who never recorded a season.
    The model therefore learned that every drafted rookie produces, and projected the
    incoming class too optimistically. We now start from the DRAFT CLASS and LEFT-join
    the rookie-season stats, so busts stay in the pool scoring 0.

    Draft classes are restricted to seasons covered by season_df, so a missing stat
    line unambiguously means "did not play", never "not collected yet".
    """
    import pandas as pd

    lo, hi = int(season_df["season"].min()), int(season_df["season"].max())

    d = ids.dropna(subset=["gsis_id", "draft_year"]).copy()
    d["draft_year"] = pd.to_numeric(d["draft_year"], errors="coerce")
    d = d[d["draft_year"].between(lo, hi)].drop_duplicates("gsis_id")
    d["season"] = d["draft_year"].astype(int)   # a rookie season == the draft year
    d = d.rename(columns={"position": "position_listed"})

    stats = (season_df[["player_id", "season", "position", "fantasy_points_league"]]
             .rename(columns={"player_id": "gsis_id", "position": "position_played"}))
    r = d.merge(stats, on=["gsis_id", "season"], how="left")

    # Prefer the position he actually played; fall back to his draft listing, which is
    # all a never-played bust has. Filtering on the listing alone would drop producers
    # whose listed position differs from where they lined up.
    r["position"] = r["position_played"].fillna(r["position_listed"])
    r = r[r["position"].isin(SKILL)].drop(columns=["position_played", "position_listed"])
    r["target_rookie"] = r["fantasy_points_league"].fillna(0.0)

    by = ids.dropna(subset=["gsis_id"]).copy()
    by["birth_year"] = pd.to_datetime(by["birthdate"], errors="coerce").dt.year
    birth = dict(zip(by["gsis_id"], by["birth_year"]))
    r["age"] = r["season"] - r["gsis_id"].map(birth)

    r["draft_ovr"] = r["draft_ovr"].fillna(260)  # undrafted ~ after last pick
    r["log_draft_ovr"] = np.log(r["draft_ovr"].clip(lower=1))

    # "played" == has a stat line. Do NOT use target_rookie > 0: a rookie can play and
    # still score <= 0 (fumbles/INTs), which is a bust, not an absence.
    played = int(r["fantasy_points_league"].notna().sum())
    logger.info("Rookie training rows: %d drafted skill players (%d played, %d never played)",
                len(r), played, len(r) - played)
    return r


def build_rookie_features(draft_class_df, ids):
    """Feature rows for an incoming draft class (e.g. 2026)."""
    df = draft_class_df.copy()
    df["draft_ovr"] = df["draft_ovr"].fillna(260)
    df["log_draft_ovr"] = np.log(df["draft_ovr"].clip(lower=1))
    if "age" not in df.columns:
        df["age"] = np.nan
    return df
