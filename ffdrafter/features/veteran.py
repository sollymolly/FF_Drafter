"""
features/veteran.py — feature engineering for players with NFL history.

The core setup is "predict next season from this season": each training row is a
player's season N (features) paired with their season N+1 outcome. Applying the
trained model to 2025 feature rows projects 2026.

TWO TARGETS (see model/project.py for how they combine):
  ppg_next   — fantasy points PER GAME in N+1 (the skill signal, de-noised of the
               games-played luck that dominates season totals). NaN if no N+1.
  avail_next — games played in N+1 as a fraction of the season (0 if the player did
               NOT return). This is where the SURVIVORSHIP fix lives: the pool keeps
               everyone who had a real season N, so players who washed out contribute
               avail_next=0 instead of being silently dropped by an inner join.

Opportunity metrics (target/carry share, air yards, WOPR) plus age carry most of
the signal; prior fantasy points anchor it. Missing values are fine — the gradient
boosting model handles NaNs natively.
"""

from __future__ import annotations

from ffdrafter.features import context
from ffdrafter.utils import get_logger

logger = get_logger(__name__)


def season_games(season: int) -> int:
    """Regular-season games in a season (the NFL expanded 16 -> 17 in 2021)."""
    return 17 if season >= 2021 else 16

# Feature columns fed to the model. Single-season stats + age, plus the opportunity-
# change signals from features/context.py (offseason roster churn the market prices in).
FEATURES = [
    "age", "games", "fantasy_points_league", "pts_per_game",
    "targets", "carries", "receptions", "target_share", "air_yards_share",
    "wopr", "receiving_air_yards", "racr",
    "passing_yards", "rushing_yards", "receiving_yards",
    "passing_tds", "rushing_tds", "receiving_tds", "completions", "attempts",
    "passing_epa", "rushing_epa", "receiving_epa",
] + context.CONTEXT_FEATURES


def _add_context(df, season_df, ids, project_base_season=None):
    """LEFT-merge the opportunity-change features onto per-(player, season) rows."""
    import pandas as pd

    ctx = context.build_context(season_df, ids, project_base_season=project_base_season)
    if ctx.empty:
        for c in context.CONTEXT_FEATURES:
            df[c] = pd.NA
        return df
    return df.merge(ctx, on=["player_id", "season"], how="left")


def _add_age(df, ids):
    """Attach each player's age during that season (from birthdate in the crosswalk)."""
    import pandas as pd

    by = ids.dropna(subset=["gsis_id"]).copy()
    by["birth_year"] = pd.to_datetime(by["birthdate"], errors="coerce").dt.year
    birth = dict(zip(by["gsis_id"], by["birth_year"]))

    out = df.copy()
    out["birth_year"] = out["player_id"].map(birth)
    out["age"] = out["season"] - out["birth_year"]
    return out


def _featurize(season_df, ids):
    """Per-(player, season) feature rows: add age + per-game scoring rate."""
    df = _add_age(season_df, ids)
    df["pts_per_game"] = df["fantasy_points_league"] / df["games"].clip(lower=1)
    return df


def build_training_table(season_df, ids, min_games: int = 3):
    """
    One row per (player, season N) with FEATURES (from N) + position + two targets:
      ppg_next   = fantasy points / game in N+1   (NaN if the player didn't return)
      avail_next = games(N+1) / season length     (0.0 if the player didn't return)

    SURVIVORSHIP FIX: the pool is every contributor in season N (games >= min_games),
    LEFT-joined to N+1 — so busts/retirements stay in as avail_next=0 rather than
    being dropped. We only keep seasons N whose outcome (N+1) is inside the data, so
    a missing N+1 always means "didn't play", never "not collected yet".

    No leakage: every feature is from season N, both targets are from season N+1.
    """
    import numpy as np

    df = _featurize(season_df, ids)
    df = _add_context(df, season_df, ids)
    max_season = int(df["season"].max())
    pool = df[(df["games"] >= min_games) & (df["season"] <= max_season - 1)].copy()

    nxt = df[["player_id", "season", "games", "fantasy_points_league"]].copy()
    nxt["season"] = nxt["season"] - 1  # attach N+1 stats to season N
    nxt = nxt.rename(columns={"games": "games_next", "fantasy_points_league": "pts_next"})

    train = pool.merge(nxt, on=["player_id", "season"], how="left")
    train["games_next"] = train["games_next"].fillna(0.0)
    train["pts_next"] = train["pts_next"].fillna(0.0)
    season_len_next = (train["season"] + 1).map(season_games)
    train["avail_next"] = (train["games_next"] / season_len_next).clip(0, 1)
    train["ppg_next"] = np.where(train["games_next"] > 0,
                                 train["pts_next"] / train["games_next"], np.nan)

    returned = int((train["games_next"] > 0).sum())
    logger.info("Veteran training pool: %d rows (%d returned, %d washed out; seasons %d-%d)",
                len(train), returned, len(train) - returned,
                int(train["season"].min()), int(train["season"].max()))
    return train


def build_projection_features(season_df, ids, base_season: int):
    """Feature rows for a single season (e.g. 2025) used to project base_season + 1."""
    df = _featurize(season_df[season_df["season"] == base_season], ids)
    # base_season's opportunity change looks ahead to base_season + 1 rosters.
    df = _add_context(df, season_df, ids, project_base_season=base_season)
    return df
