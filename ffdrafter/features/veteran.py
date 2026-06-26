"""
features/veteran.py — feature engineering for players with NFL history.

The core setup is "predict next season from this season": each training row is a
player's season N (features) paired with their season N+1 fantasy points (target).
Applying the trained model to 2025 feature rows projects 2026.

Opportunity metrics (target/carry share, air yards, WOPR) plus age carry most of
the signal; prior fantasy points anchor it. Missing values are fine — the gradient
boosting model handles NaNs natively.
"""

from __future__ import annotations

from ffdrafter.utils import get_logger

logger = get_logger(__name__)

# Feature columns fed to the model (all derived from a single season's stats + age).
FEATURES = [
    "age", "games", "fantasy_points_league", "pts_per_game",
    "targets", "carries", "receptions", "target_share", "air_yards_share",
    "wopr", "receiving_air_yards", "racr",
    "passing_yards", "rushing_yards", "receiving_yards",
    "passing_tds", "rushing_tds", "receiving_tds", "completions", "attempts",
    "passing_epa", "rushing_epa", "receiving_epa",
]


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
    One row per (player, season N) that also has a season N+1, with:
      FEATURES (from N) + position + target_next (fantasy points in N+1).

    No leakage: every feature is from season N, the target is season N+1.
    """
    df = _featurize(season_df, ids)
    df = df[df["games"] >= min_games]

    nxt = df[["player_id", "season", "fantasy_points_league"]].copy()
    nxt["season"] = nxt["season"] - 1  # relabel N+1 points as belonging to season N
    nxt = nxt.rename(columns={"fantasy_points_league": "target_next"})

    train = df.merge(nxt, on=["player_id", "season"], how="inner")
    logger.info("Veteran training rows: %d (seasons %d-%d)",
                len(train), int(train["season"].min()), int(train["season"].max()))
    return train


def build_projection_features(season_df, ids, base_season: int):
    """Feature rows for a single season (e.g. 2025) used to project base_season + 1."""
    df = _featurize(season_df[season_df["season"] == base_season], ids)
    return df
