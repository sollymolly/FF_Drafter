"""
model/backtest.py — honest out-of-sample accuracy check for the veteran model.

For a held-out target season T, train only on transitions that finished by T-1, project
T from T-1 features, and compare to what actually happened in T. We report MAE and rank
correlation, and — crucially — the same metrics for a NAIVE baseline (just reuse last
season's points). Beating naive is the bar that says the model is adding real signal;
in fantasy that bar is genuinely hard, so this keeps us honest about how far to trust it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config
from ffdrafter.data import nfl
from ffdrafter.features import veteran
from ffdrafter.model import project
from ffdrafter.utils import get_logger

logger = get_logger(__name__)


def backtest_veteran(target_season: int, history_start: int = 2018) -> pd.DataFrame:
    """Return a per-position (+ overall) table of model vs naive MAE and rank correlation."""
    season_df = nfl.season_stats(range(history_start, config.LEAGUE["season"]))
    ids = nfl.player_ids()

    # Train only on transitions whose target season is <= T-1 (no leakage of season T).
    train = veteran.build_training_table(season_df, ids)
    train = train[train["season"] <= target_season - 2]
    models = project._train_by_position(train, veteran.FEATURES, "target_next",
                                        min_rows=40, min_leaf=15)

    feat = veteran.build_projection_features(season_df, ids, base_season=target_season - 1)
    pred = project._project_by_position(models, feat, veteran.FEATURES)

    actual = (season_df[season_df["season"] == target_season]
              [["player_id", "fantasy_points_league"]]
              .rename(columns={"fantasy_points_league": "actual"}))
    naive = (feat[["player_id", "fantasy_points_league"]]
             .rename(columns={"fantasy_points_league": "naive"}))

    df = pred.merge(actual, on="player_id", how="inner").merge(naive, on="player_id", how="left")

    rows = []
    for pos in list(project.SKILL) + ["ALL"]:
        d = df if pos == "ALL" else df[df["position"] == pos]
        d = d.dropna(subset=["actual", "projected_pts", "naive"])
        if len(d) < 5:
            continue
        rows.append({
            "pos": pos,
            "n": len(d),
            "model_MAE": (d["projected_pts"] - d["actual"]).abs().mean(),
            "naive_MAE": (d["naive"] - d["actual"]).abs().mean(),
            "model_rankcorr": spearmanr(d["projected_pts"], d["actual"]).correlation,
            "naive_rankcorr": spearmanr(d["naive"], d["actual"]).correlation,
        })
    return pd.DataFrame(rows)


def run(target_seasons=(2023, 2024, 2025)) -> pd.DataFrame:
    """Backtest several seasons and print a tidy summary."""
    frames = []
    for t in target_seasons:
        bt = backtest_veteran(t)
        bt.insert(0, "season", t)
        frames.append(bt)
    out = pd.concat(frames, ignore_index=True)
    for col in ["model_MAE", "naive_MAE", "model_rankcorr", "naive_rankcorr"]:
        out[col] = out[col].round(3)
    return out
