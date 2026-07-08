"""
model/backtest.py — honest out-of-sample check: model vs naive vs market.

For a held-out season T, three forecasters order the draft-relevant pool (the
market's preseason top-N skill players):

  model  — veteran GBMs trained only on transitions finishing by T-1, plus
           rookie GBMs trained only on draft classes through T-1;
  naive  — reuse last season's points (rookies get 0);
  market — the preseason FantasyPros PPR consensus itself (a rank, no points).

All three are scored against what actually happened in T. Pool players who never
played count as 0 points — a drafted bust is a real cost, not a missing value.
Rank correlation on the pool is the common metric (ECR has no point scale);
points-MAE is reported for model vs naive as a secondary check. Beating naive
says the model adds signal; the market is the opponent that matters.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

import config
from ffdrafter.data import market_history, nfl
from ffdrafter.features import rookie, veteran
from ffdrafter.model import project
from ffdrafter.utils import get_logger

logger = get_logger(__name__)

POOL_SIZE = 200          # market's preseason top-N skill players = draft-relevant pool
TARGET_SEASONS = (2021, 2022, 2023, 2024, 2025)


def _veteran_preds(season_df, ids, target_season: int) -> pd.DataFrame:
    """Veteran model predictions for T, trained only on transitions <= T-1."""
    train = veteran.build_training_table(season_df, ids)
    train = train[train["season"] <= target_season - 2]
    models = project._train_by_position(train, veteran.FEATURES, "target_next",
                                        min_rows=40, min_leaf=15)
    feat = veteran.build_projection_features(season_df, ids, base_season=target_season - 1)
    pred = project._project_by_position(models, feat, veteran.FEATURES)
    if pred.empty:
        return pd.DataFrame(columns=["gsis_id", "projected_pts"])
    return pred[["player_id", "projected_pts"]].rename(columns={"player_id": "gsis_id"})


def _rookie_preds(season_df, ids, target_season: int) -> pd.DataFrame:
    """Rookie model predictions for T's draft class, trained on classes <= T-1."""
    rtrain = rookie.build_rookie_training(season_df, ids)
    rtrain = rtrain[rtrain["season"] <= target_season - 1]
    models = project._train_by_position(rtrain, rookie.ROOKIE_FEATURES, "target_rookie",
                                        min_rows=25, min_leaf=8)
    if not models:
        return pd.DataFrame(columns=["gsis_id", "projected_pts"])
    rfeat = rookie.build_rookie_features(nfl.draft_class(target_season), ids)
    pred = project._project_by_position(models, rfeat, rookie.ROOKIE_FEATURES)
    if pred.empty:
        return pd.DataFrame(columns=["gsis_id", "projected_pts"])
    return pred.dropna(subset=["gsis_id"])[["gsis_id", "projected_pts"]]


def backtest_three_way(target_season: int, history_start: int = 2018,
                       pool_size: int = POOL_SIZE):
    """
    One held-out season. Returns (summary_df, detail_df):
      summary_df — per position + ALL: n, model/naive/market rank corr, model/naive MAE.
      detail_df  — one row per pool player: market ranks + model/naive/actual points
                   (the raw material blend.py learns weights from).
    """
    season_df = nfl.season_stats(range(history_start, config.LEAGUE["season"]))
    ids = nfl.player_ids()

    mkt = market_history.preseason_market(target_season, ids)
    pool = mkt.head(pool_size).copy()
    unmatched = int(pool["gsis_id"].isna().sum())
    if unmatched:
        logger.info("  %d: dropping %d/%d pool players with no gsis match",
                    target_season, unmatched, len(pool))
    pool = pool.dropna(subset=["gsis_id"])

    model = pd.concat([_veteran_preds(season_df, ids, target_season),
                       _rookie_preds(season_df, ids, target_season)],
                      ignore_index=True).drop_duplicates("gsis_id")

    actual = (season_df.loc[season_df["season"] == target_season,
                            ["player_id", "fantasy_points_league"]]
              .rename(columns={"player_id": "gsis_id", "fantasy_points_league": "actual"}))
    naive = (season_df.loc[season_df["season"] == target_season - 1,
                           ["player_id", "fantasy_points_league"]]
             .rename(columns={"player_id": "gsis_id", "fantasy_points_league": "naive"}))

    d = (pool.merge(model.rename(columns={"projected_pts": "model"}), on="gsis_id", how="left")
             .merge(naive, on="gsis_id", how="left")
             .merge(actual, on="gsis_id", how="left"))
    # No projection / didn't play => 0 points. Busts cost real auction dollars.
    for c in ("model", "naive", "actual"):
        d[c] = d[c].fillna(0.0)
    d.insert(0, "season", target_season)

    rows = []
    for pos in list(project.SKILL) + ["ALL"]:
        g = d if pos == "ALL" else d[d["position"] == pos]
        if len(g) < 8:
            continue
        rows.append({
            "season": target_season, "pos": pos, "n": len(g),
            "model_rho": spearmanr(g["model"], g["actual"]).correlation,
            "naive_rho": spearmanr(g["naive"], g["actual"]).correlation,
            "market_rho": spearmanr(-g["market_ecr"], g["actual"]).correlation,
            "model_MAE": (g["model"] - g["actual"]).abs().mean(),
            "naive_MAE": (g["naive"] - g["actual"]).abs().mean(),
        })
    return pd.DataFrame(rows), d


def run(target_seasons=TARGET_SEASONS, pool_size: int = POOL_SIZE):
    """Three-way backtest across seasons. Returns (summary, detail)."""
    summaries, details = [], []
    for t in target_seasons:
        logger.info("=== backtest %d ===", t)
        s, d = backtest_three_way(t, pool_size=pool_size)
        summaries.append(s)
        details.append(d)
    summary = pd.concat(summaries, ignore_index=True)
    detail = pd.concat(details, ignore_index=True)
    num_cols = [c for c in summary.columns if c.endswith(("_rho", "_MAE"))]
    summary[num_cols] = summary[num_cols].astype(float).round(3)
    return summary, detail
