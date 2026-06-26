"""
model/project.py — train projection models and build the season's projections table.

Veteran track: per-position quantile gradient boosting, season-N features -> N+1 points.
Rookie track:  per-position quantile gradient boosting on draft capital.

Quantiles give an honest floor/median/ceiling (0.20 / 0.50 / 0.80). The output is one
table (QB/RB/WR/TE) with projected_pts + floor/ceiling, joined to espn_id so it drops
straight onto the auction board (valuation/auction.build_model_board).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import config
from ffdrafter.data import nfl
from ffdrafter.features import rookie, veteran
from ffdrafter.utils import get_logger

logger = get_logger(__name__)

SKILL = ("QB", "RB", "WR", "TE")
BANDS = {"floor": 0.20, "ceiling": 0.80}


def _fit_models(X, y, min_leaf: int = 15):
    """Central projection via mean (squared_error, less top-compression on skewed
    scoring) + floor/ceiling via quantile regression."""
    common = dict(learning_rate=0.06, max_iter=300, max_depth=3,
                  min_samples_leaf=min_leaf, l2_regularization=1.0, random_state=42)
    models = {"projected_pts": HistGradientBoostingRegressor(loss="squared_error", **common).fit(X, y)}
    for name, q in BANDS.items():
        models[name] = HistGradientBoostingRegressor(loss="quantile", quantile=q, **common).fit(X, y)
    return models


def _train_by_position(df, feature_cols, target_col, min_rows, min_leaf):
    out = {}
    for pos in SKILL:
        d = df[df["position"] == pos]
        if len(d) < min_rows:
            continue
        out[pos] = _fit_models(d[feature_cols], d[target_col], min_leaf=min_leaf)
        logger.info("  %s: trained on %d rows", pos, len(d))
    return out


def _project_by_position(models_by_pos, feat_df, feature_cols):
    rows = []
    for pos, models in models_by_pos.items():
        d = feat_df[feat_df["position"] == pos].copy()
        if d.empty:
            continue
        for name, m in models.items():
            d[name] = np.clip(m.predict(d[feature_cols]), 0, None)
        # keep quantiles ordered: floor <= projected <= ceiling
        d["floor"] = np.minimum(d["floor"], d["projected_pts"])
        d["ceiling"] = np.maximum(d["ceiling"], d["projected_pts"])
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _combine(vproj, rproj, ids):
    idmap = ids.dropna(subset=["gsis_id"]).drop_duplicates("gsis_id").set_index("gsis_id")
    base_cols = ["gsis_id", "name", "name_key", "position", "team", "espn_id",
                 "projected_pts", "floor", "ceiling", "is_rookie", "age", "draft_ovr"]

    v = vproj.copy()
    v["gsis_id"] = v["player_id"]
    v["espn_id"] = v["gsis_id"].map(idmap["espn_id"])
    v["team"] = v["gsis_id"].map(idmap["team"])
    v["draft_ovr"] = v["gsis_id"].map(idmap["draft_ovr"])

    r = rproj.copy()
    r["espn_id"] = r["gsis_id"].map(idmap["espn_id"])
    if "team" not in r.columns or r["team"].isna().all():
        r["team"] = r["gsis_id"].map(idmap["team"])

    for df in (v, r):
        for c in base_cols:
            if c not in df.columns:
                df[c] = np.nan

    out = pd.concat([v[base_cols], r[base_cols]], ignore_index=True)
    # espn_id as nullable int for clean joins to the board
    out["espn_id"] = pd.to_numeric(out["espn_id"], errors="coerce").astype("Int64")
    return out.sort_values("projected_pts", ascending=False).reset_index(drop=True)


def build_projections(target_season: int | None = None, history_start: int = 2018,
                      force_refresh: bool = False):
    """Train on history and return the projections table for target_season (default = league season)."""
    target = target_season or config.LEAGUE["season"]
    base = target - 1
    seasons = range(history_start, target)

    season_df = nfl.season_stats(seasons, force_refresh=force_refresh)
    ids = nfl.player_ids(force_refresh=force_refresh)
    dclass = nfl.draft_class(target, force_refresh=force_refresh)

    logger.info("Training veteran models...")
    vtrain = veteran.build_training_table(season_df, ids)
    vmodels = _train_by_position(vtrain, veteran.FEATURES, "target_next", min_rows=40, min_leaf=15)
    vfeat = veteran.build_projection_features(season_df, ids, base_season=base)
    vproj = _project_by_position(vmodels, vfeat, veteran.FEATURES)
    vproj["is_rookie"] = False

    logger.info("Training rookie models...")
    rtrain = rookie.build_rookie_training(season_df, ids)
    rmodels = _train_by_position(rtrain, rookie.ROOKIE_FEATURES, "target_rookie", min_rows=25, min_leaf=8)
    rfeat = rookie.build_rookie_features(dclass, ids)
    rproj = _project_by_position(rmodels, rfeat, rookie.ROOKIE_FEATURES)
    rproj["is_rookie"] = True

    proj = _combine(vproj, rproj, ids)
    logger.info("Projections: %d players (%d veterans, %d rookies) for %d",
                len(proj), int((~proj["is_rookie"]).sum()), int(proj["is_rookie"].sum()), target)
    return proj


def save_projections(proj, target_season: int | None = None):
    from ffdrafter import store
    target = target_season or config.LEAGUE["season"]
    return store.save_df(proj, config.PATHS["processed"] / f"projections_{target}.parquet")
