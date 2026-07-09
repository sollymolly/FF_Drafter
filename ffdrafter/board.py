"""
board.py — the one entry point for building a valuation board.

Both the CLI script (scripts/build_board.py) and the live app call THIS module,
so board construction is identical everywhere; only the knobs differ:

  - The CLI is the *data* path: it may hit the network (--refresh) and train
    projections when the cache is empty. Run it once before draft day.
  - The app is the *math* path: it reuses the cached inputs and re-runs only the
    league-dependent math — dollar scaling, VOR replacement levels, blend, tiers.

WHY THIS SPLIT WORKS:
  Team count (and budget) only affect that final, sub-second stage. The expensive
  stages — the ESPN market pull, projection training, news sentiment — are league-
  size independent and cached under data/. So the app can rebuild the board for
  ANY league size on the fly; changing team count never requires the CLI.
"""

from __future__ import annotations

import json

import config
from ffdrafter import store
from ffdrafter.data import market
from ffdrafter.utils import get_logger
from ffdrafter.valuation import auction

logger = get_logger(__name__)


def resolve_model_weight(override: float | None = None):
    """Explicit override wins; else learned per-position weights; else flat 0.5."""
    if override is not None:
        return override, f"model_weight={override:g}"
    wpath = config.PATHS["processed"] / "blend_weights.json"
    if wpath.exists():
        payload = json.loads(wpath.read_text())
        weights = payload["weights"]
        logger.info("Learned blend weights from %s (backtest seasons %s)",
                    wpath.name, payload.get("seasons"))
        return weights, "learned " + " ".join(f"{p}:{w:g}" for p, w in weights.items())
    logger.info("No blend_weights.json - flat 0.5 blend. "
                "Run `python -m ffdrafter.model.blend` to learn weights.")
    return 0.5, "model_weight=0.5 (default)"


def _load_projections(league: dict, refresh: bool, train_if_missing: bool):
    """Cached projections; optionally (re)train — the CLI-only slow path."""
    cache = config.PATHS["processed"] / f"projections_{league['season']}.parquet"
    proj = None if (refresh and train_if_missing) else store.load_df(cache)
    if proj is None and train_if_missing:
        from ffdrafter.model import project
        proj = project.build_projections(force_refresh=refresh)
        project.save_projections(proj)
    return proj


def build_board(league: dict | None = None, *, source: str = "auto",
                refresh: bool = False, train_if_missing: bool = False,
                narrative: bool = True, model_weight=None,
                rookie_cards: bool = False):
    """
    Build a valuation board for `league` and return (board_df, info).

    source:  "baseline" = ESPN consensus AAV only;
             "model"    = our projections blended with market (projections required);
             "auto"     = model if projections are cached, else fall back to
                          baseline — what the live app wants.
    info:    {"name": "baseline" | "model", "weight_label": str | None}
    """
    league = league or config.LEAGUE
    mkt = market.pull_espn(force_refresh=refresh)
    baseline = auction.build_baseline_board(mkt, league)
    if source == "baseline":
        return baseline, {"name": "baseline", "weight_label": None}

    proj = _load_projections(league, refresh, train_if_missing)
    if proj is None:
        if source == "model":
            raise RuntimeError(
                "No cached projections and training is disabled — run "
                "`python scripts/build_board.py --source model` once.")
        logger.info("No cached projections — serving the baseline board.")
        return baseline, {"name": "baseline", "weight_label": None}

    nudges = None
    if narrative:
        from ffdrafter.model import narrative as narrative_mod
        nudges = narrative_mod.fetch_nudges(force_refresh=refresh)

    weight, weight_label = resolve_model_weight(model_weight)
    board = auction.build_model_board(proj, baseline, league,
                                      model_weight=weight, narrative_df=nudges)
    if rookie_cards:
        from ffdrafter.model import rookie_card
        rookie_card.build_rookie_cards(proj)
    return board, {"name": "model", "weight_label": weight_label}
