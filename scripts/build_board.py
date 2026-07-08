"""
scripts/build_board.py — build a valuation board and save it.

    python scripts/build_board.py                       # baseline (ESPN consensus AAV)
    python scripts/build_board.py --source model        # our projections blended w/ market
    python scripts/build_board.py --source model --model-weight 0.7
    python scripts/build_board.py --refresh              # re-pull ESPN + retrain

The baseline board re-scales ESPN AAV onto your league economics. The model board
runs our projection engine (veteran + rookie), converts to VOR dollars, and blends
toward the market for safety. Both share one schema, so the live app consumes either.
"""

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/build_board.py` from anywhere by putting the project
# root on the path (a script's sys.path[0] is its own dir, not the project root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles default to cp1252 and choke on non-ASCII; emit UTF-8 instead.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

import config
from ffdrafter import store
from ffdrafter.data import market
from ffdrafter.valuation import auction
from ffdrafter.utils import get_logger

logger = get_logger("build_board")


def _load_or_build_projections(refresh: bool):
    from ffdrafter.model import project
    cache = config.PATHS["processed"] / f"projections_{config.LEAGUE['season']}.parquet"
    proj = None if refresh else store.load_df(cache)
    if proj is None:
        proj = project.build_projections(force_refresh=refresh)
        project.save_projections(proj)
    return proj


def _resolve_model_weight(override: float | None):
    """CLI override wins; else learned per-position weights; else flat 0.5."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the valuation board.")
    ap.add_argument("--source", choices=["baseline", "model"], default="baseline",
                    help="baseline = ESPN AAV; model = our projections blended with market")
    ap.add_argument("--refresh", action="store_true", help="re-pull ESPN + retrain")
    ap.add_argument("--model-weight", type=float, default=None,
                    help="flat model-vs-market weight override (0=market, 1=model); "
                         "default: learned per-position weights from blend_weights.json, else 0.5")
    ap.add_argument("--no-narrative", action="store_true",
                    help="skip the bounded news-sentiment nudge")
    args = ap.parse_args()

    mkt = market.pull_espn(force_refresh=args.refresh)
    baseline = auction.build_baseline_board(mkt)

    if args.source == "model":
        proj = _load_or_build_projections(args.refresh)
        nudges = None
        if not args.no_narrative:
            from ffdrafter.model import narrative
            nudges = narrative.fetch_nudges(force_refresh=args.refresh)
        weight, weight_label = _resolve_model_weight(args.model_weight)
        board = auction.build_model_board(proj, baseline, model_weight=weight,
                                          narrative_df=nudges)
        from ffdrafter.model import rookie_card
        rookie_card.build_rookie_cards(proj)
        name = "model"
    else:
        board = baseline
        name = "baseline"
    store.save_board(board, name)

    lg = config.LEAGUE
    label = f"{name} board" + (f" ({weight_label})" if name == "model" else "")
    print(f"\n{label} - {lg['season']} | {lg['teams']}-team {lg['scoring']} | ${lg['budget']}/team")
    print(f"Players: {len(board)} | total ${int(board['value'].sum())} ~= league money "
          f"${config.total_money()}")

    priced = board[board["value"] >= 2]
    counts = priced["position"].value_counts().to_dict()
    print("Players valued $2+ by position:",
          " | ".join(f"{p}:{counts.get(p, 0)}" for p in ("QB", "RB", "WR", "TE", "DST", "K")))

    cols = ["name", "position", "team", "value", "tier", "aav", "adp"]
    if "projected_pts" in board.columns:
        cols = ["name", "position", "team", "value", "tier", "projected_pts", "aav"]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print("\nTop 30 by value:")
        print(board[cols].head(30).round(1).to_string(index=False))

    print(f"\nSaved -> data/board/{name}.parquet")


if __name__ == "__main__":
    main()
