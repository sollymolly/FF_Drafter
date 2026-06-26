"""
scripts/build_board.py — build the current valuation board and save it.

Phase 2: a baseline board from ESPN consensus auction values (AAV), re-scaled to
your league's economics.

    python scripts/build_board.py             # use cached ESPN data if present
    python scripts/build_board.py --refresh   # re-pull from ESPN
"""

import argparse
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the valuation board.")
    ap.add_argument("--refresh", action="store_true", help="re-pull market data from ESPN")
    args = ap.parse_args()

    mkt = market.pull_espn(force_refresh=args.refresh)
    board = auction.build_baseline_board(mkt)
    store.save_board(board, "baseline")

    lg = config.LEAGUE
    print(f"\nBaseline board - {lg['season']} | {lg['teams']}-team {lg['scoring']} "
          f"| ${lg['budget']}/team")
    print(f"Players: {len(board)} | board $ over draftable pool ~= league money "
          f"${config.total_money()}")

    # Per-position counts of players valued at $2+ (a quick scarcity sanity check).
    priced = board[board["value"] >= 2]
    counts = priced["position"].value_counts().to_dict()
    print("Players valued $2+ by position:",
          " | ".join(f"{p}:{counts.get(p, 0)}" for p in ("QB", "RB", "WR", "TE", "DST", "K")))

    cols = ["name", "position", "team", "value", "tier", "aav", "adp"]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print("\nTop 30 by value:")
        print(board[cols].head(30).to_string(index=False))

    print("\nSaved -> data/board/baseline.parquet")


if __name__ == "__main__":
    main()
