"""
scripts/build_board.py — CLI wrapper: refresh data, build a board, save + print it.

    python scripts/build_board.py                       # baseline (ESPN consensus AAV)
    python scripts/build_board.py --source model        # our projections blended w/ market
    python scripts/build_board.py --source model --model-weight 0.7
    python scripts/build_board.py --refresh             # re-pull ESPN + retrain
    python scripts/build_board.py --teams 10            # inspect a 10-team board

All construction logic lives in ffdrafter/board.py, shared with the live app. The
app rebuilds the board in-process for whatever league size you pick at setup, so
this script exists to REFRESH DATA (network pull, model training) — never to
change league size.
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
from ffdrafter.board import build_board


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
    ap.add_argument("--teams", type=int, default=None,
                    help="override number of teams (default: config.LEAGUE['teams'])")
    ap.add_argument("--budget", type=int, default=None,
                    help="override per-team auction budget (default: config.LEAGUE['budget'])")
    args = ap.parse_args()

    lg = dict(config.LEAGUE)
    if args.teams is not None:
        lg["teams"] = args.teams
    if args.budget is not None:
        lg["budget"] = args.budget

    board, info = build_board(
        lg, source=args.source, refresh=args.refresh, train_if_missing=True,
        narrative=not args.no_narrative, model_weight=args.model_weight,
        rookie_cards=(args.source == "model"),
    )
    name = info["name"]
    store.save_board(board, name)

    label = f"{name} board" + (f" ({info['weight_label']})" if info["weight_label"] else "")
    print(f"\n{label} - {lg['season']} | {lg['teams']}-team {lg['scoring']} | ${lg['budget']}/team")
    print(f"Players: {len(board)} | total ${int(board['value'].sum())} ~= league money "
          f"${config.total_money(lg)}")

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
