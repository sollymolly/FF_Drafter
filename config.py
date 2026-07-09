"""
config.py — central, league-agnostic configuration for FF Drafter.

WHY THIS EXISTS:
  Every league-dependent number (replacement levels, total auction money, how many
  opponents you face) is DERIVED from LEAGUE["teams"] and the roster definition.
  Nothing downstream hardcodes "12". Switching between a 10-team and a 12-team
  league (or any size) is a one-line edit to LEAGUE["teams"].

  Pure standard library — safe to import anywhere (no pandas/numpy required).

USAGE:
  import config
  config.POSITIONS            # derived replacement levels for the default league
  config.total_money()        # teams * budget
  config.positions({**config.LEAGUE, "teams": 10})   # same math, 10-team league
"""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# LEAGUE SETTINGS — edit these to match your league.
# ---------------------------------------------------------------------------
LEAGUE = {
    "teams": 12,            # number of managers; 10, 12, 14, ... all supported
    "budget": 200,          # auction dollars per manager
    "scoring": "PPR",       # "PPR" | "HALF_PPR" | "STANDARD"
    "season": 2026,         # season being drafted for
    "roster_slots": {       # starting + bench composition per team
        "QB": 1,
        "RB": 2,
        "WR": 2,
        "TE": 1,
        "FLEX": 1,          # RB/WR/TE
        "DST": 1,
        "K": 1,
        "BN": 7,            # bench
    },
    "flex_positions": ["RB", "WR", "TE"],
}

# Positions that score fantasy points (everything we value).
SCORABLE_POSITIONS = ("QB", "RB", "WR", "TE", "DST", "K")

# ---------------------------------------------------------------------------
# REPLACEMENT-LEVEL KNOBS (also league-agnostic).
#   FLEX_SPLIT:    how the league's flex spots tend to be filled across RB/WR/TE.
#   STREAM_BUFFER: small extra depth for positions that get streamed off waivers.
# These shape replacement level but never reference a specific team count.
# ---------------------------------------------------------------------------
FLEX_SPLIT = {"RB": 0.45, "WR": 0.45, "TE": 0.10}
STREAM_BUFFER = {"QB": 2, "RB": 0, "WR": 0, "TE": 1, "DST": 1, "K": 1}

# ---------------------------------------------------------------------------
# THREAT-MODEL KNOBS (draft/threat.py) — opponent purchasing power & my premium.
#   deploy_fraction:    share of a team's THREAT MONEY they'll throw at one star.
#                       Threat money = spare-cash surplus above the room median
#                       + banked edge above the room average (each floored at 0).
#                       Raw surplus is huge for everyone early — that room-wide
#                       effect is inflation's job; only the advantage is a threat.
#   bench_bid_fraction: what managers with no open starter slot still bid on a
#                       star (bench/flex hoarding money), pre-max-bid cap.
#   scarcity_alpha:     fraction of the tier-cliff drop I'll pay to insure when a
#                       player is the last of his tier at a position I need.
#   rivalry_beta:       $ of denial premium per $ a credible bidder's banked edge
#                       sits above the room average (scaled by star size).
#   premium_cap:        hard ceiling on my total premium, as a share of the
#                       inflated price — never bid what you'd regret winning.
#   bench_fill_cost:    assumed $ per open bench slot when costing a roster out.
# ---------------------------------------------------------------------------
THREAT = {
    "deploy_fraction": 0.5,
    "bench_bid_fraction": 0.5,
    "scarcity_alpha": 0.30,
    "rivalry_beta": 0.25,
    "premium_cap": 0.15,
    "bench_fill_cost": 1,
}

# ---------------------------------------------------------------------------
# SCORING WEIGHTS — reception value follows the scoring mode automatically.
# ---------------------------------------------------------------------------
_RECEPTION_PTS = {"PPR": 1.0, "HALF_PPR": 0.5, "STANDARD": 0.0}

SCORING = {
    "passing_yd":    0.04,   # 1 pt / 25 yds
    "passing_td":    4,
    "interception": -2,
    "rushing_yd":    0.1,    # 1 pt / 10 yds
    "rushing_td":    6,
    "receiving_yd":  0.1,
    "receiving_td":  6,
    "reception":     _RECEPTION_PTS[LEAGUE["scoring"]],
    "fumble_lost":  -2,
    "two_pt":        2,
}

# ---------------------------------------------------------------------------
# FILE PATHS — resolve relative to ROOT so the project folder is portable.
# ---------------------------------------------------------------------------
PATHS = {
    "raw":       ROOT / "data" / "raw",
    "processed": ROOT / "data" / "processed",
    "board":     ROOT / "data" / "board",
    "session":   ROOT / "data" / "session.json",   # live draft snapshot
}


# ===========================================================================
# DERIVED VALUES — computed from the league config above.
# Import these instead of hardcoding team counts anywhere else.
# ===========================================================================
def _round_half_up(x: float) -> int:
    """Round to nearest int, halves up (intuitive, unlike Python's banker's round)."""
    return int(math.floor(x + 0.5))


def roster_size(league: dict = LEAGUE) -> int:
    """Total roster spots per team (starters + bench)."""
    return sum(league["roster_slots"].values())


def starting_slots(league: dict = LEAGUE) -> int:
    """Number of starting (non-bench) slots per team."""
    return sum(v for k, v in league["roster_slots"].items() if k != "BN")


def total_money(league: dict = LEAGUE) -> int:
    """Total auction dollars in the league = teams * budget."""
    return league["teams"] * league["budget"]


def num_opponents(league: dict = LEAGUE) -> int:
    """How many other managers you draft against."""
    return league["teams"] - 1


def compute_replacement_ranks(
    league: dict = LEAGUE,
    flex_split: dict = FLEX_SPLIT,
    stream_buffer: dict = STREAM_BUFFER,
) -> dict[str, int]:
    """
    Replacement level = rank of the last 'startable' player at each position,
    derived purely from team count + roster slots so it scales with league size:

        base      = teams * starters_at_position
        flex_add  = share of the league's flex spots that fall to this position
        buffer    = small streaming cushion for thin / streamed positions

    Example (12 teams): QB 14, RB 29, WR 29, TE 14, DST 13, K 13
    Example (10 teams): QB 12, RB 25, WR 25, TE 12, DST 11, K 11
    """
    teams = league["teams"]
    slots = league["roster_slots"]
    flex_total = teams * slots.get("FLEX", 0)

    ranks: dict[str, int] = {}
    for pos in SCORABLE_POSITIONS:
        base = teams * slots.get(pos, 0)
        flex_add = _round_half_up(flex_total * flex_split.get(pos, 0.0)) if pos in flex_split else 0
        ranks[pos] = int(base + flex_add + stream_buffer.get(pos, 0))
    return ranks


def positions(league: dict = LEAGUE) -> dict[str, dict]:
    """Per-position config: starters per team + derived replacement rank."""
    ranks = compute_replacement_ranks(league)
    return {
        pos: {
            "starters": league["roster_slots"].get(pos, 0),
            "replacement_rank": ranks[pos],
        }
        for pos in SCORABLE_POSITIONS
    }


# Convenience module-level snapshots for the default league.
ROSTER_SIZE = roster_size()
TOTAL_MONEY = total_money()
NUM_OPPONENTS = num_opponents()
POSITIONS = positions()


def _validate(league: dict = LEAGUE) -> None:
    assert league["teams"] >= 2, "Need at least 2 teams"
    assert league["budget"] >= roster_size(league), \
        "Budget must cover at least $1 per roster spot"
    assert league["scoring"] in _RECEPTION_PTS, \
        f"scoring must be one of {list(_RECEPTION_PTS)}"


_validate()


if __name__ == "__main__":
    # Sanity dump: run `python config.py`, then change LEAGUE["teams"] to confirm
    # everything below scales without touching any other file.
    print(f"League: {LEAGUE['teams']} teams | ${LEAGUE['budget']} | "
          f"{LEAGUE['scoring']} | season {LEAGUE['season']}")
    print(f"Roster size: {ROSTER_SIZE} "
          f"(starting {starting_slots()}, bench {LEAGUE['roster_slots']['BN']})")
    print(f"Total money: ${TOTAL_MONEY} | Opponents: {NUM_OPPONENTS} | "
          f"Reception pts: {SCORING['reception']}")
    print("Replacement ranks (derived):")
    for pos, cfg in POSITIONS.items():
        print(f"  {pos:<4} starters/team={cfg['starters']}  "
              f"replacement_rank={cfg['replacement_rank']}")
