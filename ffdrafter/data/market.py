"""
data/market.py — consensus auction values (AAV) + ADP from free sources.

PRIMARY SOURCE: ESPN's public fantasy "players" endpoint. Because the user drafts
on ESPN, ESPN's published auction-value-average (AAV) and ADP are the most relevant
free baseline. The raw payload is large (~all NFL players), so we parse the fields
we need and cache them to parquet; re-pull only with force_refresh.

NOTE: this is ESPN's *unofficial* read API. The shape was verified live against the
2026 season; if ESPN changes it, this module is the single place to adjust.
"""

from __future__ import annotations

import json

import requests

import config
from config import PATHS, LEAGUE
from ffdrafter import store
from ffdrafter.utils import get_logger, normalize_name_series

logger = get_logger(__name__)

ESPN_HOST = "https://lm-api-reads.fantasy.espn.com"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"),
    "Accept": "application/json",
}

# ESPN defaultPositionId -> our position label (fantasy skill positions only).
ESPN_POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DST"}

# ESPN proTeamId -> NFL abbreviation (0 = free agent / unsigned).
ESPN_TEAM = {
    0: "FA", 1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN",
    8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LAR",
    15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI",
    22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WSH",
    29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}

# ESPN publishes rankings per scoring type; map our league scoring to the closest.
_RANK_TYPE = {"PPR": "PPR", "HALF_PPR": "PPR", "STANDARD": "STANDARD"}


def _fetch_espn_raw(year: int) -> list:
    """Hit the ESPN players endpoint and return the raw list of player dicts."""
    # filterSlotIds keeps QB/RB/WR/TE/K/DST slots; limit/sort trim the payload when
    # honored (ESPN sometimes ignores the filter and returns everything — we also
    # filter client-side, so either way the result is correct).
    filt = {
        "players": {
            "limit": 1500,
            "filterSlotIds": {"value": [0, 2, 4, 6, 17, 16]},
            "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
        }
    }
    url = f"{ESPN_HOST}/apis/v3/games/ffl/seasons/{year}/players?view=kona_player_info"
    resp = requests.get(
        url, headers={**_HEADERS, "x-fantasy-filter": json.dumps(filt)}, timeout=60
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected ESPN response type: {type(data).__name__}")
    return data


def pull_espn(year: int | None = None, force_refresh: bool = False):
    """
    Return a DataFrame of ESPN market values, one row per fantasy-relevant player:
        espn_id, name, name_key, position, team, aav, adp, espn_rank, published, source

    'aav' is ESPN's average auction value (consensus winning bid); 'adp' is average
    draft position. Cached to data/raw/espn_market_<year>.parquet.
    """
    import pandas as pd

    year = year or LEAGUE["season"]
    rank_type = _RANK_TYPE.get(LEAGUE["scoring"], "PPR")
    cache = PATHS["raw"] / f"espn_market_{year}.parquet"

    if not force_refresh:
        cached = store.load_df(cache)
        if cached is not None:
            logger.info("Loaded cached ESPN market: %d players", len(cached))
            return cached

    logger.info("Fetching ESPN player universe for %s (rank type %s)...", year, rank_type)
    raw = _fetch_espn_raw(year)

    rows = []
    for entry in raw:
        try:
            p = entry.get("player", entry)
            pos = ESPN_POS.get(p.get("defaultPositionId"))
            if pos is None:
                continue  # IDP / non-skill position
            own = p.get("ownership") or {}
            ranks = (p.get("draftRanksByRankType") or {}).get(rank_type) or {}
            aav = own.get("auctionValueAverage")
            adp = own.get("averageDraftPosition")
            rows.append({
                "espn_id": p.get("id"),
                "name": p.get("fullName"),
                "position": pos,
                "team": ESPN_TEAM.get(p.get("proTeamId"), "FA"),
                "aav": float(aav) if aav is not None else 0.0,
                "adp": float(adp) if adp is not None else None,
                "espn_rank": ranks.get("rank"),
                "published": bool(ranks.get("published", False)),
                "source": "espn",
            })
        except Exception as e:  # never let one bad row kill the pull
            logger.debug("skipping malformed entry: %s", e)

    df = pd.DataFrame(rows)
    df["name_key"] = normalize_name_series(df["name"])

    # Keep the draftable pool: ESPN-ranked ("published") players, or anyone with a
    # positive AAV. This drops the long tail of camp bodies / practice-squad noise.
    keep = df["published"] | (df["aav"] > 0)
    df = df[keep].drop_duplicates("espn_id").reset_index(drop=True)

    store.save_df(df, cache)
    logger.info("ESPN market: %d players kept (%d with AAV>0)",
                len(df), int((df["aav"] > 0).sum()))
    return df
