"""
data/market_history.py — historical preseason market consensus (FantasyPros ECR).

The DynastyProcess archive of FantasyPros Expert Consensus Rankings tells us what
the market believed BEFORE each past season — the yardstick the backtest has to
beat. Beating "reuse last year's points" is table stakes; the market is the real
opponent. We use the PPR overall cheatsheet (ecr_type "ro", the ppr-cheatsheets
page) and take the last scrape on or before ~Sep 5 as "the market on draft day."

The full archive (~1.8M rows, 2020 onward) is downloaded once via nflreadpy and
cached at data/raw/fpecr_all.parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import PATHS
from ffdrafter import store
from ffdrafter.utils import get_logger, normalize_name

logger = get_logger(__name__)

ARCHIVE = PATHS["raw"] / "fpecr_all.parquet"
_COLS = ["fp_page", "ecr_type", "player", "id", "pos", "team", "ecr", "mergename", "scrape_date"]
SKILL = ("QB", "RB", "WR", "TE")

MARKET_COLUMNS = ["fp_id", "name", "name_key", "position", "team", "market_ecr",
                  "mkt_overall_rank", "mkt_pos_rank", "gsis_id", "espn_id", "scrape_date"]


def ecr_archive(force_refresh: bool = False) -> pd.DataFrame:
    """Full FantasyPros ECR history (DynastyProcess archive), cached to parquet."""
    if ARCHIVE.exists() and not force_refresh:
        return pd.read_parquet(ARCHIVE, columns=_COLS)
    import nflreadpy as nfl

    logger.info("Downloading FantasyPros ECR archive (one-time, ~100MB)...")
    df = nfl.load_ff_rankings("all")
    df = df.to_pandas() if hasattr(df, "to_pandas") else df
    store.save_df(df, ARCHIVE)
    return df[_COLS]


def preseason_market(season: int, ids: pd.DataFrame, cutoff_day: str = "09-05") -> pd.DataFrame:
    """
    The market's preseason view for `season`: one row per skill player from the
    last PPR overall-ECR scrape before kickoff, joined to gsis/espn ids.

    market_ecr is the raw consensus rank value (lower = better); mkt_overall_rank
    and mkt_pos_rank are dense 1..N ranks within the skill pool / position.
    """
    arc = ecr_archive()
    arc = arc[(arc["ecr_type"] == "ro")
              & arc["fp_page"].str.contains("ppr-cheatsheets", na=False)].copy()
    arc["scrape_date"] = pd.to_datetime(arc["scrape_date"])

    lo = pd.Timestamp(f"{season}-06-01")
    hi = pd.Timestamp(f"{season}-{cutoff_day}")
    win = arc[(arc["scrape_date"] >= lo) & (arc["scrape_date"] <= hi)]
    if win.empty:
        raise ValueError(f"No preseason PPR ECR snapshot found for {season}")
    snap = win[win["scrape_date"] == win["scrape_date"].max()].copy()

    snap = snap[snap["pos"].isin(SKILL)].copy()
    snap["fp_id"] = pd.to_numeric(snap["id"], errors="coerce").astype("Int64")
    snap["name_key"] = snap["mergename"].fillna(snap["player"]).map(normalize_name)
    snap = snap.drop_duplicates(subset=["name_key", "pos"])

    # Join to the nflverse crosswalk: fantasypros_id first, name_key fallback.
    x = ids.dropna(subset=["fantasypros_id"]).copy()
    x["fp_id"] = pd.to_numeric(x["fantasypros_id"], errors="coerce").astype("Int64")
    x = x.dropna(subset=["fp_id"]).drop_duplicates("fp_id").set_index("fp_id")
    snap = snap.join(x[["gsis_id", "espn_id"]], on="fp_id")

    miss = snap["gsis_id"].isna()
    if miss.any():
        y = (ids.dropna(subset=["gsis_id"]).drop_duplicates("name_key")
             .set_index("name_key")[["gsis_id", "espn_id"]])
        snap.loc[miss, "gsis_id"] = snap.loc[miss, "name_key"].map(y["gsis_id"])
        snap.loc[miss, "espn_id"] = snap.loc[miss, "name_key"].map(y["espn_id"])

    snap = snap.rename(columns={"player": "name", "pos": "position", "ecr": "market_ecr"})
    snap = snap.sort_values("market_ecr").reset_index(drop=True)
    snap["mkt_overall_rank"] = np.arange(1, len(snap) + 1)
    snap["mkt_pos_rank"] = snap.groupby("position")["market_ecr"].rank(method="first").astype(int)

    matched = float(snap["gsis_id"].notna().mean())
    logger.info("Market %d: scrape %s, %d skill players, %.0f%% matched to gsis",
                season, snap["scrape_date"].max().date(), len(snap), matched * 100)
    return snap[MARKET_COLUMNS]
