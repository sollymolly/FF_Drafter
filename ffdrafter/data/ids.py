"""
data/ids.py — cross-source player identity reconciliation.

Player IDs differ across nflverse / Sleeper / FantasyPros / ESPN, so we join on a
normalized name key (utils.normalize_name) and, where available, nfl_data_py's
official ID crosswalk (gsis / sleeper / espn / pfr ids).

nfl_data_py is imported lazily so this module imports cleanly before deps exist.
"""

from __future__ import annotations

from ffdrafter.utils import get_logger, normalize_name_series

logger = get_logger(__name__)


def attach_name_key(df, name_col: str = "name"):
    """Return a copy of df with a normalized 'name_key' column for fuzzy joins."""
    df = df.copy()
    df["name_key"] = normalize_name_series(df[name_col])
    return df


def load_id_map():
    """
    Return nfl_data_py's cross-source player ID crosswalk with a normalized
    'name_key' column, or None if nfl_data_py is unavailable.

    Columns typically include: name, position, gsis_id, sleeper_id, espn_id,
    pfr_id, and more — the join keys we use to reconcile sources later.
    """
    try:
        import nfl_data_py as nfl
    except Exception as e:  # not installed yet, etc.
        logger.warning("nfl_data_py unavailable (%s) — name-key joins only", e)
        return None
    try:
        ids = nfl.import_ids()
    except Exception as e:
        logger.warning("Could not load nfl id map: %s", e)
        return None

    if "name" in ids.columns:
        ids = ids.copy()
        ids["name_key"] = normalize_name_series(ids["name"])
    return ids


def merge_on_name(left, right, right_cols=None, how: str = "left"):
    """
    Join two player frames on the normalized name key, optionally limiting which
    columns come from `right`. Adds 'name_key' to either side if missing.
    """
    l = left if "name_key" in left.columns else attach_name_key(left)
    r = right if "name_key" in right.columns else attach_name_key(right)
    if right_cols:
        keep = ["name_key"] + [c for c in right_cols if c in r.columns]
        r = r[keep]
    return l.merge(r, on="name_key", how=how)
