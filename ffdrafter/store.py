"""
store.py — persistence layer.

  - Valuation boards & feature tables: Parquet (fast, columnar) under data/.
  - Live draft session: a JSON snapshot written atomically after every change,
    so a browser refresh or crash never loses an in-progress draft.

pandas/pyarrow are imported lazily inside functions, so importing this module is
cheap and does not require the heavy deps to be installed yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from config import PATHS
from ffdrafter.utils import get_logger

logger = get_logger(__name__)


def _ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# DataFrame persistence (boards, feature tables)
# ---------------------------------------------------------------------------
def save_board(df, name: str) -> Path:
    """Save a valuation board (or any DataFrame) to data/board/<name>.parquet."""
    path = PATHS["board"] / f"{name}.parquet"
    _ensure_parent(path)
    df.to_parquet(path, index=False)
    logger.info("Saved %d rows -> %s", len(df), path)
    return path


def load_board(name: str):
    """Load a board by name, or None if it does not exist yet."""
    import pandas as pd

    path = PATHS["board"] / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_df(df, path) -> Path:
    """Save a DataFrame to .parquet or .csv (inferred from the suffix)."""
    p = Path(path)
    _ensure_parent(p)
    if p.suffix == ".parquet":
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)
    logger.info("Saved %d rows -> %s", len(df), p)
    return p


def load_df(path, **kwargs):
    """Load a DataFrame from .parquet or .csv, or None if the file is missing."""
    import pandas as pd

    p = Path(path)
    if not p.exists():
        return None
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, **kwargs)


# ---------------------------------------------------------------------------
# Live draft session snapshot (crash-safe)
# ---------------------------------------------------------------------------
def save_session(state: dict, path: Optional[Path] = None) -> Path:
    """Persist the live draft state dict to JSON (atomic write via temp + replace)."""
    path = Path(path or PATHS["session"])
    _ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(path)
    return path


def load_session(path: Optional[Path] = None):
    """Load the live draft state dict, or None if no session is saved."""
    path = Path(path or PATHS["session"])
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_session(path: Optional[Path] = None) -> None:
    """Delete the saved session (start a fresh draft)."""
    path = Path(path or PATHS["session"])
    if path.exists():
        path.unlink()
        logger.info("Cleared session -> %s", path)
