"""
utils.py — small shared helpers used across the project.

Standard-library only (no pandas needed to import this module); the vectorized
name helper works on a pandas Series when you have one.
"""

from __future__ import annotations

import logging
import re

# Name suffixes stripped so "Marvin Harrison Jr." matches "Marvin Harrison"
# across data sources that disagree on whether to include the suffix.
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent timestamped format across modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def normalize_name(name: str) -> str:
    """
    Canonical lowercase key for fuzzy-joining players across data sources.

      "D.K. Metcalf"        -> "dk metcalf"
      "Ja'Marr Chase"       -> "jamarr chase"
      "Marvin Harrison Jr." -> "marvin harrison"
      "Kenneth Walker III"  -> "kenneth walker"
    """
    if not isinstance(name, str):
        return ""
    s = name.lower().replace(".", "").replace("'", "").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    parts = [p for p in s.split(" ") if p not in _SUFFIXES]
    return " ".join(parts) if parts else s


def normalize_name_series(series):
    """Vectorized normalize_name for a pandas Series."""
    return series.fillna("").map(normalize_name)
