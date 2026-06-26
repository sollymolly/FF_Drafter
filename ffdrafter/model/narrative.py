"""
model/narrative.py — bounded news-sentiment nudge ("the morale piece").

Pulls recent NFL headlines from ESPN's free news feed, scores each with VADER, and
turns them into a CAPPED per-player multiplier (default +/-10%) with the triggering
headline attached as the reason. Players not in the news get a neutral 1.0.

This is deliberately a *nudge*, not a driver: the hard model signal sets the number;
sentiment only tilts it within a hard cap, and every tilt shows its headline so you
can judge it. News sentiment is the noisiest input we have, hence the tight bound.
"""

from __future__ import annotations

import requests

from config import PATHS
from ffdrafter import store
from ffdrafter.utils import get_logger, normalize_name

logger = get_logger(__name__)

ESPN_NEWS = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news?limit=50"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

NUDGE_COLUMNS = ["espn_id", "name", "name_key", "narrative_mult", "narrative_reason", "n_articles"]


def _fetch_articles():
    r = requests.get(ESPN_NEWS, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json().get("articles", [])


def fetch_nudges(cap: float = 0.10, scale: float = 0.15, max_athletes: int = 4,
                 force_refresh: bool = False):
    """
    Return a per-player nudge table: espn_id, name, name_key, narrative_mult (the
    capped multiplier), narrative_reason (headline + signed %), n_articles.
    Only players whose strongest recent headline meaningfully moves the needle appear.

    Articles that mention more than `max_athletes` players are treated as roundups
    (rankings, "buzz" columns) where the sentiment is about the article, not any one
    player, and are skipped — this keeps the signal player-specific.
    """
    import pandas as pd
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    cache = PATHS["raw"] / "news_nudges.parquet"
    if not force_refresh:
        cached = store.load_df(cache)
        if cached is not None:
            logger.info("Cached news nudges: %d players", len(cached))
            return cached

    try:
        articles = _fetch_articles()
    except Exception as e:
        logger.warning("News fetch failed (%s) — no nudges applied.", e)
        return pd.DataFrame(columns=NUDGE_COLUMNS)

    sid = SentimentIntensityAnalyzer()
    # Per athlete, keep the single strongest-signal article (largest |compound|).
    best, counts = {}, {}
    for art in articles:
        # athleteId sits at the category level; the athlete sub-object holds the name.
        athletes = []
        for c in art.get("categories", []):
            ath = c.get("athlete")
            if not ath:
                continue
            aid = ath.get("athleteId") or c.get("athleteId")
            if aid is not None:
                athletes.append((aid, ath.get("description")))
        if len(athletes) > max_athletes:
            continue  # roundup (rankings / buzz column), not player-specific news
        text = f"{art.get('headline', '')}. {art.get('description', '')}".strip()
        comp = sid.polarity_scores(text)["compound"]
        for aid, name in athletes:
            counts[aid] = counts.get(aid, 0) + 1
            if best.get(aid) is None or abs(comp) > best[aid][0]:
                best[aid] = (abs(comp), comp, name, art.get("headline", ""))

    rows = []
    for aid, (_, comp, name, headline) in best.items():
        mult = 1 + max(-cap, min(cap, comp * scale))
        if abs(mult - 1) < 0.005:        # negligible -> treat as neutral, skip
            continue
        pct = (mult - 1) * 100
        rows.append({
            "espn_id": int(aid),
            "name": name,
            "name_key": normalize_name(name) if name else None,
            "narrative_mult": round(mult, 4),
            "narrative_reason": f"{headline} ({pct:+.0f}%)",
            "n_articles": counts.get(aid, 1),
        })

    df = pd.DataFrame(rows, columns=NUDGE_COLUMNS)
    store.save_df(df, cache)
    logger.info("News nudges: %d players moved (of %d mentioned in news)", len(df), len(best))
    return df
