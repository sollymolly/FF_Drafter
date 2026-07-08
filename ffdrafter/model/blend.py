"""
model/blend.py — learn per-position model/market blend weights from backtests.

The board blends model dollars with market dollars; the open question is how much
to trust the model at each position. Answer it empirically: for each position,
grid-search the weight w whose blend best rank-orders the draft pool across the
held-out backtest seasons.

Market ECR is a rank with no point scale, so before blending we convert positional
rank -> expected points with an isotonic curve fit on the OTHER seasons (leave-one-
out: season T's curve never sees season T). Ties within EPS of the best rho resolve
toward the market (smaller w) — when the data can't tell the difference, lean on
the anchor.

Caveat, stated plainly: w itself is tuned on the same backtest seasons we report,
so the blend's rho is slightly optimistic; the model/naive/market columns are clean
out-of-sample. Weights land in data/processed/blend_weights.json and are picked up
by scripts/build_board.py automatically.

Run:  python -m ffdrafter.model.blend
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

import config
from ffdrafter.model import backtest
from ffdrafter.utils import get_logger

logger = get_logger(__name__)

WEIGHTS_PATH = config.PATHS["processed"] / "blend_weights.json"
GRID = np.round(np.linspace(0.0, 1.0, 21), 2)
EPS = 0.002          # rho ties resolve toward the market-leaning weight
MIN_GROUP = 8        # skip (position, season) groups smaller than this


def _market_points(detail: pd.DataFrame, pos: str, season: int) -> pd.Series:
    """Positional rank -> expected points, fit on the OTHER seasons (no leakage)."""
    other = detail[(detail["position"] == pos) & (detail["season"] != season)]
    cur = detail[(detail["position"] == pos) & (detail["season"] == season)]
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(other["mkt_pos_rank"].astype(float), other["actual"].astype(float))
    return pd.Series(iso.predict(cur["mkt_pos_rank"].astype(float)), index=cur.index)


def learn_weights(detail: pd.DataFrame, seasons=None) -> dict[str, dict]:
    """
    Per-position: best w (model share), its mean rho, and the pure-market /
    pure-model endpoints for context. Needs >= 2 seasons in `detail`.
    """
    if seasons is None:
        seasons = sorted(detail["season"].unique())
    det = detail[detail["season"].isin(seasons)].copy()

    det["market_pts"] = np.nan
    for pos in det["position"].unique():
        for t in seasons:
            m = (det["position"] == pos) & (det["season"] == t)
            det.loc[m, "market_pts"] = _market_points(det, pos, t)

    results: dict[str, dict] = {}
    for pos in sorted(det["position"].unique()):
        rho_by_w: dict[float, float] = {}
        for w in GRID:
            rhos = []
            for t in seasons:
                g = det[(det["position"] == pos) & (det["season"] == t)]
                if len(g) < MIN_GROUP:
                    continue
                blended = w * g["model"] + (1 - w) * g["market_pts"]
                rhos.append(spearmanr(blended, g["actual"]).correlation)
            rho_by_w[float(w)] = float(np.nanmean(rhos)) if rhos else float("nan")
        best = max(rho_by_w.values())
        w_star = min(w for w, r in rho_by_w.items() if r >= best - EPS)
        results[pos] = {
            "w": w_star,
            "rho_at_w": rho_by_w[w_star],
            "rho_market": rho_by_w[0.0],
            "rho_model": rho_by_w[1.0],
            "curve": rho_by_w,
        }
    return results


def _print_weights(title: str, res: dict[str, dict]) -> None:
    print(f"\n{title}")
    print(f"  {'pos':<4} {'w*':>5} {'rho@w*':>8} {'rho market':>11} {'rho model':>10}")
    for pos, r in res.items():
        print(f"  {pos:<4} {r['w']:>5.2f} {r['rho_at_w']:>8.3f} "
              f"{r['rho_market']:>11.3f} {r['rho_model']:>10.3f}")


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    summary, detail = backtest.run()

    print("\n=== Three-way backtest: rank correlation with season outcome "
          f"(market's preseason top-{backtest.POOL_SIZE}) ===")
    for t in sorted(summary["season"].unique()):
        print(f"\n-- {t} --")
        print(summary[summary["season"] == t].drop(columns="season").to_string(index=False))

    print("\n=== Mean across seasons ===")
    agg = (summary.groupby("pos")[["model_rho", "naive_rho", "market_rho",
                                   "model_MAE", "naive_MAE"]].mean().round(3))
    print(agg.to_string())

    seasons = [int(t) for t in sorted(detail["season"].unique())]  # py ints: JSON-safe
    recent = seasons[-3:]
    res_all = learn_weights(detail)
    res_recent = learn_weights(detail, seasons=recent)
    _print_weights(f"=== Learned weights, all seasons {seasons} ===", res_all)
    _print_weights(f"=== Learned weights, recent seasons {recent} "
                   "(training depth matches production) ===", res_recent)

    weights = {pos: r["w"] for pos, r in res_recent.items()}
    payload = {
        "weights": weights,
        "seasons": recent,
        "pool_size": backtest.POOL_SIZE,
        "created": date.today().isoformat(),
        "scores_recent": {p: {k: v for k, v in r.items() if k != "curve"}
                          for p, r in res_recent.items()},
        "scores_all_seasons": {p: {k: v for k, v in r.items() if k != "curve"}
                               for p, r in res_all.items()},
    }
    WEIGHTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nActive weights (recent seasons): {weights}")
    print(f"Saved -> {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
