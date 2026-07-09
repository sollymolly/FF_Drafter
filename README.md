# 🏈 FF Drafter — ESPN Auction Draft Assistant

A decision-support tool for **ESPN-style fantasy football auction drafts** ($200/manager,
any whole-dollar bid). It computes its own player values from a projection model and, on
draft day, tells you the right **max bid** for any nominated player while tracking every
manager's budget in real time.

> **League-flexible:** team count, budget, scoring, and roster are all config. Replacement
> levels, total money, and opponent count are *derived* from `LEAGUE["teams"]`, so a
> 10-team and a 12-team league differ by a single number in `config.py`.

## Two subsystems, one contract

1. **Projection & Valuation engine** (pre-draft) → produces a **valuation board**
   (`player → projected pts → VOR → auction $ → tier`), using **free** data only.
2. **Live Auction Assistant** (draft day, Streamlit) → consumes the board, tracks
   budgets/inflation, and answers *"what do I pay for X?"* and *"what's the best value left?"*

The valuation board is the hand-off between them, so the live tool can run on a simple
baseline board first and swap in the smart model later without changing draft-day code.

## Project layout

```
config.py              # league settings + derived replacement levels / money (edit this)
ffdrafter/
  utils.py             # logging + player-name normalization
  store.py             # parquet boards + crash-safe JSON session snapshot
  data/                # free-source ingestion (nflverse, ESPN) + ID reconciliation ✓
  features/            # veteran + rookie feature tracks              ✓
  model/               # project.py, backtest.py, narrative.py, rookie_card.py ✓
  valuation/           # VOR + auction-dollar conversion + tiers      ✓
  draft/               # live state, inflation, recommendation engine ✓
app/streamlit_app.py   # draft-day dashboard                         ✓
scripts/build_board.py # build the valuation board                   ✓
data/{raw,processed,board}/
```

## Setup

```bash
pip install -r requirements.txt          # all free
cp .env.example .env                      # add a free CollegeFootballData key (rookies)
```

Run tooling from the project root so `config` and `ffdrafter` import correctly.

```bash
python config.py                          # print derived league settings (sanity check)
```

## Use it

```bash
python scripts/build_board.py --source model --refresh   # train + build data/board/model.parquet
python -m streamlit run app/streamlit_app.py             # open the live draft dashboard
```

`--source baseline` (default) uses ESPN consensus values; `--source model` runs our
projection engine and blends it with the market (`--model-weight 0..1`, default 0.5). The
app prefers the model board if present, else the baseline. On draft night: name your team +
opponents, then log each sale (player + price + winning manager). The dashboard shows your
max bid, every manager's budget, live inflation, the best values left, and a "what should I
pay for X?" card. State auto-saves after each sale, so a refresh/restart resumes the draft.

The pay card also prices **the risk of NOT getting a player** (`ffdrafter/draft/threat.py`):
it tracks each opponent's *banked edge* (board value bought under market — their license to
overpay and still finish ahead) and *spare star money* (budget beyond realistically finishing
their roster), forecasts the **expected closing price** from the likely bidders
(second-price auction logic — a lone rich bidder raises nothing until someone pushes them),
and lifts your suggested max by a **capped** scarcity + denial premium when a strong team is
chasing — never past a price you'd regret winning at. Nominations are weighted to drain the
teams shopping with house money first. Knobs live in `config.THREAT`; sanity-check offline
with `python scripts/sanity_threat.py [--app]`.

## How much to trust the model

We backtest three forecasters on **the market's preseason top-200 skill players** (2021–2025)
— the pool you actually draft from — and score each by rank correlation with what happened
that season (mean across seasons):

| position | model | naive (reuse last yr) | market (FantasyPros ECR) |
|----------|:-----:|:---------------------:|:------------------------:|
| QB | 0.20 | 0.32 | **0.40** |
| RB | 0.43 | 0.40 | **0.58** |
| WR | 0.46 | 0.45 | **0.58** |
| TE | 0.25 | 0.25 | **0.42** |

The **market clearly beats our model at every position** (and beats "reuse last year" too);
the model only slightly edges naive at RB/WR and trails it at QB/TE. So the per-position blend
weights — learned by `model/blend.py` to maximize held-out rank correlation — come out to
**0 across the board**. The honest, data-driven verdict: *the current model shouldn't move
prices.*

What that means in practice:
- `build_board.py --source model` loads these learned weights automatically, so the board's
  **dollar values track the market** while each player still shows the model's `projected_pts`
  beside them — a visible second opinion for spotting divergences, not a price driver.
  `--model-weight X` still forces a flat blend if you want to experiment.
- To make the model *worth* blending in, it needs better inputs than "last season's box score
  + age": per-game/role stability, snap & target trends, vacated volume, and era-normalized
  multi-year history. That's the next build phase.

Rerun the evaluation anytime with `python -m ffdrafter.model.blend` (prints the full table and
rewrites `data/processed/blend_weights.json`).

## Status

**v1 complete.** Scaffold, baseline board, live Streamlit assistant, projection engine
(nflverse data, veteran + rookie tracks, VOR→dollars, market-blended board, backtest), a
bounded news-sentiment nudge (±10% cap, reason shown), rookie deep-dive cards (draft
capital + landing spot + historical comps), and the per-opponent **threat model** (banked
edge + spare star money → expected closing price, capped scarcity/denial premium,
drain-the-rich nominations). The model board now uses **learned per-position
blend weights** (`model/blend.py`; currently 0 — see "How much to trust the model"), with
`--model-weight` as a manual override; the nudge can be skipped with `--no-narrative`.

Planned refinements: College stats (CollegeFootballData) to sharpen rookies once a key is
set; optional ESPN live-draft sync. v1 strategy scope is intentionally lean — see the
build plan.
