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
  model/               # projections (project.py) + backtest.py       ✓
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

## How much to trust the model

Backtested against held-out seasons, our projections roughly **tie** the strong "reuse last
year's points" baseline (and the market) — they do not clearly beat it, and QB is the weakest
spot. That's why the model board **blends with consensus** by default rather than trusting the
model alone; raise `--model-weight` only as far as you trust it. The model's value is its
*divergences* (where it sees an edge the market misses), tempered by that anchor.

## Status

Done: scaffold, baseline board, live Streamlit assistant, and the projection engine v1
(nflverse data, veteran + rookie tracks, VOR→dollars, market-blended board, backtest). Next:
the bounded narrative nudge + rookie deep-dive cards (Phase 5). College stats (CFBD) are a
planned rookie refinement once a key is set. v1 strategy scope is intentionally lean.
