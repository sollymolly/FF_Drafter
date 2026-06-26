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
  data/                # free-source ingestion + player ID reconciliation (ids.py)
  features/            # veteran + rookie feature tracks            (coming)
  model/               # projections, narrative nudge, backtest      (coming)
  valuation/           # VOR + auction-dollar conversion + tiers     ✓
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
python scripts/build_board.py --refresh   # pull ESPN values -> data/board/baseline.parquet
python -m streamlit run app/streamlit_app.py   # open the live draft dashboard
```

On draft night: name your team + opponents, then log each sale (player + price + winning
manager). The dashboard shows your max bid, every manager's budget, live inflation, the
best values left, and a "what should I pay for X?" card. State auto-saves after each sale,
so a refresh or restart resumes the draft.

## Status

Done: scaffold, consensus-AAV baseline board, and the live Streamlit auction assistant
(budget tracking for all managers, inflation engine, max-bid + best-values, crash-safe
resume). Next: the projection model that replaces the baseline board with our own values
(veteran + rookie tracks), then the bounded narrative nudge. v1 strategy scope is
intentionally lean — see the build plan.
