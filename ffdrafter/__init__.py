"""FF Drafter — ESPN auction draft assistant.

Package layout:
  data/        free-source ingestion + player ID reconciliation
  features/    feature engineering (veteran + rookie tracks)
  model/       projections, bounded narrative nudge, backtesting
  valuation/   VOR + auction-dollar conversion + tiers
  draft/       live draft state, inflation, recommendation engine
  store, utils persistence + shared helpers

Run tooling from the project root so `config` and `ffdrafter` are importable.
"""

__version__ = "0.1.0"
