"""
draft/state.py — live auction draft state.

The single source of truth is the append-only `sales` log. Every per-manager number
(budget left, roster, open slots, max bid) is DERIVED from it, so "undo" is just
popping the last sale and everything recomputes. The state serializes to/from a
plain dict for the crash-safe JSON snapshot (store.save_session).

League-agnostic: budget, roster size, and manager count all come from config, so a
10-team and 12-team draft work identically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import config
from ffdrafter import store
from ffdrafter.utils import get_logger, normalize_name

logger = get_logger(__name__)


@dataclass
class Sale:
    """One completed auction purchase."""
    name: str
    name_key: str
    position: str
    team: str
    price: int
    manager: str
    espn_id: Optional[int] = None


@dataclass
class DraftState:
    managers: list           # all manager names; managers[0] == my_team
    my_team: str
    budget: int
    roster_size: int
    roster_slots: dict
    teams: int
    season: int
    scoring: str
    sales: list = field(default_factory=list)   # list[Sale]

    # ----- construction -----
    @classmethod
    def new(cls, my_team: str, opponents: list, league: dict = config.LEAGUE) -> "DraftState":
        managers = [my_team] + [o for o in opponents]
        return cls(
            managers=managers,
            my_team=my_team,
            budget=league["budget"],
            roster_size=config.roster_size(league),
            roster_slots=dict(league["roster_slots"]),
            teams=league["teams"],
            season=league["season"],
            scoring=league["scoring"],
            sales=[],
        )

    # ----- mutations -----
    def record_sale(self, name: str, price: int, manager: str,
                    position: str = "", team: str = "", espn_id: Optional[int] = None) -> Sale:
        if manager not in self.managers:
            raise ValueError(f"Unknown manager: {manager!r}")
        sale = Sale(
            name=name, name_key=normalize_name(name), position=position,
            team=team, price=int(price), manager=manager, espn_id=espn_id,
        )
        self.sales.append(sale)
        return sale

    def undo_last(self) -> Optional[Sale]:
        return self.sales.pop() if self.sales else None

    # ----- queries (all derived from `sales`) -----
    def drafted_keys(self) -> set:
        return {s.name_key for s in self.sales}

    def is_drafted(self, name: str) -> bool:
        return normalize_name(name) in self.drafted_keys()

    def sales_for(self, manager: str) -> list:
        return [s for s in self.sales if s.manager == manager]

    def spent(self, manager: str) -> int:
        return sum(s.price for s in self.sales_for(manager))

    def budget_remaining(self, manager: str) -> int:
        return self.budget - self.spent(manager)

    def filled_slots(self, manager: str) -> int:
        return len(self.sales_for(manager))

    def open_slots(self, manager: str) -> int:
        return self.roster_size - self.filled_slots(manager)

    def filled_by_position(self, manager: str) -> dict:
        """Count of players a manager has drafted at each position."""
        from collections import Counter
        return dict(Counter(s.position for s in self.sales_for(manager) if s.position))

    def position_needs(self, manager: str) -> dict:
        """
        Open STARTER slots per position, flex-aware. A shared FLEX slot still open adds
        demand to every flex-eligible position (RB/WR/TE), since any of them could fill
        it — deliberately inclusive, because for nomination leverage we care whether a
        manager might still bid on a position, not the exact slot they'd use.
        Bench depth is intentionally ignored (weak demand).
        """
        filled = self.filled_by_position(manager)
        slots = self.roster_slots
        flex_positions = config.LEAGUE.get("flex_positions", ["RB", "WR", "TE"])
        needs = {p: max(0, slots.get(p, 0) - filled.get(p, 0)) for p in config.SCORABLE_POSITIONS}
        surplus = sum(max(0, filled.get(p, 0) - slots.get(p, 0)) for p in flex_positions)
        flex_open = max(0, slots.get("FLEX", 0) - surplus)
        if flex_open:
            for p in flex_positions:
                needs[p] += flex_open
        return needs

    def max_bid(self, manager: str) -> int:
        """Most this manager can bid now: must reserve $1 for each OTHER open slot."""
        opens = self.open_slots(manager)
        if opens <= 0:
            return 0
        return max(0, self.budget_remaining(manager) - (opens - 1))

    def total_remaining_money(self) -> int:
        return sum(self.budget_remaining(m) for m in self.managers)

    def total_open_slots(self) -> int:
        return sum(self.open_slots(m) for m in self.managers)

    def roster(self, manager: str) -> list:
        return self.sales_for(manager)

    # ----- persistence -----
    def to_dict(self) -> dict:
        return asdict(self)   # nested Sale dataclasses become dicts

    @classmethod
    def from_dict(cls, d: dict) -> "DraftState":
        d = dict(d)
        sales = [Sale(**s) for s in d.pop("sales", [])]
        obj = cls(**d)
        obj.sales = sales
        return obj

    def save(self, path=None):
        return store.save_session(self.to_dict(), path)

    @classmethod
    def load(cls, path=None) -> Optional["DraftState"]:
        d = store.load_session(path)
        return cls.from_dict(d) if d else None
