"""
scripts/sanity_threat.py — offline sanity harness for the threat model.

Run from the project root (no network, no cached data needed):
    python scripts/sanity_threat.py            # engine + threat checks on a synthetic board
    python scripts/sanity_threat.py --app      # also boot the Streamlit app headlessly

The synthetic market is calibrated through the REAL pipeline (build_baseline_board),
so prices, inflation, and medians behave like a live board. Scenarios covered:
  - fresh symmetric room  -> threat is silent (expected price == inflated value)
  - the Gibbs bargain     -> banked edge licenses an overpay; premium is capped
  - mid-tier player       -> star scaling keeps premiums negligible
  - broke manager         -> max_bid removes him from the threat list
  - undo                  -> everything derives back cleanly
  - league-size sweep     -> price-curve pins rise SUB-proportionally with room
                             money, and the fresh-draft roster cap never clips
                             the #1 player at 10/12/14 teams (the fresh-draft
                             anchor κ makes this hold at any size; the cap exists
                             to clamp over-commitment, never the first nomination)

--app seeds a synthetic market cache + session, runs the app via streamlit's
AppTest, and ALWAYS deletes what it seeded. It refuses to run if a real session
or market cache exists, so it can never clobber a live draft.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from ffdrafter.draft import engine, threat
from ffdrafter.draft.inflation import inflation_factor
from ffdrafter.draft.state import DraftState
from ffdrafter.utils import normalize_name


def curve(top, tail_start, n, decay=0.88):
    """AAV curve: explicit elite prices, then a smooth geometric tail (floor $1)."""
    vals = list(top)
    v = float(tail_start)
    while len(vals) < n:
        vals.append(max(1, int(round(v))))
        v *= decay
    return vals


def synthetic_market() -> pd.DataFrame:
    """
    Market frame with deliberate elite duos (tier-1 cliffs) for scarcity tests.
    The duos need a wide AAV moat below them: rank-pinning (PRICE_CURVE targets)
    prices players by OVERALL rank, so a position with 3+ players inside the
    pinned zone gets smooth within-position prices and no tier cliff survives.
    """
    entries, eid = [], [1000]

    def pos_curve(pos, names, aavs):
        for nm, a in zip(names, aavs):
            eid[0] += 1
            entries.append(dict(espn_id=eid[0], name=nm, name_key=normalize_name(nm),
                                position=pos, team="XX", aav=float(a)))

    pos_curve("RB", ["Jahmyr Gibbs", "Bijan Robinson"] + [f"RB Guy {i}" for i in range(38)],
              curve([61, 59], 24, 40, decay=0.93))
    pos_curve("WR", ["CeeDee Lamb", "Puka Nacua"] + [f"WR Guy {i}" for i in range(38)],
              curve([57, 56], 22, 40, decay=0.93))
    pos_curve("QB", [f"QB Guy {i}" for i in range(16)], curve([42], 20, 16))
    pos_curve("TE", [f"TE Guy {i}" for i in range(14)], curve([34], 15, 14))
    pos_curve("DST", [f"DST Unit {i}" for i in range(14)], [1] * 14)
    pos_curve("K", [f"Kicker {i}" for i in range(14)], [1] * 14)
    pos_curve("RB", [f"Deep RB {i}" for i in range(30)], [1] * 30)
    pos_curve("WR", [f"Deep WR {i}" for i in range(30)], [1] * 30)
    return pd.DataFrame(entries)


def mk_board(league):
    from ffdrafter.valuation.auction import build_baseline_board
    return build_baseline_board(synthetic_market(), league)


def run_engine_checks():
    # A 10-team room like the real league: with the stars premium, budgets must
    # leave space for a second stud — in a 12-team $200 room they don't.
    league10 = {**config.LEAGUE, "teams": 10}
    board = mk_board(league10)
    state = DraftState.new("Me", [f"Opp{i}" for i in range(1, 10)], league=league10)

    # ---------- league price curve pins the fitted targets ----------
    pc = getattr(config, "PRICE_CURVE", {}) or {}
    tgt = pc.get("targets") or (
        list(__import__("numpy").linspace(pc["top1_target"], pc["topn_target"],
                                          int(pc.get("top_n", 10))))
        if pc.get("top1_target") and pc.get("topn_target") else None)
    if tgt:
        top = board["value"].nlargest(len(tgt))
        for got, want in zip(top.tolist(), tgt):
            assert abs(got - want) <= 1, f"target miss: ${got} vs ${want}"
        ranked = board["value"].sort_values(ascending=False).reset_index(drop=True)
        assert ranked.is_monotonic_decreasing
        assert ranked.iloc[len(tgt)] >= 0.75 * tgt[-1], \
            f"cliff below the band: rank-{len(tgt)+1} price ${ranked.iloc[len(tgt)]}"
        print(f"price curve OK   top-{len(tgt)} pinned ${int(top.iloc[0])}→${int(top.iloc[-1])} "
              f"(avg ${top.mean():.0f}), rank-{len(tgt)+1} ${int(ranked.iloc[len(tgt)])} — no cliff")

    def val(nm):
        return int(board.loc[board["name_key"] == normalize_name(nm), "value"].iloc[0])

    # ---------- fresh draft: symmetric room => threat is quiet ----------
    f0 = inflation_factor(state, board)
    prof0 = threat.manager_profiles(state, board, f0)
    assert prof0["banked_edge"].eq(0).all()
    assert prof0["threat_money"].eq(0).all(), "symmetric room must carry no threat money"
    r0 = engine.recommend_player(state, board, "Puka Nacua", factor=f0)
    assert r0["rivalry_premium"] == 0
    assert abs(r0["expected_price"] - r0["inflated_value"]) <= 1, \
        "symmetric room should price at inflated value"
    assert r0["suggested_max"] == min(r0["inflated_value"], r0["my_max_bid"])
    print("fresh draft OK  ",
          {k: r0[k] for k in ("inflated_value", "expected_price", "cost_to_win",
                              "premium", "suggested_max")})

    # ---------- the Gibbs scenario ----------
    state.record_sale("Jahmyr Gibbs", val("Jahmyr Gibbs") - 25, "Opp1", position="RB")  # $25 under
    state.record_sale("CeeDee Lamb", val("CeeDee Lamb"), "Opp2", position="WR")         # par
    state.record_sale("QB Guy 0", val("QB Guy 0") + 5, "Opp3", position="QB")           # $5 over
    f1 = inflation_factor(state, board)
    prof1 = threat.manager_profiles(state, board, f1).set_index("manager")
    assert prof1.loc["Opp1", "banked_edge"] == 25
    assert prof1.loc["Opp2", "banked_edge"] == 0
    assert prof1.loc["Opp3", "banked_edge"] == -5
    assert prof1.loc["Opp1", "power"] == state.budget + 25
    assert prof1.loc["Opp1", "threat_money"] >= 20, \
        "the bargain hunter's banked edge is his license to overpay"
    assert prof1.loc["Opp1", "threat_money"] > prof1.loc["Opp2", "threat_money"]
    print("profiles OK\n",
          prof1[["budget_left", "banked_edge", "fill_cost", "surplus", "excess",
                 "threat_money", "power"]].sort_values("threat_money", ascending=False)
          .head(4).to_string())

    r1 = engine.recommend_player(state, board, "Puka Nacua", factor=f1)
    assert r1["threats"], "someone must be a credible bidder on a top WR"
    assert r1["threats"][0]["manager"] == "Opp1", f"Opp1 should be top threat: {r1['threats']}"
    assert r1["threats"][0]["willingness"] > r1["inflated_value"], "Gibbs team chases past value"
    assert r1["expected_price"] >= r1["inflated_value"]
    assert r1["cost_to_win"] == r1["threats"][0]["willingness"] + 1
    assert r1["rivalry_premium"] > 0, "credible bidder up on the room => denial premium"
    assert r1["scarcity_premium"] > 0, "Lamb sold => Puka is the last tier-1 WR I still need"
    assert r1["premium"] <= r1["premium_cap"]
    assert r1["suggested_max"] <= r1["inflated_value"] + r1["premium_cap"]
    assert r1["suggested_max"] <= state.max_bid("Me")
    assert r1["suggested_max"] > min(r1["inflated_value"], r1["my_max_bid"]), \
        "premium must lift my walk-away"
    print("gibbs scenario OK",
          {k: r1[k] for k in ("inflated_value", "expected_price", "cost_to_win",
                              "scarcity_premium", "rivalry_premium", "premium",
                              "suggested_max")})
    print("   threats:", r1["threats"])

    # mid-tier player: star_factor keeps uplift and premium negligible
    r_mid = engine.recommend_player(state, board, "WR Guy 15", factor=f1)
    assert r_mid["premium"] <= 3, f"no meaningful premium on a mid player, got {r_mid['premium']}"
    print("mid player OK   ",
          {k: r_mid[k] for k in ("inflated_value", "expected_price", "premium", "suggested_max")})

    # ---------- nomination: rich demand + likely buyer ----------
    nom = engine.nomination_board(state, board, n=12, factor=f1)
    assert {"likely_buyer", "rich_demand", "nominate_score"} <= set(nom.columns)
    assert (nom.loc[nom["suggestion"].str.startswith("HOLD"), "nominate_score"] == 0).all()
    print("nomination OK")

    # ---------- manager panel merge (and backward-compat without a board) ----------
    panel = engine.manager_panel(state, board, factor=f1)
    assert {"surplus", "banked_edge", "excess", "threat_money", "power"} <= set(panel.columns)
    assert "surplus" not in engine.manager_panel(state).columns
    print("panel OK")

    # ---------- broke manager can't be a threat ----------
    state.record_sale("RB Guy 0", 186, "Opp4", position="RB")   # blows nearly the whole budget
    f2 = inflation_factor(state, board)
    r2 = engine.recommend_player(state, board, "Bijan Robinson", factor=f2)
    assert all(t["manager"] != "Opp4" for t in r2["threats"]), "Opp4 max_bid ~0, cannot chase"
    prof2 = threat.manager_profiles(state, board, f2).set_index("manager")
    assert prof2.loc["Opp4", "banked_edge"] < -40
    assert prof2.loc["Opp4", "max_bid"] == 0
    assert r2["last_in_tier"] and r2["scarcity_premium"] > 0
    print("broke manager OK", {k: r2[k] for k in ("expected_price", "premium", "suggested_max")})

    # ---------- undo restores the world ----------
    state.undo_last()
    prof3 = threat.manager_profiles(state, board, inflation_factor(state, board)).set_index("manager")
    assert prof3.loc["Opp4", "banked_edge"] == 0
    print("undo OK")

    # ---------- my own scope: one big buy must clamp the next suggestion ----------
    state.record_sale("Bijan Robinson", val("Bijan Robinson") + 2, "Me", position="RB")
    f3 = inflation_factor(state, board)
    r3 = engine.recommend_player(state, board, "RB Guy 0", factor=f3)
    assert r3["my_ceiling"] < r3["inflated_value"], "big buy must shrink the balanced ceiling"
    assert r3["my_afford"] >= r3["my_ceiling"], "edge credit only relaxes the ceiling"
    assert r3["suggested_max"] == max(0, min(r3["my_max_bid"],
                                             r3["inflated_value"] + r3["premium"],
                                             r3["my_afford"]))
    assert r3["suggested_max"] < min(r3["inflated_value"], r3["my_max_bid"]), \
        "roster cap must bind well below market and max bid"
    nom2 = engine.nomination_board(state, board, n=30, factor=f3)
    rb0 = nom2[nom2["name"] == "RB Guy 0"]
    if len(rb0):
        assert not rb0.iloc[0]["suggestion"].startswith("HOLD"), \
            "unaffordable player must flip from HOLD to DRAIN"
    print("my scope OK    ",
          {k: r3[k] for k in ("inflated_value", "my_ceiling", "my_afford",
                              "suggested_max", "my_max_bid")})
    state.undo_last()

    print("\nENGINE SANITY CHECKS PASSED")


def run_curve_scaling_checks():
    """
    The fitted pins are 10-team dollars. A bigger room must lift them
    sub-proportionally (config.PRICE_CURVE elasticity) instead of dumping all
    its extra money on ranks 31+ — absolute pins inflate every mid-pool median,
    which skews fill_cost/surplus/threat_money and made the fresh-draft roster
    cap clip the #1 player ($91 for a $95 Gibbs at 12 teams).

    Scaling STRUCTURE is asserted on the synthetic board; the fresh-draft
    "roster cap must not clip the #1" invariant is asserted at EVERY swept size
    on both boards. A fixed budget cannot buy the #1 pin AND median-pool
    starters everywhere in a room richer than the fitted one (at any
    elasticity), so the invariant is delivered by the fresh-draft anchor κ
    (threat.fresh_draft_anchor), which scales my reserve to what a fresh budget
    can actually hold next to the #1 price.
    """
    pc = getattr(config, "PRICE_CURVE", {}) or {}
    if not (pc.get("targets") and pc.get("fitted_teams") and pc.get("fitted_budget")):
        print("curve scaling SKIPPED (no fitted targets or fitted size recorded)")
        return
    fitted_money = pc["fitted_teams"] * pc["fitted_budget"]

    def fresh_top_rec(board, teams, budget):
        state = DraftState.new("Me", [f"Opp{i}" for i in range(1, teams)],
                               league={**config.LEAGUE, "teams": teams, "budget": budget})
        f = inflation_factor(state, board)
        top_name = board.nlargest(1, "value").iloc[0]["name"]
        return engine.recommend_player(state, board, top_name, factor=f)

    def check_fresh_invariant(rec, teams, label):
        headroom = rec["my_afford"] - rec["inflated_value"]
        print(f"curve scaling {teams}T [{label}]: #1 ${rec['board_value']}, "
              f"afford ${rec['my_afford']}, suggested ${rec['suggested_max']} "
              f"(headroom {headroom:+d})")
        assert headroom >= 0 and rec["suggested_max"] >= rec["inflated_value"], (
            f"{teams}T fresh draft [{label}]: roster cap ${rec['my_afford']} clips "
            f"the #1 at ${rec['inflated_value']} — must only bind after over-commitment")

    tops = {}
    for teams in (10, 12, 14):
        league = {**config.LEAGUE, "teams": teams, "budget": pc["fitted_budget"]}
        rec = fresh_top_rec(mk_board(league), teams, pc["fitted_budget"])
        tops[teams] = rec["board_value"]
        check_fresh_invariant(rec, teams, "synthetic")
    assert tops[10] == pc["targets"][0], "the fitted room must reproduce the fitted pin exactly"
    assert tops[10] < tops[12] < tops[14], "pins must rise with room money"
    assert tops[14] / tops[10] < (14 * pc["fitted_budget"]) / fitted_money, \
        "pins must rise SUB-proportionally with money (willingness saturates)"

    cache = config.PATHS["raw"] / f"espn_market_{config.LEAGUE['season']}.parquet"
    if cache.exists():
        from ffdrafter.board import build_board
        for teams in (pc["fitted_teams"], 12, 14):
            league = {**config.LEAGUE, "teams": teams, "budget": pc["fitted_budget"]}
            rec = fresh_top_rec(build_board(league, source="auto")[0], teams, pc["fitted_budget"])
            check_fresh_invariant(rec, teams, "real board")
    else:
        print("curve scaling: no market cache — real-board invariant not checked")
    print("curve scaling OK")


def run_app_smoke():
    """Boot the real app headlessly on seeded synthetic data, then clean up."""
    from ffdrafter import store

    cache = config.PATHS["raw"] / f"espn_market_{config.LEAGUE['season']}.parquet"
    session = config.PATHS["session"]
    if cache.exists() or session.exists():
        sys.exit("refusing --app: a market cache or live session exists — "
                 "won't risk clobbering real draft data.")
    try:
        mkt = synthetic_market()
        mkt["adp"] = range(1, len(mkt) + 1)
        store.save_df(mkt, cache)
        state = DraftState.new("Me", [f"Opp{i}" for i in range(1, 12)])
        state.record_sale("Jahmyr Gibbs", 55, "Opp1", position="RB", team="XX", espn_id=1001)
        state.save()

        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"),
                               default_timeout=60)
        at.run()
        assert not at.exception, f"app raised: {at.exception}"
        at.selectbox(key="lookup").select("Puka Nacua").run()
        assert not at.exception, f"lookup rerun raised: {at.exception}"
        labels = [m.label for m in at.metric]
        assert "Expected price" in labels and "Suggested max" in labels, labels
        print("APP SMOKE OK — metrics:", labels)
    finally:
        cache.unlink(missing_ok=True)
        session.unlink(missing_ok=True)


if __name__ == "__main__":
    run_engine_checks()
    run_curve_scaling_checks()
    if "--app" in sys.argv:
        run_app_smoke()
