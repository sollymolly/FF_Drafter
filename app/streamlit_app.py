"""
app/streamlit_app.py — FF Drafter live auction dashboard.

Run from the project root:
    python -m streamlit run app/streamlit_app.py

Flow:
  1. First load with no saved draft -> setup wizard: league size, then team names.
  2. Draft screen -> log each sale (player + price + winning manager); the panels
     (your budget/max-bid, every manager's budget, inflation, best values, and a
     per-player "what should I pay?" card) update live. State is saved to disk after
     every change, so a browser refresh or restart resumes the draft.

The valuation board is built IN-PROCESS for the chosen league size from cached
inputs (ESPN market pull, projections, blend weights) via ffdrafter/board.py.
Changing team count never requires re-running scripts/build_board.py — that
script only refreshes the underlying data.
"""

import sys
from pathlib import Path

# `streamlit run` sets sys.path[0] to this file's dir; add the project root so
# `config` and `ffdrafter` import correctly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

import config
from ffdrafter import store
from ffdrafter.board import build_board
from ffdrafter.draft import engine
from ffdrafter.draft.inflation import inflation_factor
from ffdrafter.draft.state import DraftState

st.set_page_config(page_title="FF Drafter — Auction", layout="wide")


# ---------------------------------------------------------------------------
# Data loading / state plumbing
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Calibrating the board for your league…")
def league_board(teams: int, budget: int):
    """
    Board priced for THIS league's economics, rebuilt in-process from cached
    inputs (sub-second) and memoized on (teams, budget). Uses model values when
    projections are cached, else falls back to the ESPN baseline.
    """
    league = {**config.LEAGUE, "teams": int(teams), "budget": int(budget)}
    b, _ = build_board(league, source="auto")
    return b


@st.cache_data(show_spinner=False)
def load_rookie_cards():
    return store.load_df(config.PATHS["processed"] / f"rookie_cards_{config.LEAGUE['season']}.parquet")


def persist(state: DraftState) -> None:
    """Save to session_state AND disk (crash-safe resume)."""
    st.session_state["draft"] = state
    state.save()


# Restore an in-progress draft from disk on first page load.
if "draft" not in st.session_state:
    st.session_state["draft"] = DraftState.load()

state: DraftState | None = st.session_state.get("draft")


# ---------------------------------------------------------------------------
# SETUP WIZARD — step 1: league size, step 2: team names, then the draft screen.
# The board is recalibrated in-process for whatever size is chosen; no CLI step.
# ---------------------------------------------------------------------------
if state is None:
    st.title("🏈 FF Drafter — New Auction Draft")
    lg = config.LEAGUE

    if st.session_state.get("setup_step", "teams") == "teams":
        st.subheader("Step 1 of 2 — League size")
        st.caption(f"${lg['budget']}/manager · {lg['scoring']} · "
                   f"{config.roster_size()}-man rosters · season {lg['season']}")
        n_teams = st.number_input(
            "Number of teams", min_value=2, max_value=32, step=1,
            value=st.session_state.get("setup_teams", lg["teams"]),
            help="Board $ values recalibrate automatically to this league size.",
        )
        if st.button("Continue →", type="primary"):
            st.session_state["setup_teams"] = int(n_teams)
            st.session_state["setup_step"] = "names"
            st.rerun()
    else:
        teams = st.session_state.get("setup_teams", lg["teams"])
        league = {**lg, "teams": teams}
        st.subheader("Step 2 of 2 — Team names")
        st.caption(f"{teams}-team · ${lg['budget']}/manager · {lg['scoring']} · "
                   f"season {lg['season']}")
        if st.button("← Back to league size"):
            st.session_state["setup_step"] = "teams"
            st.rerun()

        with st.form("setup"):
            my_team = st.text_input("Your team name", value="My Team")
            st.write("Opponent names:")
            opp_cols = st.columns(3)
            opponents = []
            for i in range(config.num_opponents(league)):
                with opp_cols[i % 3]:
                    opponents.append(st.text_input(f"Opponent {i + 1}", value=f"Opponent {i + 1}",
                                                    key=f"opp_{i}"))
            start = st.form_submit_button("Start draft", type="primary")

        if start:
            names = [my_team] + opponents
            if any(not n.strip() for n in names):
                st.error("Every team needs a name.")
            elif len(set(names)) != len(names):
                st.error("Team names must be unique.")
            else:
                persist(DraftState.new(my_team.strip(), [o.strip() for o in opponents],
                                       league=league))
                for k in ("setup_step", "setup_teams"):
                    st.session_state.pop(k, None)
                st.rerun()
    st.stop()


# ---------------------------------------------------------------------------
# DRAFT SCREEN
# ---------------------------------------------------------------------------
# Board priced for the league this draft was created with (memoized per size).
try:
    board = league_board(state.teams, state.budget)
except Exception as e:
    st.error(f"Couldn't build the board ({e}). On a fresh machine, run "
             "`python scripts/build_board.py --refresh` once with internet access.")
    st.stop()

factor = inflation_factor(state, board)
panel = engine.manager_panel(state, board, factor=factor)

# ---- Sidebar: your team, inflation, all managers, danger zone ----
with st.sidebar:
    st.header(f"⭐ {state.my_team}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Budget", f"${state.budget_remaining(state.my_team)}")
    c2.metric("Max bid", f"${state.max_bid(state.my_team)}")
    c3.metric("Open", state.open_slots(state.my_team))

    inf_pct = (factor - 1) * 100
    st.metric("Market inflation", f"{factor:.2f}×",
              delta=f"{inf_pct:+.0f}% vs board",
              delta_color="inverse")
    st.caption("**< 1.00** → bargains ahead (overpays drained the room)  \n"
               "**> 1.00** → pay up (money still chasing fewer players)")

    st.divider()
    st.subheader("Managers")
    st.dataframe(
        panel[["manager", "budget_left", "max_bid", "surplus", "banked_edge", "needs"]]
        .sort_values(["surplus", "budget_left"], ascending=False),
        hide_index=True, width="stretch",
        column_config={
            "budget_left": st.column_config.NumberColumn("Budget", format="$%d"),
            "max_bid": st.column_config.NumberColumn("Max bid", format="$%d"),
            "surplus": st.column_config.NumberColumn(
                "Star $", format="$%d",
                help="Budget minus the realistic cost to finish their roster — "
                     "discretionary money they can throw at stars."),
            "banked_edge": st.column_config.NumberColumn(
                "Edge", format="$%d",
                help="Board value banked under market so far. Their projected final "
                     "team value is starting budget + edge, so this IS how far above "
                     "par (or below, if negative) their team sits."),
            "needs": st.column_config.TextColumn("Needs"),
        },
    )
    st.caption("**Star \\$** = money free after realistically finishing the roster. "
               "**Edge** = value banked under market. Teams high on both are the "
               "danger: they can overpay for a star and still finish ahead.")

    st.divider()
    if st.button("↻ Reload board", width="stretch"):
        league_board.clear()
        st.rerun()
    if st.button("Undo last sale", width="stretch",
                 disabled=not state.sales):
        undone = state.undo_last()
        persist(state)
        st.toast(f"Undid: {undone.name}" if undone else "Nothing to undo")
        st.rerun()
    with st.expander("Reset entire draft"):
        if st.button("⚠️ Confirm reset", type="secondary"):
            store.clear_session()
            del st.session_state["draft"]
            st.rerun()

# ---- Main: header metrics ----
st.title("FF Drafter — Live Auction")
_src = board["source"].iloc[0] if "source" in board.columns and len(board) else "?"
st.caption(f"{state.teams}-team · ${state.budget}/manager · "
           f"Values: {'model blend (projections + market)' if _src == 'model_blend' else 'ESPN consensus (baseline)'}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Players sold", len(state.sales))
m2.metric("$ in the room", f"${state.total_remaining_money()}")
m3.metric("Slots left (league)", state.total_open_slots())
m4.metric("Inflation", f"{factor:.2f}×")

left, right = st.columns([3, 2], gap="large")

# ---- Log a sale ----
with left:
    st.subheader("Log a sale")
    av = engine.available(board, state).sort_values("value", ascending=False)
    with st.form("sale", clear_on_submit=True):
        f1, f2, f3 = st.columns([3, 1, 2])
        player = f1.selectbox(
            "Player", av["name"].tolist(),
            help="Only undrafted players are listed.",
        )
        price = f2.number_input("Price $", min_value=1, value=1, step=1)
        manager = f3.selectbox("Won by", state.managers)
        submitted = st.form_submit_button("Record sale", type="primary")

    if submitted and player:
        prow = av[av["name"] == player].iloc[0]
        if price > state.budget_remaining(manager):
            st.error(f"{manager} only has ${state.budget_remaining(manager)} left.")
        else:
            state.record_sale(prow["name"], int(price), manager,
                              position=prow["position"], team=prow["team"],
                              espn_id=int(prow["espn_id"]) if "espn_id" in prow else None)
            persist(state)
            st.toast(f"{manager} ← {prow['name']} for ${int(price)}")
            st.rerun()

    # ---- What should I pay? ----
    st.subheader("What should I pay?")
    look = st.selectbox("Look up any player", board.sort_values("value", ascending=False)["name"].tolist(),
                        key="lookup")
    rec = engine.recommend_player(state, board, look, factor=factor)
    if rec:
        if rec["already_drafted"]:
            st.info(f"{rec['name']} is already off the board.")
        exp_price = rec.get("expected_price", rec["inflated_value"])
        afford = rec.get("my_afford")
        roster_capped = afford is not None and rec["suggested_max"] < min(
            rec["inflated_value"], rec["my_max_bid"])
        uplift = rec["suggested_max"] - rec["inflated_value"]
        if roster_capped:
            sm_delta, sm_color = f"{uplift:+d} roster cap", "normal"
        elif rec.get("premium") and uplift > 0:
            # the uplift you actually get (the premium net of the afford cap)
            sm_delta, sm_color = f"+{uplift} premium", "normal"
        else:
            sm_delta, sm_color = None, "normal"
        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.metric("Board value", f"${rec['board_value']}")
        r2.metric("Inflation-adj.", f"${rec['inflated_value']}",
                  delta=f"{rec['inflated_value'] - rec['board_value']:+d}")
        r3.metric("Expected price", f"${exp_price}",
                  delta=f"{exp_price - rec['inflated_value']:+d} vs adj",
                  delta_color="inverse",
                  help="Likely closing price given who actually bids: the winner pays "
                       "$1 over the second-highest willingness in the room (you "
                       "included, enforcing board value). Below adj = expect a bargain.")
        r4.metric("Suggested max", f"${rec['suggested_max']}",
                  delta=sm_delta, delta_color=sm_color,
                  help="Your walk-away price: market value plus a capped scarcity/denial "
                       "premium, but never more than your roster can absorb — or a "
                       "price you'd regret winning at.")
        r5.metric("Roster affords", f"${max(afford, 0)}" if afford is not None else "—",
                  help="The most you can pay and still field a median-pool starter at "
                       "every other open slot — relaxed toward steals: each $1 below "
                       "market buys back some balance damage (config edge_credit).")
        r6.metric("Your max bid", f"${rec['my_max_bid']}")
        notes = [f"{rec['position']} · {rec['team']} · tier {rec['tier']}",
                 f"{rec['opp_can_afford']} opponents can afford \\${rec['inflated_value']}"]
        if rec.get("is_rookie"):
            notes.append("🎓 rookie — see deep-dive below")
        if rec["last_in_tier"]:
            notes.append("⚠️ **last player in this tier** — value cliff behind him")
        if "edge" in rec:
            notes.append(f"our board \\${rec['board_value']} vs market \\${rec['market_value']} "
                         f"(edge {rec['edge']:+d}, trust {rec['trust'] * 100:.0f}%)")
        st.caption("  |  ".join(notes))
        # One verdict-first assessment paragraph. NOTE: captions are markdown, and
        # a PAIR of bare $ signs flips the text between them into LaTeX math —
        # every literal dollar must be escaped as \$.
        if afford is not None and not rec["already_drafted"]:
            sug, ceil_ = rec["suggested_max"], max(rec["my_ceiling"], 0)
            left_after = max(0, state.budget_remaining(state.my_team) - sug)
            slots_after = max(0, state.open_slots(state.my_team) - 1)
            top = rec.get("top_threat")
            top_w = rec["threats"][0]["willingness"] if rec.get("threats") else None
            if sug >= exp_price:      # you can pay what he should close at
                s = f"**Bid — up to \\${sug}** — the room prices him ~\\${exp_price}"
                if top and top_w and top_w > exp_price:
                    s += f", and **{top}** may chase to ~\\${top_w}"
                s += (f"; your roster covers it. Winning at \\${sug} leaves "
                      f"\\${left_after} for {slots_after} open slots.")
                if rec.get("premium") and sug == rec["inflated_value"] + rec["premium"]:
                    parts = []
                    if rec.get("scarcity_premium"):
                        parts.append(f"\\${rec['scarcity_premium']} tier cliff")
                    if rec.get("rivalry_premium"):
                        parts.append(f"\\${rec['rivalry_premium']} denial vs **{top}**")
                    s += f" Includes +\\${rec['premium']} premium ({' + '.join(parts)})."
            else:                     # the room should outrun your roster
                s = (f"**Let him go** — the room should take him to ~\\${exp_price}, "
                     f"past your \\${sug}. Your roster: balanced ceiling \\${ceil_}"
                     + (f", stretching to \\${afford} if he slides" if afford > ceil_ else "")
                     + f"; winning at \\${sug} would leave \\${left_after} for "
                       f"{slots_after} open slots. If bidding stalls cheap, go to "
                       f"\\${sug} and not a dollar more"
                     + (f" — otherwise nominate him to drain **{top}**." if top else "."))
            st.caption(s)
        if rec.get("threats"):
            st.markdown("**⚔️ Threatening teams** — most to least, and what they'd pay:")
            tdf = pd.DataFrame(rec["threats"][:5])
            tdf.insert(0, "rank", range(1, len(tdf) + 1))
            st.dataframe(
                tdf[["rank", "manager", "willingness", "edge_vs_room", "surplus"]],
                hide_index=True, width="stretch",
                column_config={
                    "rank": st.column_config.NumberColumn("#", width="small"),
                    "manager": st.column_config.TextColumn("Team"),
                    "willingness": st.column_config.NumberColumn(
                        "Would pay ~", format="$%d",
                        help="Their realistic ceiling: adjusted value + a share of their "
                             "threat money (spare cash + banked edge over the room), "
                             "capped by their max bid."),
                    "edge_vs_room": st.column_config.NumberColumn(
                        "Edge vs room", format="$%+d",
                        help="How far their banked value sits above the room average — "
                             "how strong their draft has been so far."),
                    "surplus": st.column_config.NumberColumn(
                        "Spare $", format="$%d",
                        help="Money beyond realistically finishing their roster."),
                },
            )
        if rec.get("narrative_reason"):
            st.caption(f"📰 {rec['narrative_reason']}")

# ---- Best available + my roster ----
with right:
    st.subheader("Best available")
    pos = st.radio("Position", ["ALL", "QB", "RB", "WR", "TE", "DST", "K"],
                   horizontal=True, label_visibility="collapsed")
    ba = engine.best_available(state, board, n=20, position=pos, factor=factor)
    st.dataframe(
        ba, hide_index=True, width="stretch", height=360,
        column_config={
            "value": st.column_config.NumberColumn("Board", format="$%d"),
            "inflated_value": st.column_config.NumberColumn("Adj", format="$%d"),
            "my_max_bid": st.column_config.NumberColumn("MyMax", format="$%d"),
            "opp_can_afford": st.column_config.NumberColumn("Opp$"),
            "edge": st.column_config.NumberColumn("Edge", format="$%+d",
                                                  help="model $ − market $ (model board only)"),
            "aav": st.column_config.NumberColumn("AAV", format="%.0f"),
            "adp": st.column_config.NumberColumn("ADP", format="%.0f"),
        },
    )

    st.subheader(f"My roster ({state.filled_slots(state.my_team)}/{state.roster_size})")
    mine = state.roster(state.my_team)
    if mine:
        import pandas as pd
        st.dataframe(
            pd.DataFrame([{"player": s.name, "pos": s.position, "$": s.price} for s in mine]),
            hide_index=True, width="stretch",
            column_config={"$": st.column_config.NumberColumn("$", format="$%d")},
        )
    else:
        st.caption("No players yet.")


# ---- Nomination strategy (full width) ----
st.divider()
st.subheader("🎯 Who to nominate")
st.caption("Auction leverage: **nominate players your opponents still need and can pay for** — "
           "especially teams shopping with house money — to drain budgets before your targets "
           "come up. `Rich demand` weights each real bidder by their spare star money, so two "
           "rich bidders beat three broke ones; `Likely buyer` is the richest of them. Your own "
           "targets are pushed down the list (**HOLD**).")
nom = engine.nomination_board(state, board, n=15, factor=factor)
if nom is not None and len(nom):
    st.dataframe(
        nom, hide_index=True, width="stretch",
        column_config={
            "inflated_value": st.column_config.NumberColumn("Price~", format="$%d"),
            "likely_buyer": st.column_config.TextColumn("Likely buyer"),
            "rich_demand": st.column_config.NumberColumn(
                "Rich demand", format="%.1f",
                help="Real bidders (need + can pay) weighted by threat money — spare "
                     "cash plus banked edge over the room: each counts 1 + that/budget."),
            "opp_demand": st.column_config.NumberColumn("Demand"),
            "opp_need": st.column_config.NumberColumn("Need"),
            "nominate_score": st.column_config.NumberColumn("Score", format="%d"),
        },
    )
else:
    st.caption("No available players to evaluate yet.")

# ---- Model vs Market edges (full width) ----
if "model_value" in board.columns:
    st.divider()
    st.subheader("📊 Model vs Market edges")
    only_trusted = st.checkbox(
        "Only positions the model earns trust (TE/WR/QB)", value=True,
        help="Hide positions with a learned blend weight of 0 (e.g. RB), where the model's "
             "disagreement is informational only and does not move the board price.")
    st.caption("**edge = our board $ − market $** — how far our (model-blended) price sits off "
               "consensus, *after* applying the learned trust. Positive → we value him above "
               "market (target); negative → market may be overpaying. **trust** = the blend "
               "weight driving it (RB = 0, so RB shows no edge — the model isn't trusted there).")
    edges = engine.value_edges(state, board, n=25, only_trusted=only_trusted)
    if edges is not None and len(edges):
        edges_disp = edges.assign(edge_pct=edges["edge_pct"] * 100, trust=edges["trust"] * 100)
        st.dataframe(
            edges_disp, hide_index=True, width="stretch", height=380,
            column_config={
                "market_value": st.column_config.NumberColumn("Market $", format="$%d"),
                "board_value": st.column_config.NumberColumn("Our $", format="$%d"),
                "edge": st.column_config.NumberColumn("Edge", format="$%+d"),
                "edge_pct": st.column_config.NumberColumn("Edge %", format="%+.0f%%"),
                "trust": st.column_config.NumberColumn("Trust", format="%.0f%%"),
                "direction": st.column_config.TextColumn("Read"),
            },
        )
    else:
        st.caption("No divergences to show for the current filter.")

# ---- Rookie deep-dive (full width) ----
cards = load_rookie_cards()
if cards is not None and len(cards):
    st.divider()
    st.subheader("🎓 Rookie deep-dive")
    pick = st.selectbox("Rookie", cards["name"].tolist(), key="rookie_pick")
    rc = cards[cards["name"] == pick].iloc[0]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Projected", f"{rc['projected_pts']:.0f} pts")
    d2.metric("Floor–Ceiling", f"{rc['floor']:.0f}–{rc['ceiling']:.0f}")
    d3.metric("Draft slot", f"#{int(rc['draft_ovr'])}" if pd.notna(rc["draft_ovr"]) else "UDFA")
    d4.metric("Comp avg", f"{rc['comp_avg_pts']:.0f} pts" if pd.notna(rc["comp_avg_pts"]) else "—")
    meta = f"{rc['position']} · {rc['team']} · {rc['college']}"
    if pd.notna(rc.get("age")):
        meta += f" · age {rc['age']:.0f}"
    st.caption(meta)
    if rc["comps"]:
        slot = int(rc["draft_ovr"]) if pd.notna(rc["draft_ovr"]) else 0
        st.markdown(f"**Comps — rookies drafted near #{slot}, and what they scored:** {rc['comps']}")

