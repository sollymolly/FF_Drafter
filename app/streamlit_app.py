"""
app/streamlit_app.py — FF Drafter live auction dashboard.

Run from the project root:
    python -m streamlit run app/streamlit_app.py

Flow:
  1. First load with no saved draft -> Setup screen (name your team + opponents).
  2. Draft screen -> log each sale (player + price + winning manager); the panels
     (your budget/max-bid, every manager's budget, inflation, best values, and a
     per-player "what should I pay?" card) update live. State is saved to disk after
     every change, so a browser refresh or restart resumes the draft.
"""

import sys
from pathlib import Path

# `streamlit run` sets sys.path[0] to this file's dir; add the project root so
# `config` and `ffdrafter` import correctly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

import config
from ffdrafter import store
from ffdrafter.draft import engine
from ffdrafter.draft.inflation import inflation_factor
from ffdrafter.draft.state import DraftState

st.set_page_config(page_title="FF Drafter — Auction", layout="wide")


# ---------------------------------------------------------------------------
# Data loading / state plumbing
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_board():
    # Prefer the model board if it's been built; fall back to the baseline.
    b = store.load_board("model")
    if b is None:
        b = store.load_board("baseline")
    return b


def persist(state: DraftState) -> None:
    """Save to session_state AND disk (crash-safe resume)."""
    st.session_state["draft"] = state
    state.save()


board = load_board()
if board is None:
    st.error("No valuation board found. Build it first:  `python scripts/build_board.py`")
    st.stop()

# Restore an in-progress draft from disk on first page load.
if "draft" not in st.session_state:
    st.session_state["draft"] = DraftState.load()

state: DraftState | None = st.session_state.get("draft")


# ---------------------------------------------------------------------------
# SETUP SCREEN
# ---------------------------------------------------------------------------
if state is None:
    st.title("🏈 FF Drafter — New Auction Draft")
    lg = config.LEAGUE
    st.caption(f"{lg['teams']}-team · ${lg['budget']}/manager · {lg['scoring']} · "
               f"{config.roster_size()}-man rosters · season {lg['season']}  "
               f"(edit config.py to change)")

    with st.form("setup"):
        my_team = st.text_input("Your team name", value="My Team")
        st.write("Opponent names:")
        opp_cols = st.columns(3)
        opponents = []
        for i in range(config.num_opponents()):
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
            persist(DraftState.new(my_team.strip(), [o.strip() for o in opponents]))
            st.rerun()
    st.stop()


# ---------------------------------------------------------------------------
# DRAFT SCREEN
# ---------------------------------------------------------------------------
factor = inflation_factor(state, board)
panel = engine.manager_panel(state)

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
        panel[["manager", "budget_left", "open_slots", "max_bid"]]
        .sort_values("budget_left", ascending=False),
        hide_index=True, width="stretch",
        column_config={
            "budget_left": st.column_config.NumberColumn("Budget", format="$%d"),
            "max_bid": st.column_config.NumberColumn("Max bid", format="$%d"),
            "open_slots": st.column_config.NumberColumn("Open"),
        },
    )

    st.divider()
    if st.button("↻ Reload board", width="stretch"):
        load_board.clear()
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
st.caption(f"Values: {'model blend (projections + market)' if _src == 'model_blend' else 'ESPN consensus (baseline)'}")
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
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Board value", f"${rec['board_value']}")
        r2.metric("Inflation-adj.", f"${rec['inflated_value']}",
                  delta=f"{rec['inflated_value'] - rec['board_value']:+d}")
        r3.metric("Your max bid", f"${rec['my_max_bid']}")
        r4.metric("Suggested max", f"${rec['suggested_max']}")
        notes = [f"{rec['position']} · {rec['team']} · tier {rec['tier']}",
                 f"{rec['opp_can_afford']} opponents can afford ${rec['inflated_value']}"]
        if rec["last_in_tier"]:
            notes.append("⚠️ **last player in this tier** — a small premium is justified")
        st.caption("  |  ".join(notes))

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
