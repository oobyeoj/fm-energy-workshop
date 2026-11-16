"""
FM Energy Dashboard v2 — Simulated Live Feed
---------------------------------------------
Starts with January 2016 data for all three meters.
Every real minute, the next recorded hourly reading per meter is appended
and the charts refresh automatically.
When all February records have been replayed, the feed stops.
Press "Restart simulation" to reset to the January baseline.

Run from the project root:
    ..\vibe\Scripts\streamlit.exe run dashboard/app_v2.py --server.headless true
"""

import os
import re
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Energy Dashboard v2",
    page_icon="⚡",
    layout="wide",
)

# ── Credentials & HTTP SQL setup ──────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

_raw_url = os.environ["DATABASE_URL"]
_conn_str = re.sub(r"[&?]channel_binding=require", "", _raw_url)
_host = re.search(r"@([^/]+)/", _conn_str).group(1)
_SQL_URL = f"https://{_host}/sql"
_HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": _conn_str,
}

METER_COLOURS = {
    "METER_01": "#1f77b4",
    "METER_02": "#ff7f0e",
    "METER_03": "#2ca02c",
}

# Simulation boundaries
SIM_START = pd.Timestamp("2016-01-01 00:00:00")   # first timestamp to show
SIM_SEED_END = pd.Timestamp("2016-01-31 23:00:00") # end of initial January load
SIM_DATA_END = pd.Timestamp("2016-02-29 23:00:00") # last timestamp in the dataset


# ── Neon helpers ──────────────────────────────────────────────────────────────
def _run_sql(query: str, params: list | None = None) -> dict:
    payload = {"query": query}
    if params:
        payload["params"] = params
    resp = requests.post(_SQL_URL, headers=_HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _to_df(result: dict) -> pd.DataFrame:
    if not result.get("rows"):
        return pd.DataFrame()
    fields = [col["name"] for col in result["fields"]]
    rows = [dict(zip(fields, row.values())) for row in result["rows"]]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["electricity_kwh"] = df["electricity_kwh"].astype(float)
    return df


@st.cache_data(show_spinner="Loading January baseline…")
def load_january() -> pd.DataFrame:
    """Fetch all January 2016 readings (the starting dataset)."""
    result = _run_sql(
        """
        SELECT
            timestamp AT TIME ZONE 'UTC' AS timestamp,
            meter_id,
            building_id,
            electricity_kwh
        FROM meter_readings_source
        WHERE timestamp >= $1
          AND timestamp <= $2
        ORDER BY timestamp, meter_id
        """,
        ["2016-01-01 00:00:00+00", "2016-01-31 23:00:00+00"],
    )
    return _to_df(result)


def fetch_next_hour(after_timestamp: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch one reading per meter for the single next recorded hour
    after `after_timestamp`.
    """
    result = _run_sql(
        """
        WITH next_ts AS (
            SELECT MIN(timestamp) AS ts
            FROM meter_readings_source
            WHERE timestamp > $1
              AND timestamp <= $2
        )
        SELECT
            r.timestamp AT TIME ZONE 'UTC' AS timestamp,
            r.meter_id,
            r.building_id,
            r.electricity_kwh
        FROM meter_readings_source r
        JOIN next_ts ON r.timestamp = next_ts.ts
        ORDER BY r.meter_id
        """,
        [
            after_timestamp.strftime("%Y-%m-%d %H:%M:%S+00"),
            SIM_DATA_END.strftime("%Y-%m-%d %H:%M:%S+00"),
        ],
    )
    return _to_df(result)


# ── Session state initialisation ──────────────────────────────────────────────
def _init_state():
    if "sim_df" not in st.session_state:
        st.session_state.sim_df = load_january()
    if "last_tick" not in st.session_state:
        st.session_state.last_tick = time.time()
    if "sim_done" not in st.session_state:
        st.session_state.sim_done = False


def _restart():
    """Reset simulation back to the January baseline."""
    load_january.clear()                        # clear cache so we re-fetch cleanly
    st.session_state.sim_df = load_january()
    st.session_state.last_tick = time.time()
    st.session_state.sim_done = False


_init_state()

# ── Advance the simulation if one real minute has elapsed ─────────────────────
if not st.session_state.sim_done:
    elapsed = time.time() - st.session_state.last_tick
    if elapsed >= 60:
        current_max = st.session_state.sim_df["timestamp"].max()
        if current_max < SIM_DATA_END:
            new_rows = fetch_next_hour(current_max)
            if not new_rows.empty:
                st.session_state.sim_df = pd.concat(
                    [st.session_state.sim_df, new_rows], ignore_index=True
                )
        # Update tick regardless — prevents back-to-back fetches on slow renders
        st.session_state.last_tick = time.time()
        # Check if we've now reached the last record
        if st.session_state.sim_df["timestamp"].max() >= SIM_DATA_END:
            st.session_state.sim_done = True

df = st.session_state.sim_df
meters = sorted(df["meter_id"].unique())
current_max_ts = df["timestamp"].max()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ FM Energy Dashboard v2")

status_col, btn_col = st.columns([6, 1])
with status_col:
    if st.session_state.sim_done:
        st.caption(
            f"Historical data · {df['timestamp'].min().strftime('%Y-%m-%d')} "
            f"to {current_max_ts.strftime('%Y-%m-%d %H:%M')} · "
            f"{len(df):,} readings — ✅ Simulation complete"
        )
    else:
        next_tick_in = max(0, 60 - int(time.time() - st.session_state.last_tick))
        st.caption(
            f"Historical data · {df['timestamp'].min().strftime('%Y-%m-%d')} "
            f"to {current_max_ts.strftime('%Y-%m-%d %H:%M')} · "
            f"{len(df):,} readings · ▶ Next fetch in {next_tick_in}s"
        )
with btn_col:
    if st.button("↺ Restart simulation", use_container_width=True):
        _restart()
        st.rerun()

st.divider()

# ── Section 1: Per-meter charts ───────────────────────────────────────────────
st.subheader("Hourly Electricity Usage — by Meter")

cols = st.columns(len(meters))
for col, meter_id in zip(cols, meters):
    meter_df = df[df["meter_id"] == meter_id].copy()
    building = meter_df["building_id"].iloc[0] if not meter_df.empty else ""

    fig = px.line(
        meter_df,
        x="timestamp",
        y="electricity_kwh",
        title=meter_id,
        labels={"timestamp": "", "electricity_kwh": "kWh"},
        color_discrete_sequence=[METER_COLOURS.get(meter_id, "#636efa")],
    )
    fig.update_layout(
        title_font_size=16,
        margin=dict(t=40, b=20, l=0, r=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#e5e5e5"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    fig.update_traces(line_width=1.2)

    with col:
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Building: {building}")

st.divider()

# ── Section 2: Total energy chart ────────────────────────────────────────────
st.subheader("Total Hourly Electricity Usage — All Meters")

total_df = (
    df.groupby("timestamp", as_index=False)["electricity_kwh"]
    .sum()
    .rename(columns={"electricity_kwh": "total_kwh"})
)

fig_total = px.area(
    total_df,
    x="timestamp",
    y="total_kwh",
    labels={"timestamp": "", "total_kwh": "Total kWh"},
    color_discrete_sequence=["#7b2d8b"],
)
fig_total.update_layout(
    margin=dict(t=10, b=20, l=0, r=0),
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor="#e5e5e5"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    hovermode="x unified",
)
fig_total.update_traces(line_width=1.5, opacity=0.75)

st.plotly_chart(fig_total, use_container_width=True)
st.divider()

# ── Section 3: Prediction placeholder ────────────────────────────────────────
st.subheader("🔮 Prediction")
st.info(
    "**Coming soon.** This section will display a one-hour-ahead electricity "
    "forecast for each meter using the trained Gradient Boosting model.",
    icon="🏗️",
)
st.divider()

# ── Section 4: LLM Analysis placeholder ──────────────────────────────────────
st.subheader("🤖 LLM Analysis")
st.info(
    "**Coming soon.** This section will let you ask questions about the energy "
    "data and receive an AI-generated analysis powered by an LLM.",
    icon="🏗️",
)

# ── Auto-rerun while simulation is active ────────────────────────────────────
if not st.session_state.sim_done:
    # Sleep until the next tick boundary, then trigger a rerun.
    # We use a short sleep so the countdown in the caption stays responsive.
    time.sleep(5)
    st.rerun()
