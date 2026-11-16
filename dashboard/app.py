"""
FM Energy Dashboard
-------------------
Reads historical meter readings from Neon PostgreSQL and displays:
  - Three hourly electricity charts side by side (one per meter)
  - Total energy chart (all meters summed by timestamp)
  - Placeholder sections for Prediction and LLM Analysis

Run from the project root:
    ..\vibe\Scripts\streamlit.exe run dashboard/app.py --server.headless true
"""

import os
import re

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Energy Dashboard",
    page_icon="⚡",
    layout="wide",
)

# ── Load credentials ──────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

_raw_url = os.environ["DATABASE_URL"]
# Remove channel_binding=require — not needed for the HTTP SQL API
_conn_str = re.sub(r"[&?]channel_binding=require", "", _raw_url)

# Derive the Neon HTTP SQL endpoint from the host
_host = re.search(r"@([^/]+)/", _conn_str).group(1)
_SQL_URL = f"https://{_host}/sql"
_HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": _conn_str,
}


# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner="Loading historical readings…")
def load_readings() -> pd.DataFrame:
    """Fetch all rows from meter_readings_source and return a DataFrame."""
    resp = requests.post(
        _SQL_URL,
        headers=_HEADERS,
        json={
            "query": """
                SELECT
                    timestamp AT TIME ZONE 'UTC' AS timestamp,
                    meter_id,
                    building_id,
                    electricity_kwh
                FROM meter_readings_source
                ORDER BY timestamp
            """
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    fields = [col["name"] for col in data["fields"]]
    rows = [dict(zip(fields, row.values())) for row in data["rows"]]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["electricity_kwh"] = df["electricity_kwh"].astype(float)
    return df


# ── Load data ─────────────────────────────────────────────────────────────────
df = load_readings()
meters = sorted(df["meter_id"].unique())

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ FM Energy Dashboard")
st.caption(
    f"Historical data · {df['timestamp'].min().strftime('%Y-%m-%d')} "
    f"to {df['timestamp'].max().strftime('%Y-%m-%d')} · "
    f"{len(df):,} readings across {len(meters)} meters"
)
st.divider()

# ── Section 1: Per-meter charts ───────────────────────────────────────────────
st.subheader("Hourly Electricity Usage — by Meter")

cols = st.columns(len(meters))

METER_COLOURS = {
    "METER_01": "#1f77b4",
    "METER_02": "#ff7f0e",
    "METER_03": "#2ca02c",
}

for col, meter_id in zip(cols, meters):
    meter_df = df[df["meter_id"] == meter_id].copy()
    building = meter_df["building_id"].iloc[0] if not meter_df.empty else ""

    fig = px.line(
        meter_df,
        x="timestamp",
        y="electricity_kwh",
        title=f"{meter_id}",
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
