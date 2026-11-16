"""
FM Energy Dashboard v3 — Simulated Live Feed + Prediction
----------------------------------------------------------
Extends v2 with a "Predict next hour" button in the Prediction section.
Uses the saved GBR model (models/gbr_model.joblib) and feature order
(models/feature_order.json) — no retraining.

For each meter the prediction is built from the readings revealed so far:
  hour_of_day, day_of_week, lag_1h, lag_2h, lag_3h, lag_24h, lag_168h

At least 168 hours of history per meter are required. If any lag is
missing a clear message is shown for that meter.

Run from the project root:
    ..\vibe\Scripts\streamlit.exe run dashboard/app_v3.py --server.headless true
"""

import json
import os
import re
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Energy Dashboard v3",
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
SIM_DATA_END = pd.Timestamp("2016-02-29 23:00:00")

# Model files (relative to this file's parent directory)
_ROOT = os.path.join(os.path.dirname(__file__), "..")
_MODEL_PATH = os.path.join(_ROOT, "models", "gbr_model.joblib")
_FEATURE_PATH = os.path.join(_ROOT, "models", "feature_order.json")


# ── Load model once (cached as a resource) ────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load(_MODEL_PATH)
    with open(_FEATURE_PATH) as f:
        feature_order = json.load(f)
    return model, feature_order


model, FEATURE_ORDER = load_model()


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


# ── Prediction helper ─────────────────────────────────────────────────────────
def predict_for_meter(meter_df: pd.DataFrame) -> dict:
    """
    Build feature row for the next hour and return a result dict:
      { "target_time": Timestamp, "kwh": float }   on success
      { "error": str }                              when history is insufficient
    """
    meter_df = meter_df.sort_values("timestamp").reset_index(drop=True)

    if meter_df.empty:
        return {"error": "No readings available yet."}

    latest_ts = meter_df["timestamp"].iloc[-1]
    target_ts = latest_ts + pd.Timedelta(hours=1)

    # Build a lookup: timestamp → kWh for fast lag access
    kwh_by_ts = meter_df.set_index("timestamp")["electricity_kwh"]

    def lag(hours: int) -> float | None:
        ts = latest_ts - pd.Timedelta(hours=hours - 1)
        return kwh_by_ts.get(ts, None)

    lags = {
        "lag_1h":   kwh_by_ts.get(latest_ts, None),
        "lag_2h":   kwh_by_ts.get(latest_ts - pd.Timedelta(hours=1), None),
        "lag_3h":   kwh_by_ts.get(latest_ts - pd.Timedelta(hours=2), None),
        "lag_24h":  kwh_by_ts.get(latest_ts - pd.Timedelta(hours=23), None),
        "lag_168h": kwh_by_ts.get(latest_ts - pd.Timedelta(hours=167), None),
    }

    missing = [name for name, val in lags.items() if val is None]
    if missing:
        hours_available = len(meter_df)
        return {
            "error": (
                f"Not enough history — need 168 h, have {hours_available} h. "
                f"Missing: {', '.join(missing)}."
            )
        }

    features = {
        "hour_of_day": target_ts.hour,
        "day_of_week": target_ts.dayofweek,
        **lags,
    }

    X = np.array([[features[f] for f in FEATURE_ORDER]])
    prediction = float(model.predict(X)[0])

    return {"target_time": target_ts, "kwh": prediction}


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    if "sim_df" not in st.session_state:
        st.session_state.sim_df = load_january()
    if "last_tick" not in st.session_state:
        st.session_state.last_tick = time.time()
    if "sim_done" not in st.session_state:
        st.session_state.sim_done = False
    if "predictions" not in st.session_state:
        st.session_state.predictions = {}   # meter_id → result dict


def _restart():
    load_january.clear()
    st.session_state.sim_df = load_january()
    st.session_state.last_tick = time.time()
    st.session_state.sim_done = False
    st.session_state.predictions = {}


_init_state()

# ── Advance simulation ────────────────────────────────────────────────────────
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
        st.session_state.last_tick = time.time()
        if st.session_state.sim_df["timestamp"].max() >= SIM_DATA_END:
            st.session_state.sim_done = True

df = st.session_state.sim_df
meters = sorted(df["meter_id"].unique())
current_max_ts = df["timestamp"].max()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ FM Energy Dashboard v3")

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

# ── Section 3: Prediction ─────────────────────────────────────────────────────
st.subheader("🔮 Prediction")

if st.button("Predict next hour", type="primary"):
    results = {}
    for meter_id in meters:
        meter_df = df[df["meter_id"] == meter_id].copy()
        results[meter_id] = predict_for_meter(meter_df)
    st.session_state.predictions = results

if st.session_state.predictions:
    pred_cols = st.columns(len(meters))
    for col, meter_id in zip(pred_cols, meters):
        result = st.session_state.predictions.get(meter_id, {})
        with col:
            st.markdown(f"**{meter_id}**")
            if "error" in result:
                st.warning(result["error"], icon="⚠️")
            else:
                target = result["target_time"].strftime("%Y-%m-%d %H:%M")
                kwh = result["kwh"]
                st.metric(
                    label=f"Target: {target}",
                    value=f"{kwh:.2f} kWh",
                )
else:
    st.caption("Press **Predict next hour** to run the model on the currently visible readings.")

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
    time.sleep(5)
    st.rerun()
