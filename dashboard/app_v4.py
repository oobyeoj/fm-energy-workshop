"""
FM Energy Dashboard v4 — Live Feed + Prediction + LLM Analysis
---------------------------------------------------------------
Extends v3: replaces the LLM Analysis placeholder with a working panel.

Fixed prompt: "Give me a summary of the energy usage for METER_02 for the past 2 days"
When Send is pressed:
  - The latest 48 hours of METER_02 readings currently visible are prepared in Python
  - Stats (total, average, min, max kWh) are calculated
  - A JSON payload is built with the meter name, time range, stats, and all 48 readings
  - The JSON is attached to the fixed prompt and sent to OpenAI Chat Completions
  - The response is stored in session state so reruns / auto-ticks do not re-call the API

Shows: response text, "Based on readings up to <time>", LLM call count, and a
collapsed expander showing the exact JSON sent.

Run from the project root:
    ..\vibe\Scripts\streamlit.exe run dashboard/app_v4.py --server.headless true
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
from openai import OpenAI

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FM Energy Dashboard v4",
    page_icon="⚡",
    layout="wide",
)

# ── Credentials ───────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

_raw_url = os.environ["DATABASE_URL"]
_conn_str = re.sub(r"[&?]channel_binding=require", "", _raw_url)
_host = re.search(r"@([^/]+)/", _conn_str).group(1)
_SQL_URL = f"https://{_host}/sql"
_HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": _conn_str,
}

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

METER_COLOURS = {
    "METER_01": "#1f77b4",
    "METER_02": "#ff7f0e",
    "METER_03": "#2ca02c",
}

SIM_DATA_END = pd.Timestamp("2016-02-29 23:00:00")

_ROOT         = os.path.join(os.path.dirname(__file__), "..")
_MODEL_PATH   = os.path.join(_ROOT, "models", "gbr_model.joblib")
_FEATURE_PATH = os.path.join(_ROOT, "models", "feature_order.json")

LLM_PROMPT = "Give me a summary of the energy usage for METER_02 for the past 2 days"

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    mdl = joblib.load(_MODEL_PATH)
    with open(_FEATURE_PATH) as f:
        feat = json.load(f)
    return mdl, feat

@st.cache_resource(show_spinner=False)
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)

model, FEATURE_ORDER = load_model()
oai_client = get_openai_client()

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
            meter_id, building_id, electricity_kwh
        FROM meter_readings_source
        WHERE timestamp >= $1 AND timestamp <= $2
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
            WHERE timestamp > $1 AND timestamp <= $2
        )
        SELECT r.timestamp AT TIME ZONE 'UTC' AS timestamp,
               r.meter_id, r.building_id, r.electricity_kwh
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
    meter_df = meter_df.sort_values("timestamp").reset_index(drop=True)
    if meter_df.empty:
        return {"error": "No readings available yet."}

    latest_ts  = meter_df["timestamp"].iloc[-1]
    target_ts  = latest_ts + pd.Timedelta(hours=1)
    kwh_by_ts  = meter_df.set_index("timestamp")["electricity_kwh"]

    lags = {
        "lag_1h":   kwh_by_ts.get(latest_ts, None),
        "lag_2h":   kwh_by_ts.get(latest_ts - pd.Timedelta(hours=1), None),
        "lag_3h":   kwh_by_ts.get(latest_ts - pd.Timedelta(hours=2), None),
        "lag_24h":  kwh_by_ts.get(latest_ts - pd.Timedelta(hours=23), None),
        "lag_168h": kwh_by_ts.get(latest_ts - pd.Timedelta(hours=167), None),
    }
    missing = [k for k, v in lags.items() if v is None]
    if missing:
        return {
            "error": (
                f"Not enough history — need 168 h, have {len(meter_df)} h. "
                f"Missing: {', '.join(missing)}."
            )
        }

    features = {"hour_of_day": target_ts.hour, "day_of_week": target_ts.dayofweek, **lags}
    X = np.array([[features[f] for f in FEATURE_ORDER]])
    return {"target_time": target_ts, "kwh": float(model.predict(X)[0])}

# ── LLM helper ────────────────────────────────────────────────────────────────
def build_meter02_payload(sim_df: pd.DataFrame) -> tuple[dict, str]:
    """
    Take the last 48 visible hours for METER_02, compute stats,
    and return (payload_dict, json_string).
    """
    m2 = (
        sim_df[sim_df["meter_id"] == "METER_02"]
        .sort_values("timestamp")
        .tail(48)
        .reset_index(drop=True)
    )

    readings = [
        {"timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"), "kwh": round(row["electricity_kwh"], 3)}
        for _, row in m2.iterrows()
    ]
    kwh_vals = m2["electricity_kwh"].tolist()

    payload = {
        "meter": "METER_02",
        "time_range": {
            "from": m2["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M") if not m2.empty else "n/a",
            "to":   m2["timestamp"].iloc[-1].strftime("%Y-%m-%d %H:%M") if not m2.empty else "n/a",
        },
        "statistics": {
            "total_kwh":   round(sum(kwh_vals), 3),
            "average_kwh": round(sum(kwh_vals) / len(kwh_vals), 3) if kwh_vals else 0,
            "min_kwh":     round(min(kwh_vals), 3) if kwh_vals else 0,
            "max_kwh":     round(max(kwh_vals), 3) if kwh_vals else 0,
        },
        "hourly_readings": readings,
    }
    return payload, json.dumps(payload, indent=2)

def call_openai(prompt: str, data_json: str) -> str:
    """Send the fixed prompt + JSON data context to OpenAI and return the reply text."""
    user_message = f"{prompt}\n\nData context (JSON):\n{data_json}"
    response = oai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an energy analyst assistant. "
                    "You will be given hourly electricity meter readings in JSON format "
                    "and asked to summarise and interpret the data clearly and concisely."
                ),
            },
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content

# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    if "sim_df"       not in st.session_state: st.session_state.sim_df       = load_january()
    if "last_tick"    not in st.session_state: st.session_state.last_tick    = time.time()
    if "sim_done"     not in st.session_state: st.session_state.sim_done     = False
    if "predictions"  not in st.session_state: st.session_state.predictions  = {}
    if "llm_response" not in st.session_state: st.session_state.llm_response = None
    if "llm_payload"  not in st.session_state: st.session_state.llm_payload  = None
    if "llm_latest_ts"not in st.session_state: st.session_state.llm_latest_ts= None
    if "llm_call_count"not in st.session_state:st.session_state.llm_call_count = 0

def _restart():
    load_january.clear()
    st.session_state.sim_df        = load_january()
    st.session_state.last_tick     = time.time()
    st.session_state.sim_done      = False
    st.session_state.predictions   = {}
    st.session_state.llm_response  = None
    st.session_state.llm_payload   = None
    st.session_state.llm_latest_ts = None
    # Note: llm_call_count is intentionally kept across restarts (session total)

_init_state()

# ── Advance simulation ────────────────────────────────────────────────────────
if not st.session_state.sim_done:
    if time.time() - st.session_state.last_tick >= 60:
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
st.title("⚡ FM Energy Dashboard v4")

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
        meter_df, x="timestamp", y="electricity_kwh", title=meter_id,
        labels={"timestamp": "", "electricity_kwh": "kWh"},
        color_discrete_sequence=[METER_COLOURS.get(meter_id, "#636efa")],
    )
    fig.update_layout(
        title_font_size=16, margin=dict(t=40, b=20, l=0, r=0),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#e5e5e5"),
        plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
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
    total_df, x="timestamp", y="total_kwh",
    labels={"timestamp": "", "total_kwh": "Total kWh"},
    color_discrete_sequence=["#7b2d8b"],
)
fig_total.update_layout(
    margin=dict(t=10, b=20, l=0, r=0),
    xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#e5e5e5"),
    plot_bgcolor="white", paper_bgcolor="white", hovermode="x unified",
)
fig_total.update_traces(line_width=1.5, opacity=0.75)
st.plotly_chart(fig_total, use_container_width=True)
st.divider()

# ── Section 3: Prediction ─────────────────────────────────────────────────────
st.subheader("🔮 Prediction")

if st.button("Predict next hour", type="primary"):
    results = {}
    for meter_id in meters:
        results[meter_id] = predict_for_meter(df[df["meter_id"] == meter_id].copy())
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
                st.metric(
                    label=f"Target: {result['target_time'].strftime('%Y-%m-%d %H:%M')}",
                    value=f"{result['kwh']:.2f} kWh",
                )
else:
    st.caption("Press **Predict next hour** to run the model on the currently visible readings.")

st.divider()

# ── Section 4: LLM Analysis ───────────────────────────────────────────────────
st.subheader("🤖 LLM Analysis")

# Read-only prompt display
st.text_input(
    label="Prompt",
    value=LLM_PROMPT,
    disabled=True,
    label_visibility="collapsed",
)

send_col, _ = st.columns([1, 5])
with send_col:
    send_pressed = st.button("Send", type="primary", use_container_width=True)

if send_pressed:
    m2_latest = df[df["meter_id"] == "METER_02"]["timestamp"].max()
    m2_hours  = len(df[df["meter_id"] == "METER_02"])

    if m2_hours < 1:
        st.warning("No METER_02 readings are available yet.", icon="⚠️")
    else:
        payload_dict, payload_json = build_meter02_payload(df)
        with st.spinner("Calling OpenAI…"):
            response_text = call_openai(LLM_PROMPT, payload_json)
        st.session_state.llm_response   = response_text
        st.session_state.llm_payload    = payload_json
        st.session_state.llm_latest_ts  = m2_latest
        st.session_state.llm_call_count += 1

# Display stored response
if st.session_state.llm_response:
    latest_label = (
        st.session_state.llm_latest_ts.strftime("%Y-%m-%d %H:%M")
        if st.session_state.llm_latest_ts else "unknown"
    )
    st.caption(
        f"Based on readings up to **{latest_label}** · "
        f"LLM requests this session: **{st.session_state.llm_call_count}**"
    )
    st.markdown(st.session_state.llm_response)

    with st.expander("Data sent to the LLM", expanded=False):
        st.code(st.session_state.llm_payload, language="json")
else:
    st.caption("Press **Send** to analyse the latest 48 hours of METER_02 data.")

# ── Auto-rerun while simulation is active ────────────────────────────────────
if not st.session_state.sim_done:
    time.sleep(5)
    st.rerun()
