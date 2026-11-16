"""
Upload three_meters.csv to Neon via the HTTP SQL API.
Uses the Neon-Connection-String header instead of psycopg2.
"""

import csv
import json
import os
import re

import requests
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

raw_url = os.environ["DATABASE_URL"]
# Strip channel_binding=require (blocked by some firewalls / not needed for HTTP API)
conn_str = re.sub(r"[&?]channel_binding=require", "", raw_url)

# Derive the HTTP SQL endpoint from the host in the connection string
# postgresql://user:pass@<host>/dbname?...  →  https://<host>/sql
host_match = re.search(r"@([^/]+)/", conn_str)
if not host_match:
    raise ValueError("Cannot parse host from DATABASE_URL")
host = host_match.group(1)
sql_url = f"https://{host}/sql"

HEADERS = {
    "Content-Type": "application/json",
    "Neon-Connection-String": conn_str,
}

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "three_meters.csv")
BATCH_SIZE = 500


def run_sql(query: str, params: list | None = None) -> dict:
    payload = {"query": query}
    if params:
        payload["params"] = params
    resp = requests.post(sql_url, headers=HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Create table ──────────────────────────────────────────────────────────────
print("Creating table meter_readings_source …")
run_sql("""
CREATE TABLE IF NOT EXISTS meter_readings_source (
    id             BIGSERIAL PRIMARY KEY,
    timestamp      TIMESTAMPTZ NOT NULL,
    meter_id       TEXT        NOT NULL,
    building_id    TEXT        NOT NULL,
    electricity_kwh DOUBLE PRECISION NOT NULL,
    UNIQUE (meter_id, timestamp)
)
""")
print("  Table ready.")

# ── Load CSV ──────────────────────────────────────────────────────────────────
print("Loading CSV …")
with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))
print(f"  {len(rows):,} rows read.")

# ── Upload in batches (parameterised INSERT … ON CONFLICT DO NOTHING) ─────────
# The Neon HTTP API accepts a single query + params list per request.
# We batch with a multi-row VALUES clause, building params as a flat list.

def upload_batch(batch: list[dict]) -> int:
    n = len(batch)
    # Build  ($1,$2,$3,$4), ($5,$6,$7,$8), …
    placeholders = ", ".join(
        f"(${i*4+1}, ${i*4+2}, ${i*4+3}, ${i*4+4})"
        for i in range(n)
    )
    query = f"""
        INSERT INTO meter_readings_source (timestamp, meter_id, building_id, electricity_kwh)
        VALUES {placeholders}
        ON CONFLICT (meter_id, timestamp) DO NOTHING
    """
    params = []
    for row in batch:
        params.extend([
            row["timestamp"],
            row["meter_id"],
            row["building_id"],
            float(row["electricity_kwh"]),
        ])
    result = run_sql(query, params)
    return result.get("rowCount", 0)

inserted = 0
for start in range(0, len(rows), BATCH_SIZE):
    batch = rows[start : start + BATCH_SIZE]
    inserted += upload_batch(batch)
    print(f"  uploaded rows {start+1}–{start+len(batch)} …")

print(f"\nDone. Rows inserted (new): {inserted:,}")

# ── Summary per meter ─────────────────────────────────────────────────────────
print("\n── Summary per meter ──────────────────────────────────────────────────")
result = run_sql("""
    SELECT
        meter_id,
        COUNT(*)                        AS row_count,
        MIN(timestamp) AT TIME ZONE 'UTC' AS earliest,
        MAX(timestamp) AT TIME ZONE 'UTC' AS latest
    FROM meter_readings_source
    GROUP BY meter_id
    ORDER BY meter_id
""")

# Print column headers
fields = [col["name"] for col in result["fields"]]
print("  " + "  ".join(f"{h:<30}" for h in fields))
print("  " + "-" * (32 * len(fields)))
for row in result["rows"]:
    # rows are dicts when rowAsArray is false
    if isinstance(row, dict):
        values = [row[f] for f in fields]
    else:
        values = row
    print("  " + "  ".join(f"{str(v):<30}" for v in values))
