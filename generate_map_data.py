"""
Convert chicago_licenses_enriched.csv → businesses.json for the map panel.
Run locally (where the CSV lives), then commit businesses.json to the repo.

Usage:
    python generate_map_data.py
"""

import ast
import json
import pandas as pd

CSV_PATH = "chicago_licenses_enriched.csv"
OUT_PATH = "businesses.json"
MAX_RECORDS = 40000

print("Loading CSV…")
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Total rows: {len(df):,}")

# Active businesses only (event=0 means still open)
df = df[df["event"] == 0].copy()
print(f"Active businesses: {len(df):,}")

# Drop rows without coordinates
df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

# Parse ALL_LICENSES (stored as string repr of a Python list in CSV)
def parse_licenses(val):
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) and result else ["Unknown"]
    except Exception:
        return [str(val)]

df["ALL_LICENSES_PARSED"] = df["ALL_LICENSES"].apply(parse_licenses)
df["PRIMARY_LICENSE"]     = df["ALL_LICENSES_PARSED"].apply(lambda x: x[0])

# Extract start year
df["START_YEAR"] = pd.to_datetime(df["MIN_START_DATE"], errors="coerce").dt.year
df["START_YEAR"] = df["START_YEAR"].fillna(0).astype(int)

# Sort by most recent, sample if needed
df = df.sort_values("START_YEAR", ascending=False)
if len(df) > MAX_RECORDS:
    df = df.head(MAX_RECORDS)

print(f"Exporting {len(df):,} records…")

records = []
for _, row in df.iterrows():
    name = str(row.get("LEGAL_NAME", "")).strip()[:45] or "Unknown"
    addr = str(row.get("ADDRESS", "")).strip()
    all_types = row["ALL_LICENSES_PARSED"]
    records.append({
        "n":   name,
        "a":   addr,
        "t":   str(row["PRIMARY_LICENSE"]),   # primary — used for dot color
        "ts":  all_types,                     # all types — used for tooltip & filter
        "y":   int(row["START_YEAR"]) if row["START_YEAR"] > 0 else None,
        "lat": round(float(row["LATITUDE"]),  5),
        "lng": round(float(row["LONGITUDE"]), 5),
    })

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, separators=(",", ":"))

size_mb = len(json.dumps(records)) / 1_000_000
print(f"Saved {OUT_PATH}  ({len(records):,} businesses, ~{size_mb:.1f} MB)")
