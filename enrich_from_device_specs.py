"""
Enrich final_enriched_output.csv with device_specs.csv.

Rules:
- Match on device_manufacturer + device_model (case-insensitive)
- Fill only blank cells — never overwrite existing data
- cooling_system "No" → "Standard", "Yes" → skip (too vague to map)
- launch_date normalised to "YYYY-MM" (handles both YYYY-MM and DD-MM-YYYY)
- price_inr: numeric in device_specs, directly usable
- After all enrichment: any still-blank cooling_system → "none"
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

OUTPUT_DIR = Path("data/output")

SOURCE_FILE  = OUTPUT_DIR / "final_enriched_output.csv"
SPECS_FILE   = OUTPUT_DIR / "device_specs.csv"
OUT_FILE     = OUTPUT_DIR / "final_enriched_output_v2.csv"

# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading files...")
df = pd.read_csv(SOURCE_FILE)
df_specs = pd.read_csv(SPECS_FILE, low_memory=False)

# Drop unnamed trailing columns
df_specs = df_specs.loc[:, ~df_specs.columns.str.startswith("Unnamed")]

print(f"  final_enriched_output : {len(df):>7,} rows")
print(f"  device_specs          : {len(df_specs):>7,} rows, cols: {df_specs.columns.tolist()}")

# ── Normalise helpers ─────────────────────────────────────────────────────────

def norm_key(s) -> str:
    if pd.isna(s):
        return ""
    return " ".join(str(s).lower().split())

def norm_launch_date(val) -> str | None:
    """Convert DD-MM-YYYY or YYYY-MM-DD or YYYY-MM → YYYY-MM. Returns None on failure."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Already YYYY-MM
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return s
    # YYYY-MM-DD
    m = re.fullmatch(r"(\d{4})-(\d{2})-\d{2}", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # DD-MM-YYYY
    m = re.fullmatch(r"(\d{2})-(\d{2})-(\d{4})", s)
    if m:
        return f"{m.group(3)}-{m.group(2)}"
    # MM-YYYY or MM/YYYY
    m = re.fullmatch(r"(\d{2})[/-](\d{4})", s)
    if m:
        return f"{m.group(2)}-{m.group(1)}"
    return None

def norm_cooling(val) -> str | None:
    """Map device_specs cooling values to canonical form."""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s == "no":
        return "Standard"
    # "Yes" is too vague — skip it so we don't overwrite with wrong type
    return None

def is_blank(val) -> bool:
    if pd.isna(val):
        return True
    return str(val).strip() == ""

# ── Prepare device_specs lookup ───────────────────────────────────────────────

# Normalise launch_date and cooling_system in specs before building lookup
df_specs["launch_date"]    = df_specs["launch_date"].apply(norm_launch_date)
df_specs["cooling_system"] = df_specs["cooling_system"].apply(norm_cooling)
df_specs["_key"] = df_specs["device_manufacturer"].apply(norm_key) + "|" + df_specs["device_model"].apply(norm_key)

# Fillable columns = overlap between specs and final (excluding keys)
SKIP = {"device_manufacturer", "device_model", "_key"}
specs_cols = set(df_specs.columns) - SKIP
final_cols = set(df) - SKIP
fillable   = sorted(specs_cols & final_cols)

print(f"\nFillable columns from device_specs: {fillable}")

# Where duplicates exist keep the one with the most non-null values
df_specs["_score"] = df_specs[fillable].notna().sum(axis=1)
df_specs = df_specs.sort_values("_score", ascending=False)
lookup = (
    df_specs[["_key"] + fillable]
    .drop_duplicates(subset="_key", keep="first")
    .set_index("_key")
)
print(f"Lookup table: {len(lookup):,} unique keys")

# ── Enrich ───────────────────────────────────────────────────────────────────

df["_key"] = df["device_manufacturer"].apply(norm_key) + "|" + df["device_model"].apply(norm_key)

filled_counts = {col: 0 for col in fillable}
matched = 0

for idx, row in df.iterrows():
    key = row["_key"]
    if key not in lookup.index:
        continue
    matched += 1
    src = lookup.loc[key]
    for col in fillable:
        if is_blank(row[col]) and not is_blank(src[col]):
            df.at[idx, col] = src[col]
            filled_counts[col] += 1

print(f"\nRows matched in device_specs : {matched:,} / {len(df):,}")

# ── Post-process: blank cooling_system → "none" ───────────────────────────────

cooling_none_count = df["cooling_system"].apply(is_blank).sum()
df["cooling_system"] = df["cooling_system"].apply(
    lambda v: "none" if is_blank(v) else v
)
print(f"cooling_system blanks filled with 'none': {cooling_none_count:,}")

# ── Save ──────────────────────────────────────────────────────────────────────

df.drop(columns=["_key"], inplace=True)
df.to_csv(OUT_FILE, index=False)

# ── Report ───────────────────────────────────────────────────────────────────

print(f"\nSaved: {OUT_FILE.name}")
print("\nCells filled from device_specs:")
total = 0
for col, cnt in sorted(filled_counts.items(), key=lambda x: -x[1]):
    if cnt > 0:
        print(f"  {col:<30} +{cnt:,}")
        total += cnt
print(f"\nTotal new cells filled: {total:,}")

# ── Full coverage summary ─────────────────────────────────────────────────────

print("\nFinal coverage:")
df_v1 = pd.read_csv(SOURCE_FILE)
data_cols = [c for c in df.columns if c not in ("device_manufacturer", "device_model")]
print(f"  {'Column':<30} {'Before':>8}  {'After':>8}  {'Gain':>6}")
print(f"  {'-'*58}")
for col in data_cols:
    before = int(df_v1[col].notna().sum()) if col in df_v1.columns else 0
    # For cooling_system after = all rows (we filled blanks with "none")
    after  = int(df[col].notna().sum())
    gain   = after - before
    gain_str = f"+{gain}" if gain > 0 else str(gain)
    print(f"  {col:<30} {before:>8,}  {after:>8,}  {gain_str:>6}")
