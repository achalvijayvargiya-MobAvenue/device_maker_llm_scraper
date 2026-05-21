"""
Enrich final_combined_output.csv with matching records from unique_apps.csv.

Match key : device_manufacturer + device_model  (case-insensitive, stripped)
Strategy  : For every missing cell in final_combined_output, if a matching row
            exists in unique_apps and that row has a value for the same column,
            fill it in.

Columns skipped from unique_apps (incompatible format):
  - price_inr      : unique_apps stores text ("Low", "Very High"), not INR numbers
  - cooling_system : unique_apps stores "Yes"/"No", not canonical cooling type
  - rn             : internal unique_apps counter, not relevant
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("data/output")

FINAL_FILE   = OUTPUT_DIR / "final_combined_output.csv"
UNIQUE_FILE  = OUTPUT_DIR / "unique_apps.csv"
ENRICHED_FILE = OUTPUT_DIR / "final_enriched_output.csv"

# Columns in unique_apps to NOT use for filling (format mismatch)
SKIP_FROM_UNIQUE = {"price_inr", "cooling_system", "rn"}

# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading files...")
df_final  = pd.read_csv(FINAL_FILE)
df_unique = pd.read_csv(UNIQUE_FILE)

print(f"  final_combined_output : {len(df_final):>7,} rows, {len(df_final.columns)} cols")
print(f"  unique_apps           : {len(df_unique):>7,} rows, {len(df_unique.columns)} cols")

# ── Normalize match keys ──────────────────────────────────────────────────────

def norm(s) -> str:
    """Lowercase, strip, collapse whitespace."""
    if pd.isna(s):
        return ""
    return " ".join(str(s).lower().split())

df_final["_key"]  = df_final["device_manufacturer"].apply(norm) + "|" + df_final["device_model"].apply(norm)
df_unique["_key"] = df_unique["device_manufacturer"].apply(norm) + "|" + df_unique["device_model"].apply(norm)

# ── Build lookup from unique_apps ─────────────────────────────────────────────
# Where duplicate keys exist keep the first occurrence

# Columns in unique_apps that can fill final columns (overlapping, compatible)
unique_cols = set(df_unique.columns) - SKIP_FROM_UNIQUE - {"device_manufacturer", "device_model", "_key"}
final_cols  = set(df_final.columns)  - {"device_manufacturer", "device_model", "_key"}
fillable_cols = sorted(unique_cols & final_cols)  # intersection

print(f"\nFillable columns from unique_apps: {fillable_cols}")

lookup = (
    df_unique[["_key"] + fillable_cols]
    .drop_duplicates(subset="_key", keep="first")
    .set_index("_key")
)

print(f"Lookup table built: {len(lookup):,} unique device keys in unique_apps")

# ── Enrich ───────────────────────────────────────────────────────────────────

def _is_blank(val) -> bool:
    if pd.isna(val):
        return True
    return str(val).strip() == ""

filled_counts = {col: 0 for col in fillable_cols}
matched_rows  = 0

for idx, row in df_final.iterrows():
    key = row["_key"]
    if key not in lookup.index:
        continue

    matched_rows += 1
    src = lookup.loc[key]

    for col in fillable_cols:
        if _is_blank(row[col]) and not _is_blank(src[col]):
            df_final.at[idx, col] = src[col]
            filled_counts[col] += 1

# ── Save ──────────────────────────────────────────────────────────────────────

df_final.drop(columns=["_key"], inplace=True)
df_final.to_csv(ENRICHED_FILE, index=False)

# ── Report ───────────────────────────────────────────────────────────────────

print(f"\nRows matched in unique_apps : {matched_rows:,} / {len(df_final):,}")
print(f"Saved enriched file         : {ENRICHED_FILE.name}")
print("\nCells filled per column:")
total_filled = 0
for col, cnt in sorted(filled_counts.items(), key=lambda x: -x[1]):
    if cnt > 0:
        print(f"  {col:<30} +{cnt:,} cells filled")
        total_filled += cnt
print(f"\nTotal cells filled: {total_filled:,}")

# ── Coverage before vs after ──────────────────────────────────────────────────
print("\nCoverage comparison (all 17k rows):")
df_before = pd.read_csv(FINAL_FILE)
df_after  = pd.read_csv(ENRICHED_FILE)

data_cols = [c for c in df_after.columns if c not in ("device_manufacturer", "device_model")]
print(f"  {'Column':<30} {'Before':>8}  {'After':>8}  {'Gain':>6}")
print(f"  {'-'*58}")
for col in data_cols:
    before = int(df_before[col].notna().sum()) if col in df_before.columns else 0
    after  = int(df_after[col].notna().sum())
    gain   = after - before
    gain_str = f"+{gain}" if gain > 0 else str(gain)
    print(f"  {col:<30} {before:>8,}  {after:>8,}  {gain_str:>6}")
