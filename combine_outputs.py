"""
Combine all_device_model_output.csv + pending_device_output.csv into one
final output file, dropping rows where every column except device_manufacturer
and device_model is blank.
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data/output")

FILE_A = OUTPUT_DIR / "all_device_model_output.csv"
FILE_B = OUTPUT_DIR / "pending_device_output.csv"
FINAL  = OUTPUT_DIR / "final_combined_output.csv"

# Identity columns — these are NOT checked for blankness
ID_COLS = {"device_manufacturer", "device_model"}

# ── Load ──────────────────────────────────────────────────────────────────────

df_a = pd.read_csv(FILE_A)
df_b = pd.read_csv(FILE_B)

print(f"Loaded {FILE_A.name}: {len(df_a):,} rows, {len(df_a.columns)} cols")
print(f"Loaded {FILE_B.name}: {len(df_b):,} rows, {len(df_b.columns)} cols")

# Drop unnamed / empty trailing columns (artefact of extra commas)
df_a = df_a.loc[:, ~df_a.columns.str.startswith("Unnamed")]
df_b = df_b.loc[:, ~df_b.columns.str.startswith("Unnamed")]

# ── Align columns (use union, fill missing with NaN) ─────────────────────────

all_cols = list(dict.fromkeys(list(df_a.columns) + list(df_b.columns)))
df_a = df_a.reindex(columns=all_cols)
df_b = df_b.reindex(columns=all_cols)

# ── Combine ───────────────────────────────────────────────────────────────────

combined = pd.concat([df_a, df_b], ignore_index=True)
print(f"\nCombined total  : {len(combined):,} rows")

# ── Filter: drop rows where ALL non-ID columns are blank ─────────────────────

data_cols = [c for c in combined.columns if c not in ID_COLS]

# A row is "all blank" if every data column is NaN or empty string
def _is_blank(val) -> bool:
    if pd.isna(val):
        return True
    return str(val).strip() == ""

all_blank_mask = combined[data_cols].apply(
    lambda row: all(_is_blank(v) for v in row), axis=1
)

dropped = all_blank_mask.sum()
final = combined[~all_blank_mask].reset_index(drop=True)

print(f"Dropped (all-blank data cols): {dropped:,} rows")
print(f"Final output    : {len(final):,} rows")

# ── Save ──────────────────────────────────────────────────────────────────────

final.to_csv(FINAL, index=False)
print(f"\nSaved to: {FINAL}")
print("\nColumn list:")
for col in final.columns:
    non_null = final[col].notna().sum()
    pct = 100 * non_null / len(final) if len(final) else 0
    print(f"  {col:<30} {non_null:>7,} filled  ({pct:.1f}%)")

# ── Split: complete vs incomplete ─────────────────────────────────────────────

ALL_COLS = list(final.columns)

def _is_blank(val) -> bool:
    if pd.isna(val):
        return True
    return str(val).strip() == ""

any_blank_mask = final[ALL_COLS].apply(
    lambda row: any(_is_blank(v) for v in row), axis=1
)

complete   = final[~any_blank_mask].reset_index(drop=True)
incomplete = final[any_blank_mask].reset_index(drop=True)

COMPLETE_FILE   = OUTPUT_DIR / "final_complete.csv"
INCOMPLETE_FILE = OUTPUT_DIR / "final_incomplete.csv"

complete.to_csv(COMPLETE_FILE, index=False)
incomplete.to_csv(INCOMPLETE_FILE, index=False)

print(f"\nComplete   (all fields filled) : {len(complete):>7,} rows  ->  {COMPLETE_FILE.name}")
print(f"Incomplete (at least 1 missing): {len(incomplete):>7,} rows  ->  {INCOMPLETE_FILE.name}")
print(f"Total check: {len(complete) + len(incomplete):,} == {len(final):,}")
