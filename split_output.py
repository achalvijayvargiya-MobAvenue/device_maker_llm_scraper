"""
Split final_combined_output.csv into:
  - final_complete.csv   : rows where every column has a value
  - final_incomplete.csv : rows where at least one column is missing
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data/output")

FINAL           = OUTPUT_DIR / "final_combined_output.csv"
COMPLETE_FILE   = OUTPUT_DIR / "final_complete.csv"
INCOMPLETE_FILE = OUTPUT_DIR / "final_incomplete.csv"

# ── Load ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(FINAL)
print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")

# ── Split ─────────────────────────────────────────────────────────────────────

def _is_blank(val) -> bool:
    if pd.isna(val):
        return True
    return str(val).strip() == ""

any_blank_mask = df.apply(lambda row: any(_is_blank(v) for v in row), axis=1)

complete   = df[~any_blank_mask].reset_index(drop=True)
incomplete = df[any_blank_mask].reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────

complete.to_csv(COMPLETE_FILE, index=False)
incomplete.to_csv(INCOMPLETE_FILE, index=False)

print(f"\nComplete   (all fields filled) : {len(complete):>7,} rows  ->  {COMPLETE_FILE.name}")
print(f"Incomplete (at least 1 missing): {len(incomplete):>7,} rows  ->  {INCOMPLETE_FILE.name}")
print(f"Total: {len(complete) + len(incomplete):,}  (matches input: {len(complete) + len(incomplete) == len(df)})")

# ── Show which columns are causing most incompleteness ───────────────────────
if len(incomplete):
    print("\nTop columns with missing values (in incomplete file):")
    missing_counts = incomplete.apply(
        lambda col: col.apply(_is_blank).sum()
    ).sort_values(ascending=False)
    for col, cnt in missing_counts[missing_counts > 0].items():
        pct = 100 * cnt / len(incomplete)
        print(f"  {col:<30} {cnt:>6,} missing  ({pct:.1f}%)")
