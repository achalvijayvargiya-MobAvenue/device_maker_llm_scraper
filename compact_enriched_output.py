"""
Compact final_enriched_output_v2.csv for faster downstream processing.

Column transforms:
  price_inr      -> VL / L / M / H / VH buckets
  antutu_score   -> L / M / H / P buckets
  chipset_tier   -> l / m / h
  gpu_class      -> w / m / h
  wifi           -> 4 / 5 / 6 / 6e / 7
  cooling_system -> S / U / VC / LC / GS

After transforms, every blank / none / null cell in the file becomes 'U'.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("data/output")
IN_FILE = OUTPUT_DIR / "final_enriched_output_v2.csv"
OUT_FILE = OUTPUT_DIR / "final_enriched_output_v3.csv"

CHIPSET_TIER_MAP = {"low": "l", "mid": "m", "high": "h"}
GPU_CLASS_MAP = {"weak": "w", "mid": "m", "high": "h"}
WIFI_MAP = {
    "wifi 4": "4",
    "wifi 5": "5",
    "wifi 6": "6",
    "wifi 6e": "6e",
    "wifi 7": "7",
}
COOLING_MAP = {
    "standard": "S",
    "none": "U",
    "vapor chamber": "VC",
    "liquid cooling": "LC",
    "graphite sheet": "GS",
}

MISSING_TOKENS = frozenset({"", "none", "null", "nan"})


def is_missing(val) -> bool:
    if pd.isna(val):
        return True
    return str(val).strip().lower() in MISSING_TOKENS


def bucket_price_inr(val) -> str:
    if is_missing(val):
        return "U"
    try:
        price = float(val)
    except (TypeError, ValueError):
        return "U"
    if price < 10_000:
        return "VL"
    if price < 25_000:
        return "L"
    if price < 50_000:
        return "M"
    if price < 80_000:
        return "H"
    return "VH"


def bucket_antutu_score(val) -> str:
    if is_missing(val):
        return "U"
    try:
        score = float(val)
    except (TypeError, ValueError):
        return "U"
    if score < 500_000:
        return "L"
    if score < 1_000_000:
        return "M"
    if score < 2_000_000:
        return "H"
    return "P"


def map_chipset_tier(val) -> str:
    if is_missing(val):
        return "U"
    key = str(val).strip().lower()
    return CHIPSET_TIER_MAP.get(key, "U")


def map_gpu_class(val) -> str:
    if is_missing(val):
        return "U"
    key = str(val).strip().lower()
    return GPU_CLASS_MAP.get(key, "U")


def map_wifi(val) -> str:
    if is_missing(val):
        return "U"
    key = str(val).strip().lower()
    return WIFI_MAP.get(key, "U")


def map_cooling_system(val) -> str:
    if is_missing(val):
        return "U"
    key = str(val).strip().lower()
    return COOLING_MAP.get(key, "U")


def replace_missing_with_u(val):
    if is_missing(val):
        return "U"
    return val


def main() -> None:
    print(f"Loading {IN_FILE}...")
    df = pd.read_csv(IN_FILE)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    df["price_inr"] = df["price_inr"].map(bucket_price_inr)
    df["antutu_score"] = df["antutu_score"].map(bucket_antutu_score)
    df["chipset_tier"] = df["chipset_tier"].map(map_chipset_tier)
    df["gpu_class"] = df["gpu_class"].map(map_gpu_class)
    df["wifi"] = df["wifi"].map(map_wifi)
    df["cooling_system"] = df["cooling_system"].map(map_cooling_system)

    before_u = int(df.apply(lambda row: any(is_missing(v) for v in row), axis=1).sum())
    df = df.map(replace_missing_with_u)
    print(f"  Rows with at least one missing value -> U: {before_u:,}")

    df.to_csv(OUT_FILE, index=False)
    print(f"Saved: {OUT_FILE}")

    for col in ("price_inr", "antutu_score", "chipset_tier", "gpu_class", "wifi", "cooling_system"):
        print(f"\n{col} value counts:")
        print(df[col].value_counts().head(12))


if __name__ == "__main__":
    main()
