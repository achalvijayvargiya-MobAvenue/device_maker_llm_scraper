"""
Utility functions:
- JSON / CSV output writers
- Cache persistence (disk-backed JSON)
- Checkpoint persistence
- Input file loaders (CSV / JSON)
- Cost estimation helper
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from app.models import DeviceInput, DeviceSpec, RunSummary

logger = logging.getLogger(__name__)

# Ordered output columns — existing table columns first (preserved), then all
# additions from Feature_Example.txt.  5g_supported uses the aliased key name.
_CSV_COLUMNS = [
    # Display
    "display_size_inch",
    "screen_resolution",
    "display_refresh_hz",
    # Pricing
    "price_inr",
    # Camera
    "back_camera_mp_total",
    "front_camera_mp",
    # Processor / Performance
    "cpu_gpu",
    "chipset",
    "chipset_tier",
    "cpu_cores",
    "gpu_class",
    "antutu_score",
    # Memory
    "ram_gb",
    "storage_gb",
    # Battery & Thermal
    "battery_mah",
    "cooling_system",
    # Connectivity
    "wifi",
    "nfc",
    "5g_supported",
    # Launch
    "launch_date",
    "months_since_launch",
    # Identifiers (mapped from input brand/model)
    "device_manufacturer",
    "device_model",
]


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_json(specs: list[DeviceSpec], path: Path) -> None:
    """
    Serialize a list of DeviceSpec objects to a JSON file.

    Parameters
    ----------
    specs:
        Normalized device specifications.
    path:
        Destination file path (created / overwritten).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [s.to_flat_dict() for s in specs]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)
    logger.info("JSON output written to %s (%d records)", path, len(records))


def write_csv(specs: list[DeviceSpec], path: Path) -> None:
    """
    Write a list of DeviceSpec objects to a CSV file.

    Parameters
    ----------
    specs:
        Normalized device specifications.
    path:
        Destination file path (created / overwritten).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [s.to_flat_dict() for s in specs]
    df = pd.DataFrame(records, columns=_CSV_COLUMNS)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("CSV output written to %s (%d rows)", path, len(df))


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------


def load_devices_from_csv(path: Path) -> list[DeviceInput]:
    """
    Load device list from a CSV file.

    Expected columns: brand, model (case-insensitive, extra columns ignored).

    Parameters
    ----------
    path:
        Path to the input CSV.

    Returns
    -------
    list[DeviceInput]
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    if "brand" not in df.columns or "model" not in df.columns:
        raise ValueError(
            f"Input CSV must have 'brand' and 'model' columns. Found: {list(df.columns)}"
        )

    devices = [
        DeviceInput(brand=row["brand"], model=row["model"])
        for _, row in df.iterrows()
        if pd.notna(row["brand"]) and pd.notna(row["model"])
    ]
    logger.info("Loaded %d devices from %s", len(devices), path)
    return devices


def load_devices_from_json(path: Path) -> list[DeviceInput]:
    """
    Load device list from a JSON file (array of {brand, model} objects).

    Parameters
    ----------
    path:
        Path to the input JSON.

    Returns
    -------
    list[DeviceInput]
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a top-level array.")

    devices = [DeviceInput(**item) for item in data]
    logger.info("Loaded %d devices from %s", len(devices), path)
    return devices


def load_devices(path: Path) -> list[DeviceInput]:
    """
    Auto-detect file type and load device list.

    Supports .csv and .json extensions.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_devices_from_csv(path)
    if suffix == ".json":
        return load_devices_from_json(path)
    raise ValueError(f"Unsupported input file type: {suffix}. Use .csv or .json")


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------


def load_cache(cache_file: Path) -> dict[str, Any]:
    """Load the LLM response cache from disk; return empty dict if not found."""
    if cache_file.exists():
        try:
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            logger.debug("Cache loaded: %d entries from %s", len(data), cache_file)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load cache (%s); starting fresh.", exc)
    return {}


def save_cache(cache_file: Path, cache: dict[str, Any]) -> None:
    """Persist the LLM response cache to disk atomically."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
    tmp.replace(cache_file)
    logger.debug("Cache saved: %d entries to %s", len(cache), cache_file)


# ---------------------------------------------------------------------------
# Checkpoint persistence
# ---------------------------------------------------------------------------


def load_checkpoint(checkpoint_file: Path) -> dict[str, Any]:
    """Load resume checkpoint from disk; return empty structure if not found."""
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, encoding="utf-8") as f:
                data = json.load(f)
            completed = len(data.get("completed", {}))
            failed = len(data.get("failed", {}))
            logger.info(
                "Checkpoint loaded: %d completed, %d failed batches",
                completed,
                failed,
            )
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load checkpoint (%s); starting fresh.", exc)
    return {"completed": {}, "failed": {}}


def save_checkpoint(checkpoint_file: Path, checkpoint: dict[str, Any]) -> None:
    """Persist run checkpoint atomically."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    tmp.replace(checkpoint_file)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    cost_per_1k_input: float,
    cost_per_1k_output: float,
) -> float:
    """
    Estimate USD cost for a given token count.

    Parameters
    ----------
    input_tokens, output_tokens:
        Actual token counts from the API response.
    cost_per_1k_input, cost_per_1k_output:
        Price per 1 000 tokens (from Settings).

    Returns
    -------
    float
        Estimated cost in USD.
    """
    return (input_tokens / 1000.0) * cost_per_1k_input + (
        output_tokens / 1000.0
    ) * cost_per_1k_output


def print_summary(summary: RunSummary) -> None:
    """Print a human-readable run summary to stdout."""
    sep = "=" * 50
    print(sep)
    print("  EXTRACTION RUN SUMMARY")
    print(sep)
    print(f"  Total devices        : {summary.total_devices}")
    print(f"  Successful           : {summary.successful_extractions}")
    print(f"  Failed               : {summary.failed_extractions}")
    print(f"  Total batches        : {summary.total_batches}")
    print(f"  Failed batches       : {summary.failed_batches}")
    print(f"  Cache hits           : {summary.cache_hits}")
    print(f"  Input tokens         : {summary.total_input_tokens:,}")
    print(f"  Output tokens        : {summary.total_output_tokens:,}")
    print(f"  Estimated cost (USD) : ${summary.estimated_cost_usd:.4f}")
    print(f"  Total latency (ms)   : {summary.total_latency_ms:,.0f}")
    print(sep)
