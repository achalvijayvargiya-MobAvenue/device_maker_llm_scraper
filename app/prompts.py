"""
LLM prompt construction for device specification extraction.

Design principles:
- Temperature=0 reduces hallucination.
- Explicit null instruction ensures unknown fields don't get invented values.
- Schema is embedded in the system prompt so the model knows the exact contract.
- Device list is injected into the user message to separate instruction from data.
- Model-only fallback prompt is used when brand is generic or first pass returns blanks.
"""

from __future__ import annotations

import json
from typing import Any

from app.models import DeviceInput

# Brands that carry no identification value — treat device as model-only query.
GENERIC_BRANDS: frozenset[str] = frozenset(
    {
        "generic",
        "unknown",
        "unbranded",
        "oem",
        "no brand",
        "nobrand",
        "noname",
        "misc",
        "other",
        "n/a",
        "na",
        "brand",
        "device",
    }
)


def is_generic_brand(brand: str) -> bool:
    """Return True if the brand string conveys no useful manufacturer identity."""
    return brand.strip().lower() in GENERIC_BRANDS


def retry_model_string(device: DeviceInput) -> str:
    """
    Build the single search string used in a model-only retry query.

    - Generic brand  → use model name alone   (e.g. "tab15")
    - Known brand    → combine brand + model  (e.g. "blackview tab 80")
      Avoids duplication when model already starts with the brand name.
    """
    if is_generic_brand(device.brand):
        return device.model
    brand_l = device.brand.lower()
    model_l = device.model.lower()
    if model_l.startswith(brand_l):
        return device.model
    return f"{device.brand} {device.model}"


SYSTEM_PROMPT = """\
You are a mobile device specification extraction engine with access to your training knowledge.

YOUR TASK:
Extract accurate technical specifications for each mobile device in the provided list.

STRICT OUTPUT RULES:
1. Return ONLY a valid JSON array — no markdown, no backticks, no explanations.
2. Each element corresponds to one device in the input list, in the same order.
3. Use null ONLY when you have absolutely no information about a field — completeness is the priority.
4. Use your best knowledge and estimate for technical specs. Do NOT invent device_manufacturer or device_model names, but DO fill every technical field you can reasonably estimate.
5. Normalize all values according to the rules below before outputting.

NORMALIZATION RULES:
- display_size_inch: float in inches (e.g., 6.7)
- screen_resolution: normalize to nearest bucket — 720, 1080, 1440, or 2160
- display_refresh_hz: integer Hz only (e.g., 60, 90, 120, 144)
- price_inr: integer, latest official India launch price — numeric only, no symbol
- back_camera_mp_total: integer — sum of all rear camera megapixels
  (e.g., 50 + 12 + 10 → 72; single camera 108 → 108)
- front_camera_mp: integer — front/selfie camera megapixels (primary lens only)
- cpu_gpu: string — chipset name + GPU name separated by " / "
  (e.g., "Snapdragon 8 Gen 3 / Adreno 750", "Apple A17 Pro / Apple GPU")
- chipset: exact marketing name only (e.g., "Snapdragon 8 Gen 3", "Dimensity 9300")
- chipset_tier: exactly one of "low", "mid", "high"
  • high → Snapdragon 8-series, Apple A-series, Dimensity 9000+, Exynos 2xxx, Kirin 9xx
  • mid  → Snapdragon 6/7-series, Dimensity 7000-series, Helio G80+, Exynos 1xxx
  • low  → Snapdragon 4-series, Helio G35/G85, Unisoc, Tiger, entry-level chips
- cpu_cores: integer (e.g., 4, 6, 8)
- gpu_class: exactly one of "weak", "mid", "high"
  • high → Adreno 7xx/8xx, Apple GPU, Immortalis, Mali G7xx+
  • mid  → Adreno 6xx, Mali G6xx/G7x
  • weak → Adreno 5xx and below, Mali G5x and below, PowerVR
- antutu_score: integer — approximate AnTuTu v10 benchmark score
- ram_gb: integer only (e.g., 8, 12, 16) — use base/standard variant
- storage_gb: integer only (e.g., 128, 256) — use base/standard variant
- battery_mah: integer mAh only (e.g., 5000)
- cooling_system: one of "Vapor Chamber", "Liquid Cooling", "Graphite Sheet", "Standard", null
  • Vapor Chamber → flagship/gaming phones with large vapor chamber
  • Liquid Cooling → mid-range with liquid cooling pipes
  • Graphite Sheet → budget/mid-range with graphite thermal pad
  • Standard → basic thermal management, no special system
- wifi: exactly one of "wifi 5", "wifi 6", "wifi 6e", "wifi 7"
- nfc: 0 or 1 only
- 5g_supported: 0 or 1 only
- launch_date: string in "YYYY-MM" format (e.g., "2024-01") — month of official announcement
- months_since_launch: integer — months from launch_date to May 2026
- device_manufacturer: official brand name (e.g., "Samsung", "Apple")
- device_model: official model name (e.g., "Galaxy S24 Ultra", "iPhone 15 Pro")

REQUIRED OUTPUT SCHEMA (one object per device, preserve input order):
[
  {
    "device_manufacturer": "string",
    "device_model": "string",
    "display_size_inch": float | null,
    "screen_resolution": 720 | 1080 | 1440 | 2160 | null,
    "display_refresh_hz": integer | null,
    "price_inr": integer | null,
    "back_camera_mp_total": integer | null,
    "front_camera_mp": integer | null,
    "cpu_gpu": "string" | null,
    "chipset": "string" | null,
    "chipset_tier": "low" | "mid" | "high" | null,
    "cpu_cores": integer | null,
    "gpu_class": "weak" | "mid" | "high" | null,
    "antutu_score": integer | null,
    "ram_gb": integer | null,
    "storage_gb": integer | null,
    "battery_mah": integer | null,
    "cooling_system": "Vapor Chamber" | "Liquid Cooling" | "Graphite Sheet" | "Standard" | null,
    "wifi": "wifi 5" | "wifi 6" | "wifi 6e" | "wifi 7" | null,
    "nfc": 0 | 1 | null,
    "5g_supported": 0 | 1 | null,
    "launch_date": "YYYY-MM" | null,
    "months_since_launch": integer | null
  }
]
"""


def build_user_message(devices: list[DeviceInput]) -> str:
    """
    Build the user-turn message that lists devices to extract.

    Parameters
    ----------
    devices:
        Batch of DeviceInput objects.

    Returns
    -------
    str
        Formatted user message to send as the ``user`` role.
    """
    device_list = [{"brand": d.brand, "model": d.model} for d in devices]
    devices_json = json.dumps(device_list, indent=2, ensure_ascii=False)
    return (
        f"Extract specifications for the following {len(devices)} device(s):\n\n"
        f"{devices_json}\n\n"
        "Return ONLY the JSON array. No other text."
    )


def build_messages(devices: list[DeviceInput]) -> list[dict[str, Any]]:
    """
    Construct the full OpenAI messages payload for a batch.

    Parameters
    ----------
    devices:
        Batch of DeviceInput objects to extract specs for.

    Returns
    -------
    list[dict]
        OpenAI-compatible messages list.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(devices)},
    ]


def build_model_only_user_message(devices: list[DeviceInput]) -> str:
    """
    Build the user message for a model-only retry pass.

    Sends only a ``model`` field per device (no brand), using
    :func:`retry_model_string` to construct each search string.
    This helps the LLM find data when brand is generic or unhelpful.

    Parameters
    ----------
    devices:
        Devices to retry. Each entry contributes one search string.

    Returns
    -------
    str
    """
    device_list = [{"model": retry_model_string(d)} for d in devices]
    devices_json = json.dumps(device_list, indent=2, ensure_ascii=False)
    return (
        f"Extract specifications for the following {len(devices)} device(s). "
        "Identify each device using ONLY the model string — ignore any brand context:\n\n"
        f"{devices_json}\n\n"
        "Return ONLY the JSON array. No other text."
    )


def build_model_only_messages(devices: list[DeviceInput]) -> list[dict[str, Any]]:
    """
    Full messages payload for a model-only retry batch.

    Parameters
    ----------
    devices:
        Devices to query with model-only search strings.

    Returns
    -------
    list[dict]
        OpenAI-compatible messages list.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_model_only_user_message(devices)},
    ]


ENRICHMENT_SYSTEM_PROMPT = """\
You are a mobile device specification expert. You will be given a list of devices
with their chipset names already known. Your job is to fill in THREE specific fields
that cannot be scraped from spec sheets:

1. antutu_score  — approximate AnTuTu v10 benchmark score (integer). Use the
   typical/average score for the chipset. Return null if genuinely unknown.
2. cooling_system — exactly one of:
   "Vapor Chamber", "Liquid Cooling", "Graphite Sheet", "Standard", or null.
   • Vapor Chamber  → flagship/gaming phones with large vapor chamber heat spreader
   • Liquid Cooling → mid-range devices with liquid cooling pipes
   • Graphite Sheet → budget/mid with graphite thermal pad
   • Standard       → basic thermal management, no special system
   • null           → truly unknown
3. price_inr — official India launch price in INR (integer, no symbol). Return null
   if the device was never officially sold in India or price is unknown.

STRICT OUTPUT RULES:
1. Return ONLY a valid JSON array — no markdown, no backticks, no explanations.
2. Each element corresponds to one device in the input list, in the same order.
3. Use null (JSON null) for any field you are not certain about.
4. Do NOT guess antutu scores — only return them if you know the chipset's typical score.

REQUIRED OUTPUT SCHEMA:
[
  {
    "device_manufacturer": "string",
    "device_model": "string",
    "antutu_score": integer | null,
    "cooling_system": "Vapor Chamber" | "Liquid Cooling" | "Graphite Sheet" | "Standard" | null,
    "price_inr": integer | null
  }
]
"""


def build_enrichment_user_message(devices: list) -> str:
    """
    Build the user message for an enrichment-only LLM pass.

    Parameters
    ----------
    devices:
        List of dicts with keys: brand, model, chipset (can be None).

    Returns
    -------
    str
    """
    device_list = [
        {
            "brand": d.get("brand", ""),
            "model": d.get("model", ""),
            "chipset": d.get("chipset") or "unknown",
        }
        for d in devices
    ]
    devices_json = json.dumps(device_list, indent=2, ensure_ascii=False)
    return (
        f"Fill in antutu_score, cooling_system, and price_inr for the following "
        f"{len(devices)} device(s):\n\n"
        f"{devices_json}\n\n"
        "Return ONLY the JSON array. No other text."
    )


def build_enrichment_messages(devices: list) -> list[dict]:
    """Full messages payload for the enrichment pass."""
    return [
        {"role": "system", "content": ENRICHMENT_SYSTEM_PROMPT},
        {"role": "user", "content": build_enrichment_user_message(devices)},
    ]


def build_repair_messages(
    original_devices: list[DeviceInput],
    malformed_response: str,
    error_detail: str,
) -> list[dict[str, Any]]:
    """
    Build a follow-up repair prompt when the initial response was not valid JSON.

    Parameters
    ----------
    original_devices:
        The devices that were originally queried.
    malformed_response:
        The invalid LLM output that needs correction.
    error_detail:
        The JSON parse error message for context.

    Returns
    -------
    list[dict]
        Repair messages to send as a new completion call.
    """
    repair_instruction = (
        "Your previous response was not valid JSON.\n"
        f"Parse error: {error_detail}\n\n"
        "Previous response:\n"
        f"{malformed_response}\n\n"
        "Fix the JSON and return ONLY the corrected JSON array. No other text."
    )
    messages = build_messages(original_devices)
    messages.append({"role": "assistant", "content": malformed_response})
    messages.append({"role": "user", "content": repair_instruction})
    return messages
