"""
Device pre-filter and brand normalization.

Layer 1 of the multi-layer extraction pipeline.

Classifies each DeviceInput as VALID or SKIP based on rule-based heuristics,
and normalizes brand names (e.g. 'lge' -> 'LG') before scraping/LLM lookup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Optional

from app.logger import get_logger
from app.models import DeviceInput

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Skip rules applied to the model string
# ---------------------------------------------------------------------------

_SKIP_MODEL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"[-_\.]test\b", re.I),                       # -TEST, _TEST, .test suffix
    re.compile(r"\btest[-_\.]build\b", re.I),                 # test-build variants
    re.compile(r"^en[-_][a-z]{2}$", re.I),                   # en_us, en_gb locale strings
    re.compile(r"^[0-9\-_\.]+$"),                             # Pure numeric / punctuation
    re.compile(r"^generic$", re.I),                           # Literal "generic"
    re.compile(r"^(null|none|n/a|na|unknown|undefined)$", re.I),  # Null placeholders
    re.compile(r"^\d{4}_\d+[a-z_]+$", re.I),                 # Firmware image names: 2024_65c350ne_toshiba
    re.compile(r"^(adt-3[-_])", re.I),                        # Android TV box internal names
]

# TV / non-mobile detection via model string
_TV_MODEL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bpus\d{4}\b", re.I),           # Philips TV: PUS6162, PUS8505
    re.compile(r"\b4t-c\b", re.I),               # Sharp 4K TV: 4T-C50BN
    re.compile(r"\bc\d{3,4}(ne|nf|eu|us)\b", re.I),  # TV model codes: C350NE
    re.compile(r"_toshiba$", re.I),              # Toshiba TV firmware labels
    re.compile(r"^\d{2}[a-z]{1,2}\d{3,4}[a-z]{1,3}$", re.I),  # 65A8H, 55NANO81 LG TV codes
]

# Brands that are clearly non-mobile / TV platforms
_NON_MOBILE_BRANDS: frozenset[str] = frozenset(
    {
        "vidaa",       # Hisense TV OS
        "adt-3",       # Android TV 3 streaming box
        "starhub",     # Cable/ISP STB
    }
)

# Brands recognised as TV brands when model ALSO looks like a TV model code
_CONDITIONAL_TV_BRANDS: frozenset[str] = frozenset(
    {
        "philips",
        "sharp",
        "toshiba",
        "hisense",
        "skyworth",
    }
)

# ---------------------------------------------------------------------------
# Brand normalization
# ---------------------------------------------------------------------------

# Direct brand code → canonical marketing name
_BRAND_NORM_MAP: dict[str, str] = {
    "lge": "LG",
    "lg electronics": "LG",
    "tct (alcatel)": "Alcatel",
    "tct": "Alcatel",
    "motorola mobility": "Motorola",
    "htc corporation": "HTC",
    "samsung electronics": "Samsung",
    "huawei technologies": "Huawei",
    "xiaomi communications": "Xiaomi",
    "oppo electronics": "OPPO",
    "bbk electronics": "Vivo",
    "lt electronics": "LT",
    "lt": "LT",
    "dixon+india": "Dixon",
    "ace (global)": "Ace",
    "logic mobility": "Logic",
    "tct-alcatel": "Alcatel",
}

# Generic placeholder brand values — real brand should be inferred from model
_GENERIC_BRANDS: frozenset[str] = frozenset(
    {
        "android",
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
        "mobile",
        "nuu",   # NUU is a reseller that often carries other brand phones
    }
)

# Model prefix → inferred real brand  (order matters — first match wins)
_MODEL_PREFIX_TO_BRAND: list[tuple[re.Pattern[str], Optional[str]]] = [
    (re.compile(r"^rmx", re.I),           "Realme"),
    (re.compile(r"^cph", re.I),           "OPPO"),
    (re.compile(r"^sm-", re.I),           "Samsung"),
    (re.compile(r"^lm-", re.I),           "LG"),
    (re.compile(r"^m\d{4}j", re.I),       "Xiaomi"),
    (re.compile(r"^pixel\s", re.I),       "Google"),
    (re.compile(r"^pixel\d", re.I),       "Google"),
    (re.compile(r"^moto\s", re.I),        "Motorola"),
    (re.compile(r"^moto[eg]", re.I),      "Motorola"),
    (re.compile(r"^tecno\s", re.I),       "Tecno"),
    (re.compile(r"^itel\s", re.I),        "itel"),
    (re.compile(r"^infinix\s", re.I),     "Infinix"),
    (re.compile(r"^redmi\s", re.I),       "Redmi"),
    (re.compile(r"^poco\s", re.I),        "POCO"),
    (re.compile(r"^nokia\s", re.I),       "Nokia"),
    (re.compile(r"^mblu\s", re.I),        "Motorola"),
    (re.compile(r"^oneplus\s", re.I),     "OnePlus"),
    (re.compile(r"^realme\s", re.I),      "Realme"),
    (re.compile(r"^vivo\s", re.I),        "Vivo"),
    (re.compile(r"^oppo\s", re.I),        "OPPO"),
    (re.compile(r"^huawei\s", re.I),      "Huawei"),
    (re.compile(r"^honor\s", re.I),       "Honor"),
    (re.compile(r"^hibreak\b", re.I),     "Hibreak"),
    (re.compile(r"^gmc-", re.I),          "GMC"),
    (re.compile(r"^asr\d+", re.I),        None),   # Internal SoC name, not a device
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FilterResult:
    """Outcome of classifying one DeviceInput."""

    device: DeviceInput
    skip: bool
    skip_reason: str = ""
    normalized_brand: str = ""
    normalized_model: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_non_ascii(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


def _normalize_brand(device: DeviceInput) -> tuple[str, str]:
    """
    Return (normalized_brand, normalized_model).

    - Applies _BRAND_NORM_MAP for known internal brand codes.
    - For generic/android brand, attempts to infer real brand from model prefix.
    """
    brand_l = device.brand.strip().lower()
    model = device.model.strip()

    # Direct lookup
    if brand_l in _BRAND_NORM_MAP:
        return _BRAND_NORM_MAP[brand_l], model

    # Generic brand: try to infer from model prefix
    if brand_l in _GENERIC_BRANDS:
        for pattern, inferred in _MODEL_PREFIX_TO_BRAND:
            if pattern.match(model):
                if inferred is None:
                    # Model resolved to "not a real device" — flag for skip upstream
                    return device.brand.strip(), model
                return inferred, model
        # Could not infer — return original brand stripped
        return device.brand.strip(), model

    # No normalization needed
    return device.brand.strip(), model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_device(device: DeviceInput) -> FilterResult:
    """
    Classify a single DeviceInput as VALID or SKIP.

    Checks are applied in priority order (cheapest/most specific first).
    Returns a FilterResult with skip=True and a reason string when skipping.
    """
    brand_l = device.brand.strip().lower()
    model = device.model.strip()

    # Non-ASCII characters (garbled encoding, Chinese chars, etc.)
    if _has_non_ascii(model):
        return FilterResult(device=device, skip=True, skip_reason="non_ascii_model")
    if _has_non_ascii(device.brand):
        return FilterResult(device=device, skip=True, skip_reason="non_ascii_brand")

    # Known non-mobile brand unconditionally
    if brand_l in _NON_MOBILE_BRANDS:
        return FilterResult(
            device=device,
            skip=True,
            skip_reason=f"non_mobile_brand:{device.brand}",
        )

    # Model string skip patterns
    for pattern in _SKIP_MODEL_PATTERNS:
        if pattern.search(model):
            return FilterResult(
                device=device,
                skip=True,
                skip_reason=f"skip_model_pattern:{pattern.pattern}",
            )

    # TV model patterns (brand-independent — some phones have similar-looking codes,
    # but these patterns are specific enough to TV products)
    for pattern in _TV_MODEL_PATTERNS:
        if pattern.search(model):
            return FilterResult(
                device=device,
                skip=True,
                skip_reason=f"tv_model_pattern:{pattern.pattern}",
            )

    # Conditional TV check: brand in conditional set AND model has TV-like structure
    if brand_l in _CONDITIONAL_TV_BRANDS:
        # Short all-uppercase alphanumeric with no vowel clusters = likely TV SKU
        if re.match(r"^[A-Z0-9]{6,12}$", model.replace("-", "").replace(" ", "")):
            return FilterResult(
                device=device,
                skip=True,
                skip_reason=f"conditional_tv:{device.brand}:{model}",
            )

    # Generic brand + model that resolved to "not a real device"
    if brand_l in _GENERIC_BRANDS:
        norm_brand, norm_model = _normalize_brand(device)
        # If brand is still generic after inference, keep it valid — let the
        # scraper / LLM decide; only skip if model itself is a known non-device string
        for pattern in _MODEL_PREFIX_TO_BRAND:
            pat, inferred = pattern
            if pat.match(model) and inferred is None:
                return FilterResult(
                    device=device,
                    skip=True,
                    skip_reason=f"internal_soc_name:{model}",
                )
        return FilterResult(
            device=device,
            skip=False,
            normalized_brand=norm_brand,
            normalized_model=norm_model,
        )

    # Valid — normalize brand
    norm_brand, norm_model = _normalize_brand(device)
    return FilterResult(
        device=device,
        skip=False,
        normalized_brand=norm_brand,
        normalized_model=norm_model,
    )


def filter_devices(
    devices: list[DeviceInput],
) -> tuple[list[DeviceInput], list[dict]]:
    """
    Classify all devices and split into valid vs. skipped groups.

    Parameters
    ----------
    devices:
        Raw input list from CSV/JSON.

    Returns
    -------
    tuple[list[DeviceInput], list[dict]]
        - valid_devices: brand/model normalized, ready for scraping/LLM.
        - skipped_records: dicts with original brand/model + skip_reason, for audit CSV.
    """
    valid: list[DeviceInput] = []
    skipped: list[dict] = []

    for device in devices:
        result = classify_device(device)
        if result.skip:
            skipped.append(
                {
                    "brand": device.brand,
                    "model": device.model,
                    "skip_reason": result.skip_reason,
                }
            )
            logger.debug("SKIP %s %s — %s", device.brand, device.model, result.skip_reason)
        else:
            valid.append(
                DeviceInput(
                    brand=result.normalized_brand or device.brand,
                    model=result.normalized_model or device.model,
                )
            )

    skip_pct = 100.0 * len(skipped) / max(len(devices), 1)
    logger.info(
        "Filter: %d valid, %d skipped (%.1f%% skip rate)",
        len(valid),
        len(skipped),
        skip_pct,
    )
    return valid, skipped
