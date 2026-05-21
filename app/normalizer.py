"""
Post-extraction normalization layer.

Converts raw LLM output (RawDeviceSpec) into a validated DeviceSpec
by applying type coercion, classification, and date math.
All functions are pure and unit-testable.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Optional

from app.models import DeviceSpec, RawDeviceSpec
from app.logger import get_logger

logger = get_logger(__name__)

# Reference date: May 2026 (current month)
_REFERENCE_DATE = date(2026, 5, 1)

# ---------------------------------------------------------------------------
# Chipset → tier classification rules (first match wins)
# ---------------------------------------------------------------------------
_CHIPSET_TIER_RULES: list[tuple[re.Pattern[str], str]] = [
    # --- HIGH ---
    (re.compile(r"snapdragon\s+8\s+gen\s+[0-9]", re.I), "high"),
    (re.compile(r"snapdragon\s+8[0-9]{2}", re.I), "high"),
    (re.compile(r"snapdragon\s+[Xx]\s*(elite|plus)", re.I), "high"),
    (re.compile(r"apple\s+[am]\d+", re.I), "high"),
    (re.compile(r"dimensity\s+9[0-9]{3}", re.I), "high"),
    (re.compile(r"exynos\s+2[0-9]{3}", re.I), "high"),
    (re.compile(r"kirin\s+9[0-9]{2}", re.I), "high"),
    (re.compile(r"tensor\s+g[3-9]", re.I), "high"),          # Google Tensor G3+
    # --- MID ---
    (re.compile(r"snapdragon\s+7s?\s+gen\s+[0-9]", re.I), "mid"),
    (re.compile(r"snapdragon\s+7[0-9]{2}", re.I), "mid"),
    (re.compile(r"snapdragon\s+6\s+gen\s+[0-9]", re.I), "mid"),
    (re.compile(r"snapdragon\s+6[0-9]{2}", re.I), "mid"),
    (re.compile(r"dimensity\s+[78][0-9]{3}", re.I), "mid"),
    (re.compile(r"dimensity\s+6[0-9]{3}", re.I), "mid"),      # Dimensity 6100/6300
    (re.compile(r"helio\s+g[89][0-9]", re.I), "mid"),
    (re.compile(r"helio\s+g[67][0-9]", re.I), "mid"),         # Helio G70/G80/G85
    (re.compile(r"exynos\s+1[0-9]{3}", re.I), "mid"),
    (re.compile(r"kirin\s+[678][0-9]{2}", re.I), "mid"),      # Kirin 710-890
    (re.compile(r"tensor\s+g[12]", re.I), "mid"),             # Google Tensor G1/G2
    # --- LOW ---
    (re.compile(r"snapdragon\s+4\s+gen\s+[0-9]", re.I), "low"),
    (re.compile(r"snapdragon\s+4[0-9]{2}", re.I), "low"),
    (re.compile(r"snapdragon\s+[0-3][0-9]{2}", re.I), "low"),  # SD 200/425/430/450
    (re.compile(r"dimensity\s+[0-9]{3}(?!\d)", re.I), "low"),  # Dimensity 700/810 3-digit
    (re.compile(r"helio\s+g[3-5][0-9]", re.I), "low"),
    (re.compile(r"helio\s+[ae]\d+", re.I), "low"),
    (re.compile(r"helio\s+p\d+", re.I), "low"),
    (re.compile(r"unisoc", re.I), "low"),
    (re.compile(r"sc\d{4}", re.I), "low"),                     # Unisoc SC9863A etc.
    (re.compile(r"tiger\s+t\d+", re.I), "low"),
    (re.compile(r"spreadtrum", re.I), "low"),
]

# ---------------------------------------------------------------------------
# GPU → class classification rules (first match wins)
# ---------------------------------------------------------------------------
_GPU_CLASS_RULES: list[tuple[re.Pattern[str], str]] = [
    # --- HIGH ---
    (re.compile(r"adreno\s+7[4-9]\d", re.I), "high"),
    (re.compile(r"adreno\s+8\d{2}", re.I), "high"),
    (re.compile(r"apple\s+gpu", re.I), "high"),
    (re.compile(r"mali.*(g[7-9]\d|g1[0-9]\d)", re.I), "high"),
    (re.compile(r"immortalis", re.I), "high"),
    # --- MID ---
    (re.compile(r"adreno\s+6[4-9]\d", re.I), "mid"),
    (re.compile(r"adreno\s+7[0-3]\d", re.I), "mid"),
    (re.compile(r"mali.*(g5[2-9]|g[67]\d)", re.I), "mid"),    # Mali G52/G57/G68/G72+
    # --- WEAK ---
    (re.compile(r"adreno\s+[0-5]\d\d", re.I), "weak"),
    (re.compile(r"mali.*(g[0-4]\d|g5[01])", re.I), "weak"),   # Mali G51 and below
    (re.compile(r"mali.*(t[0-9]{3}|mp\d)", re.I), "weak"),    # Mali-T series (legacy)
    (re.compile(r"powervr", re.I), "weak"),
    (re.compile(r"vivante", re.I), "weak"),
]

# ---------------------------------------------------------------------------
# WiFi normalization rules
# ---------------------------------------------------------------------------
_WIFI_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"wi.?fi\s*7|802\.11be", re.I), "wifi 7"),
    (re.compile(r"wi.?fi\s*6e|802\.11ax.*6ghz", re.I), "wifi 6e"),
    (re.compile(r"wi.?fi\s*6|802\.11ax", re.I), "wifi 6"),
    (re.compile(r"wi.?fi\s*5|802\.11ac", re.I), "wifi 5"),
]

# ---------------------------------------------------------------------------
# Resolution → bucket mapping
# ---------------------------------------------------------------------------
_RESOLUTION_BUCKETS: list[tuple[int, int]] = [
    (3840, 2160),
    (2560, 1440),
    (1920, 1080),
    (1280, 720),
]

# ---------------------------------------------------------------------------
# Cooling system canonical values
# ---------------------------------------------------------------------------
_COOLING_MAP: dict[str, str] = {
    "vapor chamber": "Vapor Chamber",
    "vapor-chamber": "Vapor Chamber",
    "vapour chamber": "Vapor Chamber",
    "liquid cooling": "Liquid Cooling",
    "liquid-cooling": "Liquid Cooling",
    "graphite sheet": "Graphite Sheet",
    "graphite": "Graphite Sheet",
    "standard": "Standard",
}


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------


def _to_int(value: Any) -> Optional[int]:
    """Coerce any numeric-ish value to int, return None on failure."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    cleaned = re.sub(r"[^\d.]", "", str(value))
    if cleaned:
        try:
            return int(float(cleaned))
        except ValueError:
            pass
    return None


def _to_float(value: Any) -> Optional[float]:
    """Coerce any numeric-ish value to float."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    cleaned = re.sub(r"[^\d.]", "", str(value))
    if cleaned:
        try:
            return float(cleaned)
        except ValueError:
            pass
    return None


def _to_binary(value: Any) -> Optional[int]:
    """Coerce boolean/int/string to 0 or 1."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value else 0
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "y"):
        return 1
    if s in ("0", "false", "no", "n"):
        return 0
    return None


# ---------------------------------------------------------------------------
# Classification / normalization functions
# ---------------------------------------------------------------------------


def classify_chipset_tier(chipset: Optional[str]) -> Optional[str]:
    """Classify chipset string into 'low', 'mid', or 'high'."""
    if not chipset:
        return None
    for pattern, tier in _CHIPSET_TIER_RULES:
        if pattern.search(chipset):
            return tier
    return None


def classify_gpu_class(gpu_or_chipset: Optional[str]) -> Optional[str]:
    """Classify GPU / chipset string into 'weak', 'mid', or 'high'."""
    if not gpu_or_chipset:
        return None
    for pattern, cls in _GPU_CLASS_RULES:
        if pattern.search(gpu_or_chipset):
            return cls
    return None


def normalize_wifi(wifi: Optional[str]) -> Optional[str]:
    """
    Normalize wifi version to 'wifi 5', 'wifi 6', 'wifi 6e', or 'wifi 7'.
    If the string doesn't match a known pattern, returns it as-is (don't null it).
    """
    if not wifi:
        return None
    for pattern, standard in _WIFI_RULES:
        if pattern.search(wifi):
            return standard
    # Pass through the raw value rather than losing data
    return wifi.strip()


def normalize_resolution(raw: Any) -> Optional[int]:
    """
    Map a raw resolution value to the nearest standard bucket (720/1080/1440/2160).
    If a plain integer is passed (e.g. already bucketed by LLM), return it directly
    rather than forcing it through bucket logic and potentially losing precision.
    """
    if raw is None:
        return None
    # Plain integer already provided by LLM — accept it directly
    if isinstance(raw, int) and raw >= 100:
        return raw
    s = str(raw).lower()
    if "2160" in s or "4k" in s or "uhd" in s:
        return 2160
    if "1440" in s or "qhd" in s or "wqhd" in s:
        return 1440
    if "1080" in s or "fhd" in s:
        return 1080
    if "720" in s or "hd" in s:
        return 720
    nums = [int(n) for n in re.findall(r"\d+", s) if int(n) > 100]
    if not nums:
        return None
    vertical = max(nums)
    for threshold, bucket in _RESOLUTION_BUCKETS:
        if vertical >= threshold * 0.85:
            return bucket
    return 720


def normalize_cooling_system(raw: Optional[str]) -> Optional[str]:
    """
    Normalize cooling system string to canonical form.
    Falls back to the raw string (stripped) rather than discarding unrecognized values.
    """
    if not raw:
        return None
    normalized = _COOLING_MAP.get(raw.strip().lower())
    return normalized if normalized is not None else raw.strip()


def calculate_months_since_launch(launch_date: Optional[str]) -> Optional[int]:
    """
    Calculate months elapsed from launch_date to _REFERENCE_DATE (May 2026).

    Parameters
    ----------
    launch_date:
        String in "YYYY-MM" or "YYYY" format (e.g. "2024-01" or "2024").

    Returns
    -------
    int | None
    """
    if not launch_date:
        return None
    s = str(launch_date).strip()
    try:
        m_full = re.match(r"^(\d{4})-(\d{2})", s)
        m_year = re.match(r"^(\d{4})", s)
        if m_full:
            year, month = int(m_full.group(1)), int(m_full.group(2))
        elif m_year:
            year, month = int(m_year.group(1)), 1
        else:
            return None
        d = date(year, month, 1)
        delta = (
            (_REFERENCE_DATE.year - d.year) * 12
            + (_REFERENCE_DATE.month - d.month)
        )
        return max(0, delta)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main normalizer
# ---------------------------------------------------------------------------


def normalize(raw: RawDeviceSpec) -> DeviceSpec:
    """
    Convert a RawDeviceSpec (permissive LLM output) into a validated DeviceSpec.

    Applies:
    - Type coercion (int, float, binary)
    - Chipset tier / GPU class derivation from chipset name when LLM value is missing
    - WiFi string normalization
    - Resolution bucketing
    - Cooling system canonicalization
    - months_since_launch recalculation from launch_date for consistency

    Parameters
    ----------
    raw:
        RawDeviceSpec parsed from LLM JSON output.

    Returns
    -------
    DeviceSpec
    """
    # --- Numeric coercions ---
    display_size_inch   = _to_float(raw.display_size_inch)
    screen_resolution   = normalize_resolution(raw.screen_resolution)
    display_refresh_hz  = _to_int(raw.display_refresh_hz)
    price_inr           = _to_int(raw.price_inr)
    back_camera_mp_total = _to_int(raw.back_camera_mp_total)
    front_camera_mp     = _to_int(raw.front_camera_mp)
    cpu_cores           = _to_int(raw.cpu_cores)
    antutu_score        = _to_int(raw.antutu_score)
    ram_gb              = _to_int(raw.ram_gb)
    storage_gb          = _to_int(raw.storage_gb)
    battery_mah         = _to_int(raw.battery_mah)
    nfc                 = _to_binary(raw.nfc)
    five_g              = _to_binary(raw.five_g_supported)

    # --- String fields ---
    chipset  = raw.chipset or None
    cpu_gpu  = raw.cpu_gpu or None
    launch_date = raw.launch_date

    # --- Derived / normalized ---
    cooling_system = normalize_cooling_system(raw.cooling_system)
    wifi = normalize_wifi(raw.wifi)

    # chipset_tier: use LLM value if valid, else derive from chipset name
    raw_tier = (raw.chipset_tier or "").strip().lower()
    chipset_tier = (
        raw_tier if raw_tier in ("low", "mid", "high")
        else classify_chipset_tier(chipset)
    )

    # gpu_class: use LLM value if valid, else derive from chipset name (contains GPU name too)
    raw_gpu = (raw.gpu_class or "").strip().lower()
    gpu_class = (
        raw_gpu if raw_gpu in ("weak", "mid", "high")
        else classify_gpu_class(cpu_gpu or chipset)
    )

    # Always recalculate months_since_launch from launch_date for accuracy
    months_since_launch = calculate_months_since_launch(launch_date)

    spec = DeviceSpec(
        display_size_inch=display_size_inch,
        screen_resolution=screen_resolution,
        display_refresh_hz=display_refresh_hz,
        price_inr=price_inr,
        back_camera_mp_total=back_camera_mp_total,
        front_camera_mp=front_camera_mp,
        cpu_gpu=cpu_gpu,
        chipset=chipset,
        chipset_tier=chipset_tier,
        cpu_cores=cpu_cores,
        gpu_class=gpu_class,
        antutu_score=antutu_score,
        ram_gb=ram_gb,
        storage_gb=storage_gb,
        battery_mah=battery_mah,
        cooling_system=cooling_system,
        wifi=wifi,
        nfc=nfc,
        five_g_supported=five_g,
        launch_date=launch_date,
        months_since_launch=months_since_launch,
        device_manufacturer=raw.device_manufacturer,
        device_model=raw.device_model,
    )

    logger.debug(
        "Normalized %s %s → tier=%s gpu=%s wifi=%s res=%s cooling=%s months=%s",
        raw.device_manufacturer, raw.device_model,
        chipset_tier, gpu_class, wifi, screen_resolution,
        cooling_system, months_since_launch,
    )

    return spec


def derive_missing_fields(spec: DeviceSpec) -> DeviceSpec:
    """
    Fill in deterministically-derivable fields that are currently null.

    Applies:
    - ``chipset_tier`` derived from ``chipset`` when not already set.
    - ``gpu_class`` derived from ``cpu_gpu`` (which contains the GPU name) when not set.
    - ``months_since_launch`` recalculated from ``launch_date`` when not set.

    This is useful after GSMArena scraping, where these fields come from
    code classification rather than LLM output.

    Parameters
    ----------
    spec:
        A DeviceSpec that may have null chipset_tier / gpu_class / months_since_launch.

    Returns
    -------
    DeviceSpec
        A new DeviceSpec with derived fields filled where they were None.
    """
    data = spec.model_dump()

    if data.get("chipset_tier") is None and data.get("chipset"):
        data["chipset_tier"] = classify_chipset_tier(data["chipset"])

    if data.get("gpu_class") is None:
        data["gpu_class"] = classify_gpu_class(
            data.get("cpu_gpu") or data.get("chipset")
        )

    if data.get("months_since_launch") is None and data.get("launch_date"):
        data["months_since_launch"] = calculate_months_since_launch(data["launch_date"])

    # Re-validate the wifi field using the normalizer rules in case it's a raw string
    if data.get("wifi"):
        normalized_wifi = normalize_wifi(data["wifi"])
        data["wifi"] = normalized_wifi

    # Rename five_g_supported key if it was serialized as 5g_supported
    if "5g_supported" in data and "five_g_supported" not in data:
        data["five_g_supported"] = data.pop("5g_supported")

    return DeviceSpec(**{k: v for k, v in data.items() if k in DeviceSpec.model_fields})
