"""
Maps raw GSMArena spec dict -> DeviceSpec-compatible flat dict.

Handles parsing of human-readable strings like:
  "6.7 inches, ..."        -> display_size_inch = 6.7
  "1080 x 2400 pixels"     -> screen_resolution = 1080
  "Octa-core (1x3.2 GHz)"  -> cpu_cores = 8
  "5000 mAh"               -> battery_mah = 5000
  etc.

Fields NOT available on GSMArena (filled by LLM enrichment):
  antutu_score, cooling_system, price_inr
"""

from __future__ import annotations

import re
from typing import Any, Optional

from app.logger import get_logger
from app.normalizer import (
    classify_chipset_tier,
    classify_gpu_class,
    normalize_wifi,
    normalize_resolution,
    calculate_months_since_launch,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _first_float(text: str) -> Optional[float]:
    """Extract the first float/int from a string."""
    m = re.search(r"(\d+\.?\d*)", text)
    return float(m.group(1)) if m else None


def _first_int(text: str) -> Optional[int]:
    """Extract the first integer from a string."""
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else None


def _parse_display_size(text: str) -> Optional[float]:
    """Parse '6.7 inches, 107.4 cm2 ...' -> 6.7"""
    m = re.search(r"([\d.]+)\s*inch", text, re.I)
    if m:
        return float(m.group(1))
    return _first_float(text)


def _parse_resolution(text: str) -> Optional[int]:
    """
    Parse '1080 x 2400 pixels, ...' -> 1080 (the lower of the two dimensions,
    which is the width / short side for portrait phones).
    """
    nums = [int(n) for n in re.findall(r"\d+", text) if 200 < int(n) < 10000]
    if not nums:
        return None
    # Take the smaller of the two main dimensions (portrait width)
    nums_sorted = sorted(nums)[:2]
    if len(nums_sorted) == 2:
        return normalize_resolution(str(min(nums_sorted)))
    return normalize_resolution(str(nums_sorted[0]))


def _parse_refresh_rate(text: str) -> Optional[int]:
    """Parse '120Hz, 1-120Hz adaptive' -> 120"""
    m = re.search(r"(\d+)\s*[Hh]z", text)
    return int(m.group(1)) if m else None


def _parse_camera_mp(text: str) -> Optional[int]:
    """
    Parse camera spec string and return total MP.

    Handles:
      "50 MP" -> 50
      "50 MP (wide) + 12 MP (ultrawide) + 10 MP (telephoto)" -> 72
      "Triple 50+8+2 MP" -> 60
    """
    # Sum all MP values found in the string
    mp_values = [int(m) for m in re.findall(r"(\d+)\s*MP", text, re.I)]
    if mp_values:
        return sum(mp_values)
    # Fallback: sum numbers before "+" separators that look like MP values
    plus_nums = [int(n) for n in re.findall(r"(\d+)\s*\+", text) if 0 < int(n) < 300]
    if plus_nums:
        return sum(plus_nums)
    return None


def _parse_front_camera_mp(text: str) -> Optional[int]:
    """Parse front camera — take the first (primary) MP value only."""
    m = re.search(r"(\d+)\s*MP", text, re.I)
    return int(m.group(1)) if m else None


def _parse_cpu_cores(text: str) -> Optional[int]:
    """
    Parse CPU core count.

    Handles:
      "Octa-core ..." -> 8
      "8x Cortex-A55" -> 8
      "4+4 core" -> 8
      "Quad-core" -> 4
    """
    name_map = {
        "single": 1, "dual": 2, "quad": 4, "hexa": 6,
        "octa": 8, "deca": 10,
    }
    text_l = text.lower()
    for name, count in name_map.items():
        if name in text_l:
            return count

    # "4+4" or "1+3+4" style
    parts = re.findall(r"(\d+)\s*\+", text)
    if parts:
        return sum(int(p) for p in parts)

    # Plain number before "core" or "x"
    m = re.search(r"(\d+)\s*x\b", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*-?\s*core", text, re.I)
    if m:
        return int(m.group(1))

    return None


def _parse_ram_storage(text: str) -> tuple[Optional[int], Optional[int]]:
    """
    Parse 'Internal' memory spec.

    Returns (ram_gb, storage_gb) for the base variant.

    Handles:
      "128GB 6GB RAM"                       -> (6, 128)
      "64GB 4GB RAM, 128GB 6GB RAM"         -> (4, 64)  # base variant
      "6 GB RAM, 128 GB"                    -> (6, 128)
      "256 GB, 8 GB RAM"                    -> (8, 256)
    """
    # Find all (storage, ram) or (ram, storage) pairs
    # Pattern: "NNN GB NNN GB RAM" or "NNN GB RAM, NNN GB"
    all_variants: list[tuple[int, int]] = []

    # Try to find pairs of storage + ram
    # GSMArena often formats: "128GB 6GB RAM" or "8 GB RAM 256 GB"
    storage_matches = re.findall(r"(\d+)\s*GB(?!\s*RAM)", text, re.I)
    ram_matches = re.findall(r"(\d+)\s*GB\s*RAM", text, re.I)

    if not ram_matches:
        # Some older entries: "4 GB RAM" without matching storage
        ram_matches = re.findall(r"(\d+)\s*GB", text, re.I)
        storage_matches = []

    ram_gb = int(ram_matches[0]) if ram_matches else None
    storage_gb = int(storage_matches[0]) if storage_matches else None

    # Sanity check: RAM rarely exceeds 24 GB, storage rarely < 4 GB
    if ram_gb and ram_gb > 128:
        # Might have swapped — swap back if storage makes more sense
        if storage_gb and storage_gb <= 128:
            ram_gb, storage_gb = storage_gb, ram_gb

    return ram_gb, storage_gb


def _parse_battery(text: str) -> Optional[int]:
    """Parse '5000 mAh, non-removable' -> 5000"""
    m = re.search(r"(\d{3,5})\s*mAh", text, re.I)
    return int(m.group(1)) if m else None


def _parse_wifi(text: str) -> Optional[str]:
    """
    Parse WiFi spec.

    First tries the normalizer rules (which handle "Wi-Fi 7", "Wi-Fi 6e" etc.).
    Then handles GSMArena's slash-separated format: "802.11 a/b/g/n/ac/6e".
    """
    result = normalize_wifi(text)
    if result:
        return result

    # GSMArena slash-separated: "802.11 a/b/g/n/ac/6e, ..."
    t = text.lower().replace(" ", "")
    # Check from highest to lowest standard
    if "be" in t or "wifi7" in t:
        return "wifi 7"
    if "/6e" in t or "6e" in t:
        return "wifi 6e"
    if "/ax" in t or "wifi6" in t or "/6," in t:
        return "wifi 6"
    if "/ac" in t or "wifi5" in t:
        return "wifi 5"
    return None


def _parse_nfc(text: str) -> Optional[int]:
    """Parse NFC presence. Returns 1 if present, 0 if explicitly absent, None if unknown."""
    t = text.strip().lower()
    if not t or t in ("no", "n/a", "-"):
        return 0
    if "yes" in t or "nfc" in t:
        return 1
    return None


def _parse_5g(network_tech: str) -> Optional[int]:
    """Parse 5G from network technology field (e.g. 'GSM / HSPA / LTE / 5G')."""
    return 1 if "5g" in network_tech.lower() else 0


def _parse_launch_date(announced: str) -> Optional[str]:
    """
    Parse GSMArena 'Announced' field to YYYY-MM format.

    Handles:
      "2024, January"         -> "2024-01"
      "2024, January 15"      -> "2024-01"
      "Q1 2024"               -> "2024-01"
      "2024"                  -> "2024-01"   (January as default)
    """
    month_map = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "jun": "06", "jul": "07", "aug": "08", "sep": "09",
        "oct": "10", "nov": "11", "dec": "12",
    }
    quarter_map = {"q1": "01", "q2": "04", "q3": "07", "q4": "10"}

    text = announced.strip().lower()

    # "2024, January" or "January 2024"
    year_m = re.search(r"\b(\d{4})\b", text)
    if not year_m:
        return None
    year = year_m.group(1)

    for name, code in month_map.items():
        if name in text:
            return f"{year}-{code}"

    for qname, qcode in quarter_map.items():
        if qname in text:
            return f"{year}-{qcode}"

    return f"{year}-01"


def _parse_chipset(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse chipset field.

    Returns (chipset_name, cpu_gpu_string).

    GSMArena format: "Qualcomm SM8550-AB Snapdragon 8 Gen 2 (4 nm)"
    We extract the marketing name after the model number.
    """
    # Strip parenthetical suffixes like "(4 nm)" "(6 nm)"
    clean = re.sub(r"\([^)]*nm[^)]*\)", "", text).strip()
    clean = re.sub(r"\([^)]*\)", "", clean).strip()

    # Try to identify the marketing chipset name:
    # Qualcomm: "Snapdragon 8 Gen 2"
    # MediaTek: "Dimensity 9200", "Helio G99"
    # Apple: "A17 Pro"
    # Samsung: "Exynos 2400"
    # Kirin / HiSilicon
    patterns = [
        re.compile(r"(snapdragon[\s\w+]+)", re.I),
        re.compile(r"(dimensity\s+[\w+]+)", re.I),
        re.compile(r"(helio\s+[\w+]+)", re.I),
        re.compile(r"(exynos\s+[\w+]+)", re.I),
        re.compile(r"(kirin\s+[\w+]+)", re.I),
        re.compile(r"(tensor\s+[\w+]+)", re.I),
        re.compile(r"(apple\s+[am]\d+[\w+]*)", re.I),
        re.compile(r"(unisoc\s+[\w+]+)", re.I),
        re.compile(r"(tiger\s+[t]\d+)", re.I),
    ]
    for pat in patterns:
        m = pat.search(clean)
        if m:
            chipset_name = m.group(1).strip().rstrip(",")
            return chipset_name, None  # GPU added separately from the GPU field

    # Fallback: use full cleaned text
    chipset_name = clean.rstrip(",").strip()
    return (chipset_name if chipset_name else None), None


# ---------------------------------------------------------------------------
# Main mapping function
# ---------------------------------------------------------------------------


def map_gsmarena_specs(
    specs: dict[str, dict[str, str]],
    original_brand: str,
    original_model: str,
) -> dict[str, Any]:
    """
    Map a parsed GSMArena spec dict to a flat field dict compatible with DeviceSpec.

    Parameters
    ----------
    specs:
        Output from ``parse_spec_page``.
    original_brand, original_model:
        The brand/model from our input (used as fallback for device_manufacturer/model).

    Returns
    -------
    dict[str, Any]
        Flat dict with DeviceSpec field names as keys. Fields not available on
        GSMArena (antutu_score, cooling_system, price_inr) are set to None.
    """
    def get(section: str, *labels: str) -> Optional[str]:
        """Get first matching label from a section (case-insensitive)."""
        sec = specs.get(section, {})
        for label in labels:
            for k, v in sec.items():
                if k.strip().lower() == label.lower():
                    return v
        return None

    def get_by_data_spec(section: str, data_spec: str) -> Optional[str]:
        """
        Some GSMArena fields use a data-spec attribute as the canonical identifier.
        We store them under their label text, but fall back to checking the value
        content if needed.  This helper checks the section dict for a key matching
        the data_spec attribute value.
        """
        sec = specs.get(section, {})
        for k, v in sec.items():
            if k.lower().replace(" ", "") == data_spec.lower().replace("-", ""):
                return v
        return None

    # --- Device name ---
    meta = specs.get("_meta", {})
    device_title = meta.get("device_title", "")
    device_manufacturer = original_brand
    device_model = original_model
    if device_title:
        parts = device_title.split(" ", 1)
        if len(parts) == 2:
            device_manufacturer = parts[0]
            device_model = parts[1]

    # --- Display ---
    # GSMArena: the "Type" row (first in Display section) contains refresh rate.
    # Example: "Dynamic LTPO AMOLED 2X, 120Hz, HDR10+, 2600 nits"
    display_type_raw = get("Display", "Type")
    display_raw = get("Display", "Size", "Screen size")
    display_size_inch = _parse_display_size(display_raw) if display_raw else None

    resolution_raw = get("Display", "Resolution")
    screen_resolution = _parse_resolution(resolution_raw) if resolution_raw else None

    # Refresh rate: usually embedded in the "Type" field
    display_refresh_hz = None
    for candidate in [
        get("Display", "Refresh rate", "Refresh Rate"),
        display_type_raw,
        resolution_raw,
    ]:
        if candidate:
            display_refresh_hz = _parse_refresh_rate(candidate)
            if display_refresh_hz:
                break

    # --- Chipset / CPU / GPU ---
    chipset_raw = get("Platform", "Chipset")
    chipset_name, _ = _parse_chipset(chipset_raw) if chipset_raw else (None, None)

    cpu_raw = get("Platform", "CPU")
    cpu_cores = _parse_cpu_cores(cpu_raw) if cpu_raw else None

    gpu_raw = get("Platform", "GPU")

    # Build cpu_gpu string
    cpu_gpu: Optional[str] = None
    if chipset_name and gpu_raw:
        # Take only the first GPU name if there are multiple (region variants separated by -)
        gpu_short = gpu_raw.split(" - ")[0].strip()
        cpu_gpu = f"{chipset_name} / {gpu_short}"
    elif chipset_name:
        cpu_gpu = chipset_name

    chipset_tier = classify_chipset_tier(chipset_name)
    gpu_class = classify_gpu_class(gpu_raw or cpu_gpu)

    # --- Memory ---
    memory_raw = get("Memory", "Internal")
    ram_gb, storage_gb = _parse_ram_storage(memory_raw) if memory_raw else (None, None)

    # --- Camera ---
    # GSMArena main camera section: first row (data-spec="cam1modules") contains
    # the lens descriptors. The row label varies ("Single", "Triple", "Quad", etc.)
    # but also might be just the mp count as the label. We search all values
    # in the section for MP patterns.
    def extract_camera_mp_from_section(section_name: str) -> Optional[int]:
        sec = specs.get(section_name, {})
        # Try labeled rows first (Single, Dual, Triple, Quad, Penta)
        for label in ("Single", "Dual", "Triple", "Quad", "Penta", "Main"):
            val = sec.get(label)
            if val:
                mp = _parse_camera_mp(val)
                if mp:
                    return mp
        # Try the first value in the section that contains "MP"
        for k, v in sec.items():
            if "MP" in v or "mp" in v:
                mp = _parse_camera_mp(v)
                if mp:
                    return mp
        return None

    # Also try to find the camera section value directly (cam1modules data-spec)
    # by searching all values in Main Camera for MP patterns
    back_camera_mp_total = (
        extract_camera_mp_from_section("Main Camera")
        or extract_camera_mp_from_section("Camera")
        or extract_camera_mp_from_section("Rear Camera")
    )

    front_section_names = ["Selfie camera", "Front Camera", "Front", "Selfie Camera"]
    front_camera_mp = None
    for sec_name in front_section_names:
        front_camera_mp = extract_camera_mp_from_section(sec_name)
        if front_camera_mp:
            # Front camera: take first/primary lens only, not sum
            sec = specs.get(sec_name, {})
            for k, v in sec.items():
                m = re.search(r"(\d+)\s*MP", v, re.I)
                if m:
                    front_camera_mp = int(m.group(1))
                    break
            break

    # --- Battery ---
    # "Type" row: "Li-Ion 4000 mAh, non-removable"
    battery_raw = get("Battery", "Type", "Capacity")
    battery_mah = _parse_battery(battery_raw) if battery_raw else None

    # --- Connectivity ---
    # WiFi: labeled "WLAN" (not "Wi-Fi") on GSMArena
    wifi_raw = get("Comms", "WLAN", "Wi-Fi", "WiFi", "wlan")
    wifi = _parse_wifi(wifi_raw) if wifi_raw else None

    nfc_raw = get("Comms", "NFC")
    nfc = _parse_nfc(nfc_raw) if nfc_raw is not None else None

    # 5G: detect from "5G bands" presence in Network section
    network_section = specs.get("Network", {})
    five_g_supported = 1 if any("5G" in k or "5g" in k.lower() for k in network_section) else 0

    # Speed field can also confirm 5G
    speed_raw = get("Network", "Speed", "Technology")
    if speed_raw and "5G" in speed_raw:
        five_g_supported = 1

    # --- Launch date ---
    # GSMArena uses "Status" field: "Available. Released 2024, January 24"
    # or "Announced" field: "2024, January"
    announced_raw = get("Launch", "Announced") or get("Launch", "Status")
    launch_date = _parse_launch_date(announced_raw) if announced_raw else None
    months_since_launch = calculate_months_since_launch(launch_date)

    # --- Price INR ---
    # GSMArena Misc section has a "Price" field: "₹ 45,000 / $ 268.99 / ..."
    price_raw = get("Misc", "Price")
    price_inr: Optional[int] = None
    if price_raw:
        # Extract INR price: matches ₹[space]45,000 or ₹45000
        m = re.search(r"[₹]\s*[\u2009]?\s*([\d,]+)", price_raw)
        if m:
            try:
                price_inr = int(m.group(1).replace(",", "").replace("\u2009", ""))
            except ValueError:
                pass

    result = {
        "device_manufacturer": device_manufacturer,
        "device_model": device_model,
        "display_size_inch": display_size_inch,
        "screen_resolution": screen_resolution,
        "display_refresh_hz": display_refresh_hz,
        "price_inr": price_inr,     # From Misc > Price if available
        "back_camera_mp_total": back_camera_mp_total,
        "front_camera_mp": front_camera_mp,
        "cpu_gpu": cpu_gpu,
        "chipset": chipset_name,
        "chipset_tier": chipset_tier,
        "cpu_cores": cpu_cores,
        "gpu_class": gpu_class,
        "antutu_score": None,       # Not on GSMArena — filled by LLM enrichment
        "ram_gb": ram_gb,
        "storage_gb": storage_gb,
        "battery_mah": battery_mah,
        "cooling_system": None,     # Not on GSMArena — filled by LLM enrichment
        "wifi": wifi,
        "nfc": nfc,
        "five_g_supported": five_g_supported,
        "launch_date": launch_date,
        "months_since_launch": months_since_launch,
    }

    logger.debug(
        "Mapped %s %s: chipset=%s tier=%s ram=%s storage=%s battery=%s",
        device_manufacturer,
        device_model,
        chipset_name,
        chipset_tier,
        ram_gb,
        storage_gb,
        battery_mah,
    )
    return result
