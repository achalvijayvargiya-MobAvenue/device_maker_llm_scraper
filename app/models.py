"""
Pydantic v2 data models for device input, raw LLM output, and normalized output.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Input  (unchanged — still uses brand/model as source column names)
# ---------------------------------------------------------------------------


class DeviceInput(BaseModel):
    """A single device lookup request sourced from the input CSV/JSON."""

    brand: str = Field(..., min_length=1, description="Device manufacturer name")
    model: str = Field(..., min_length=1, description="Device model / version string")

    @field_validator("brand", "model", mode="before")
    @classmethod
    def _strip(cls, v: str) -> str:
        return str(v).strip()


# ---------------------------------------------------------------------------
# Raw LLM extraction output — permissive, accepts string numbers etc.
# ---------------------------------------------------------------------------


class RawDeviceSpec(BaseModel):
    """
    Schema used to parse direct LLM JSON output.
    All fields are optional so partial responses don't cause hard failures.
    Field names match exactly what the LLM is instructed to return.
    """

    # Identifiers
    device_manufacturer: Optional[str] = None
    device_model: Optional[str] = None

    # Display
    display_size_inch: Optional[float | str] = None
    screen_resolution: Optional[int | str] = None
    display_refresh_hz: Optional[int | str] = None

    # Pricing
    price_inr: Optional[int | str | float] = None

    # Camera
    back_camera_mp_total: Optional[int | str | float] = None
    front_camera_mp: Optional[int | str | float] = None

    # Processor / Performance
    cpu_gpu: Optional[str] = None
    chipset: Optional[str] = None
    chipset_tier: Optional[str] = None
    cpu_cores: Optional[int | str] = None
    gpu_class: Optional[str] = None
    antutu_score: Optional[int | str | float] = None

    # Memory
    ram_gb: Optional[int | str | float] = None
    storage_gb: Optional[int | str | float] = None

    # Battery & Thermal
    battery_mah: Optional[int | str] = None
    cooling_system: Optional[str] = None

    # Connectivity
    wifi: Optional[str] = None
    nfc: Optional[int | str | bool] = None
    five_g_supported: Optional[int | str | bool] = Field(
        None, alias="5g_supported", serialization_alias="5g_supported"
    )

    # Launch
    launch_date: Optional[str] = None          # "YYYY-MM" e.g. "2024-01"
    months_since_launch: Optional[int | str | float] = None

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Normalized (final) output model — strict types
# ---------------------------------------------------------------------------


class DeviceSpec(BaseModel):
    """
    Fully normalized device specification.
    Column order matches the required CSV/JSON output layout.
    """

    # Display
    display_size_inch: Optional[float] = Field(None, ge=0.1)
    screen_resolution: Optional[int] = None
    display_refresh_hz: Optional[int] = Field(None, ge=1)

    # Pricing
    price_inr: Optional[int] = Field(None, ge=0)

    # Camera
    back_camera_mp_total: Optional[int] = Field(None, ge=0)
    front_camera_mp: Optional[int] = Field(None, ge=0)

    # Processor / Performance
    cpu_gpu: Optional[str] = None
    chipset: Optional[str] = None
    chipset_tier: Optional[str] = None
    cpu_cores: Optional[int] = Field(None, ge=1)
    gpu_class: Optional[str] = None
    antutu_score: Optional[int] = Field(None, ge=0)

    # Memory
    ram_gb: Optional[int] = Field(None, ge=1)
    storage_gb: Optional[int] = Field(None, ge=1)

    # Battery & Thermal
    battery_mah: Optional[int] = Field(None, ge=0)
    cooling_system: Optional[str] = None

    # Connectivity
    wifi: Optional[str] = None
    nfc: Optional[int] = Field(None, ge=0, le=1)
    five_g_supported: Optional[int] = Field(None, ge=0, le=1)

    # Launch
    launch_date: Optional[str] = None
    months_since_launch: Optional[int] = Field(None, ge=0)

    # Identifiers (end of row, mapped from input brand/model)
    device_manufacturer: Optional[str] = None
    device_model: Optional[str] = None

    @field_validator("chipset_tier")
    @classmethod
    def _validate_chipset_tier(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v_low = v.strip().lower()
        # Accept known values; also tolerate common variants
        if v_low in ("low", "mid", "medium", "high"):
            return "mid" if v_low == "medium" else v_low
        return v.strip()  # pass through unknown values rather than nulling

    @field_validator("gpu_class")
    @classmethod
    def _validate_gpu_class(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v_low = v.strip().lower()
        if v_low in ("weak", "low", "mid", "medium", "high"):
            if v_low in ("weak", "low"):
                return "weak"
            if v_low == "medium":
                return "mid"
            return v_low
        return v.strip()  # pass through unknown values

    @field_validator("wifi")
    @classmethod
    def _validate_wifi(cls, v: Optional[str]) -> Optional[str]:
        # Pass through any non-null wifi string — normalization already ran
        return v.strip() if v is not None else None

    @field_validator("screen_resolution")
    @classmethod
    def _validate_resolution(cls, v: Optional[int]) -> Optional[int]:
        # Accept any plausible resolution — don't restrict to 4 buckets
        if v is not None and v < 100:
            return None  # clearly garbage
        return v

    def to_flat_dict(self) -> dict:
        """
        Return a flat dict using canonical output column names.
        Renames five_g_supported → 5g_supported for CSV/JSON output.
        """
        data = self.model_dump()
        data["5g_supported"] = data.pop("five_g_supported")
        return data


# ---------------------------------------------------------------------------
# Batch result envelope
# ---------------------------------------------------------------------------


class BatchResult(BaseModel):
    """Tracks extraction outcome for one batch."""

    batch_index: int
    devices_requested: list[DeviceInput]
    devices_extracted: list[DeviceSpec] = Field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    from_cache: bool = False


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------


class RunSummary(BaseModel):
    """Aggregated statistics for a complete extraction run."""

    total_devices: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    total_batches: int = 0
    failed_batches: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    cache_hits: int = 0
