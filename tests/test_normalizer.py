"""
Unit tests for app.normalizer.

Tests cover type coercion helpers, chipset/GPU classification,
WiFi normalization, resolution bucketing, and months calculation.
"""

from __future__ import annotations

import pytest

from app.normalizer import (
    _to_binary,
    _to_float,
    _to_int,
    calculate_months_since_launch,
    classify_chipset_tier,
    classify_gpu_class,
    normalize,
    normalize_resolution,
    normalize_wifi,
)
from app.models import RawDeviceSpec


# ---------------------------------------------------------------------------
# _to_int
# ---------------------------------------------------------------------------


class TestToInt:
    def test_integer_passthrough(self):
        assert _to_int(8) == 8

    def test_float_truncates(self):
        assert _to_int(8.9) == 8

    def test_string_with_unit(self):
        assert _to_int("12 GB") == 12

    def test_mah_string(self):
        assert _to_int("5000mAh") == 5000

    def test_none(self):
        assert _to_int(None) is None

    def test_empty_string(self):
        assert _to_int("") is None

    def test_non_numeric_string(self):
        assert _to_int("N/A") is None


# ---------------------------------------------------------------------------
# _to_binary
# ---------------------------------------------------------------------------


class TestToBinary:
    def test_true(self):
        assert _to_binary(True) == 1

    def test_false(self):
        assert _to_binary(False) == 0

    def test_int_1(self):
        assert _to_binary(1) == 1

    def test_int_0(self):
        assert _to_binary(0) == 0

    def test_string_yes(self):
        assert _to_binary("yes") == 1

    def test_string_no(self):
        assert _to_binary("no") == 0

    def test_none(self):
        assert _to_binary(None) is None

    def test_invalid_string(self):
        assert _to_binary("maybe") is None


# ---------------------------------------------------------------------------
# classify_chipset_tier
# ---------------------------------------------------------------------------


class TestClassifyChipsetTier:
    @pytest.mark.parametrize(
        "chipset,expected",
        [
            ("Snapdragon 8 Gen 3", "high"),
            ("Snapdragon 8 Gen 1", "high"),
            ("Apple A17 Pro", "high"),
            ("Dimensity 9300", "high"),
            ("Exynos 2400", "high"),
            ("Snapdragon 7 Gen 2", "mid"),
            ("Snapdragon 695", "mid"),
            ("Dimensity 7200", "mid"),
            ("Helio G99", "mid"),
            ("Snapdragon 4 Gen 2", "low"),
            ("Helio G35", "low"),
            ("Unisoc T610", "low"),
        ],
    )
    def test_known_chipsets(self, chipset, expected):
        assert classify_chipset_tier(chipset) == expected

    def test_none_returns_none(self):
        assert classify_chipset_tier(None) is None

    def test_unknown_returns_none(self):
        assert classify_chipset_tier("FutureTech X9999") is None


# ---------------------------------------------------------------------------
# classify_gpu_class
# ---------------------------------------------------------------------------


class TestClassifyGpuClass:
    @pytest.mark.parametrize(
        "gpu,expected",
        [
            ("Adreno 750", "high"),
            ("Immortalis G925", "high"),
            ("Adreno 642L", "mid"),
            ("Mali-G68", "mid"),
            ("Adreno 506", "weak"),
            ("PowerVR GE8320", "weak"),
        ],
    )
    def test_known_gpus(self, gpu, expected):
        assert classify_gpu_class(gpu) == expected

    def test_none_returns_none(self):
        assert classify_gpu_class(None) is None


# ---------------------------------------------------------------------------
# normalize_wifi
# ---------------------------------------------------------------------------


class TestNormalizeWifi:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Wi-Fi 7", "wifi 7"),
            ("WiFi 6E", "wifi 6e"),
            ("802.11ax", "wifi 6"),
            ("802.11ac", "wifi 5"),
            ("wi fi 6", "wifi 6"),
        ],
    )
    def test_known_wifi(self, raw, expected):
        assert normalize_wifi(raw) == expected

    def test_none_returns_none(self):
        assert normalize_wifi(None) is None


# ---------------------------------------------------------------------------
# normalize_resolution
# ---------------------------------------------------------------------------


class TestNormalizeResolution:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (1080, 1080),
            (2400, 1080),  # 2400 vertical → FHD bucket
            ("1080x2400", 1080),
            ("FHD", 1080),
            ("QHD+", 1440),
            ("4K", 2160),
            ("720p", 720),
            (720, 720),
        ],
    )
    def test_resolution_buckets(self, raw, expected):
        assert normalize_resolution(raw) == expected

    def test_none(self):
        assert normalize_resolution(None) is None


# ---------------------------------------------------------------------------
# calculate_months_since_launch
# ---------------------------------------------------------------------------


class TestMonthsSinceLaunch:
    def test_2024_launch(self):
        # From Jan 2024 to May 2026 = 28 months
        assert calculate_months_since_launch(2024) == 28

    def test_2022_launch(self):
        # From Jan 2022 to May 2026 = 52 months
        assert calculate_months_since_launch(2022) == 52

    def test_none_returns_none(self):
        assert calculate_months_since_launch(None) is None

    def test_non_negative(self):
        assert calculate_months_since_launch(2027) == 0


# ---------------------------------------------------------------------------
# Full normalize() pipeline
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_full_pipeline(self):
        raw = RawDeviceSpec(
            brand="Samsung",
            model="Galaxy S24 Ultra",
            ram_gb="12 GB",
            storage_gb="256GB",
            chipset_tier=None,          # Should be derived from chipset
            cpu_cores="8",
            gpu_class=None,             # Should be derived from chipset
            screen_refresh_rate="120Hz",
            battery_capacity="5000mAh",
            launch_year="2024",
            months_since_launch=None,   # Should be calculated
            five_g_supported=True,
            price_inr="129999",
            screen_size="6.8",
            screen_resolution="3088x1440",
            chipset="Snapdragon 8 Gen 3",
            wifi="Wi-Fi 7",
            nfc="yes",
            antutu_score="2100000",
        )

        spec = normalize(raw)

        assert spec.brand == "Samsung"
        assert spec.model == "Galaxy S24 Ultra"
        assert spec.ram_gb == 12
        assert spec.storage_gb == 256
        assert spec.chipset_tier == "high"
        assert spec.cpu_cores == 8
        assert spec.gpu_class == "high"
        assert spec.screen_refresh_rate == 120
        assert spec.battery_capacity == 5000
        assert spec.launch_year == 2024
        assert spec.months_since_launch == 28
        assert spec.five_g_supported == 1
        assert spec.price_inr == 129999
        assert spec.screen_size == pytest.approx(6.8)
        assert spec.screen_resolution == 1440
        assert spec.chipset == "Snapdragon 8 Gen 3"
        assert spec.wifi == "wifi 7"
        assert spec.nfc == 1
        assert spec.antutu_score == 2100000

    def test_null_fields_graceful(self):
        raw = RawDeviceSpec(brand="Test", model="X1")
        spec = normalize(raw)
        assert spec.brand == "Test"
        assert spec.model == "X1"
        assert spec.ram_gb is None

    def test_to_flat_dict_key_naming(self):
        raw = RawDeviceSpec(brand="OnePlus", model="12", five_g_supported=1)
        spec = normalize(raw)
        d = spec.to_flat_dict()
        assert "5g_supported" in d
        assert "five_g_supported" not in d
