"""
Unit tests for app.models — Pydantic schema validation and serialization.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import DeviceInput, DeviceSpec, RawDeviceSpec


class TestDeviceInput:
    def test_valid(self):
        d = DeviceInput(brand="Samsung", model="Galaxy S24")
        assert d.brand == "Samsung"

    def test_strips_whitespace(self):
        d = DeviceInput(brand="  Apple  ", model="  iPhone 15 Pro  ")
        assert d.brand == "Apple"
        assert d.model == "iPhone 15 Pro"

    def test_missing_brand_raises(self):
        with pytest.raises(ValidationError):
            DeviceInput(brand="", model="X")

    def test_missing_model_raises(self):
        with pytest.raises(ValidationError):
            DeviceInput(brand="Samsung", model="")


class TestRawDeviceSpec:
    def test_alias_5g(self):
        raw = RawDeviceSpec.model_validate({"5g_supported": 1})
        assert raw.five_g_supported == 1

    def test_all_none(self):
        raw = RawDeviceSpec()
        assert raw.brand is None
        assert raw.ram_gb is None

    def test_accepts_string_numbers(self):
        raw = RawDeviceSpec(ram_gb="12 GB", storage_gb="256")
        assert raw.ram_gb == "12 GB"


class TestDeviceSpec:
    def test_to_flat_dict_key(self):
        spec = DeviceSpec(brand="Xiaomi", model="Note 13", five_g_supported=1)
        d = spec.to_flat_dict()
        assert "5g_supported" in d
        assert d["5g_supported"] == 1
        assert "five_g_supported" not in d

    def test_invalid_chipset_tier_coerced_to_none(self):
        spec = DeviceSpec(chipset_tier="ultra")
        assert spec.chipset_tier is None

    def test_invalid_gpu_class_coerced_to_none(self):
        spec = DeviceSpec(gpu_class="premium")
        assert spec.gpu_class is None

    def test_invalid_wifi_coerced_to_none(self):
        spec = DeviceSpec(wifi="wifi 4")
        assert spec.wifi is None

    def test_invalid_resolution_coerced_to_none(self):
        spec = DeviceSpec(screen_resolution=480)
        assert spec.screen_resolution is None

    def test_valid_full_spec(self):
        spec = DeviceSpec(
            brand="OnePlus",
            model="12",
            ram_gb=16,
            storage_gb=256,
            chipset_tier="high",
            cpu_cores=8,
            gpu_class="high",
            screen_refresh_rate=120,
            battery_capacity=5400,
            launch_year=2024,
            months_since_launch=16,
            five_g_supported=1,
            price_inr=64999,
            screen_size=6.82,
            screen_resolution=1080,
            chipset="Snapdragon 8 Gen 3",
            wifi="wifi 7",
            nfc=1,
            antutu_score=2050000,
        )
        assert spec.brand == "OnePlus"
        assert spec.five_g_supported == 1
