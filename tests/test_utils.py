"""
Unit tests for app.utils — output writers and input loaders.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.models import DeviceSpec
from app.utils import write_csv, write_json, load_devices_from_csv, load_devices_from_json


@pytest.fixture()
def sample_specs() -> list[DeviceSpec]:
    return [
        DeviceSpec(
            brand="Samsung",
            model="Galaxy S24",
            ram_gb=8,
            storage_gb=128,
            chipset_tier="high",
            five_g_supported=1,
            nfc=1,
            wifi="wifi 7",
            screen_resolution=1080,
        ),
        DeviceSpec(
            brand="Xiaomi",
            model="Redmi Note 13",
            ram_gb=6,
            storage_gb=128,
            chipset_tier="mid",
            five_g_supported=1,
            nfc=0,
            wifi="wifi 6",
            screen_resolution=1080,
        ),
    ]


class TestWriteJson:
    def test_creates_valid_json(self, tmp_path, sample_specs):
        out = tmp_path / "output.json"
        write_json(sample_specs, out)
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["brand"] == "Samsung"
        assert "5g_supported" in data[0]
        assert "five_g_supported" not in data[0]

    def test_creates_parent_dirs(self, tmp_path, sample_specs):
        out = tmp_path / "deep" / "nested" / "output.json"
        write_json(sample_specs, out)
        assert out.exists()


class TestWriteCsv:
    def test_creates_valid_csv(self, tmp_path, sample_specs):
        out = tmp_path / "output.csv"
        write_csv(sample_specs, out)
        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 2
        assert rows[0]["brand"] == "Samsung"
        assert "5g_supported" in rows[0]


class TestLoadDevicesFromCsv:
    def test_loads_correctly(self, tmp_path):
        csv_content = "brand,model\nSamsung,Galaxy S24\nXiaomi,Note 13\n"
        p = tmp_path / "devices.csv"
        p.write_text(csv_content)
        devices = load_devices_from_csv(p)
        assert len(devices) == 2
        assert devices[0].brand == "Samsung"

    def test_missing_column_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("name,version\nX,Y\n")
        with pytest.raises(ValueError, match="brand"):
            load_devices_from_csv(p)


class TestLoadDevicesFromJson:
    def test_loads_correctly(self, tmp_path):
        data = [{"brand": "OnePlus", "model": "12"}]
        p = tmp_path / "devices.json"
        p.write_text(json.dumps(data))
        devices = load_devices_from_json(p)
        assert len(devices) == 1
        assert devices[0].model == "12"

    def test_not_array_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text('{"brand": "X"}')
        with pytest.raises(ValueError, match="array"):
            load_devices_from_json(p)
