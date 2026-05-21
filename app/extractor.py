"""
Core extraction logic: sends one batch to the LLM and returns DeviceSpec list.

Handles:
- Cache lookup / store
- Prompt construction
- LLM call + retry
- JSON parsing + repair
- Pydantic validation
- Normalization
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Optional

from app.config import get_settings
from app.llm_client import LLMClient, LLMResponse
from app.logger import get_logger
from app.models import BatchResult, DeviceInput, DeviceSpec, RawDeviceSpec
from app.normalizer import normalize
from app.prompts import (
    build_messages,
    build_model_only_messages,
    build_repair_messages,
    build_enrichment_messages,
    is_generic_brand,
)
from app.retry_handler import extract_json_with_repair, with_retry
from app.utils import load_cache, save_cache

logger = get_logger(__name__)

# Key spec fields used to decide whether a result is "blank".
# A spec needs at least this many of these fields filled to be considered useful.
_BLANK_KEY_FIELDS: tuple[str, ...] = (
    "ram_gb",
    "storage_gb",
    "battery_mah",
    "display_size_inch",
    "cpu_gpu",
)
_BLANK_THRESHOLD = 2  # fewer than 2 key fields filled → trigger model-only retry


def _count_filled(spec: DeviceSpec) -> int:
    """Count how many key fields are non-null in a DeviceSpec."""
    return sum(1 for f in _BLANK_KEY_FIELDS if getattr(spec, f) is not None)


def _is_blank(spec: DeviceSpec) -> bool:
    """Return True if the spec is missing most key technical fields."""
    return _count_filled(spec) < _BLANK_THRESHOLD


def _cache_key(devices: list[DeviceInput]) -> str:
    """Deterministic cache key for a brand+model batch."""
    key_data = json.dumps(
        [{"brand": d.brand.lower(), "model": d.model.lower()} for d in devices],
        sort_keys=True,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _cache_key_model_only(devices: list[DeviceInput]) -> str:
    """Deterministic cache key for a model-only retry batch."""
    key_data = json.dumps(
        [{"model": d.model.lower()} for d in devices],
        sort_keys=True,
    ) + ":model_only"
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def _parse_and_normalize(
    raw_items: list[dict],
    expected_devices: list[DeviceInput],
    request_id: str,
) -> list[DeviceSpec]:
    """
    Validate each raw dict against RawDeviceSpec, then normalize.

    Missing items are filled with a skeleton DeviceSpec using the original
    brand/model so the output row count always matches the input.
    """
    results: list[DeviceSpec] = []

    for i, device_input in enumerate(expected_devices):
        if i < len(raw_items):
            raw_dict = raw_items[i]
            # Handle '5g_supported' key alias
            if "5g_supported" in raw_dict and "five_g_supported" not in raw_dict:
                raw_dict["five_g_supported"] = raw_dict.pop("5g_supported")
            try:
                raw_spec = RawDeviceSpec.model_validate(raw_dict)
                # Ensure manufacturer/model are populated from input as fallback
                if not raw_spec.device_manufacturer:
                    raw_spec.device_manufacturer = device_input.brand
                if not raw_spec.device_model:
                    raw_spec.device_model = device_input.model
                spec = normalize(raw_spec)
            except Exception as exc:
                logger.warning(
                    "Validation error for %s %s (request_id=%s): %s",
                    device_input.brand,
                    device_input.model,
                    request_id,
                    exc,
                )
                spec = DeviceSpec(
                    device_manufacturer=device_input.brand,
                    device_model=device_input.model,
                )
        else:
            logger.warning(
                "LLM returned fewer items than requested; "
                "using empty spec for %s %s (request_id=%s)",
                device_input.brand,
                device_input.model,
                request_id,
            )
            spec = DeviceSpec(
                device_manufacturer=device_input.brand,
                device_model=device_input.model,
            )

        results.append(spec)

    return results


async def _llm_extract(
    llm: LLMClient,
    messages: list[dict],
    devices: list[DeviceInput],
    request_id: str,
    *,
    json_mode: bool = False,
) -> tuple[list[dict], Optional[LLMResponse]]:
    """
    Run a single LLM extraction call with JSON repair.

    Returns
    -------
    tuple[list[dict], LLMResponse | None]
        Parsed raw JSON items and the last LLM response object (for token counts).

    Raises
    ------
    Exception
        Any unrecoverable error from the LLM or JSON repair.
    """
    llm_response: Optional[LLMResponse] = None

    async def _call() -> str:
        nonlocal llm_response
        llm_response = await llm.complete(
            messages, request_id=request_id, json_mode=json_mode
        )
        return llm_response.content

    async def _repair(malformed: str, error_detail: str) -> str:
        repair_msgs = build_repair_messages(devices, malformed, error_detail)
        nonlocal llm_response
        llm_response = await llm.complete(
            repair_msgs, request_id=f"{request_id}-repair"
        )
        return llm_response.content

    raw_items = await extract_json_with_repair(
        llm_call=lambda _msgs: with_retry(lambda: _call(), request_id=request_id),
        repair_call=lambda mal, err: with_retry(
            lambda: _repair(mal, err), request_id=request_id
        ),
        messages=messages,
        max_repair_attempts=2,
        request_id=request_id,
    )
    return raw_items, llm_response


class DeviceExtractor:
    """
    Stateful extractor that wraps the LLM client and cache.

    Intended to be used as an async context manager.

    Usage
    -----
    async with DeviceExtractor() as extractor:
        result = await extractor.extract_batch(devices, batch_index=0)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._llm: Optional[LLMClient] = None
        self._cache: dict[str, list[dict]] = {}

    async def __aenter__(self) -> "DeviceExtractor":
        self._llm = LLMClient()
        await self._llm.__aenter__()
        if self._settings.enable_cache:
            self._cache = load_cache(self._settings.cache_file)
        return self

    async def __aexit__(self, *args) -> None:
        if self._llm is not None:
            await self._llm.__aexit__(*args)
        if self._settings.enable_cache:
            save_cache(self._settings.cache_file, self._cache)

    async def enrich_batch(
        self,
        specs: list[DeviceSpec],
        *,
        batch_index: int = 0,
    ) -> list[DeviceSpec]:
        """
        Enrichment-only pass: fill antutu_score, cooling_system, price_inr
        for devices already scraped from GSMArena.

        Sends a cheap, focused LLM call for just 3 fields.  The heavy
        structural fields (chipset, RAM, battery, etc.) are already populated
        from GSMArena, so we use JSON mode for reliable output and keep
        batch_size small to avoid contamination.

        Parameters
        ----------
        specs:
            DeviceSpec objects coming from the GSMArena layer.
        batch_index:
            Used for request-ID logging.

        Returns
        -------
        list[DeviceSpec]
            Same length as input, with enriched fields merged in.
        """
        request_id = f"enrich-{batch_index:04d}-{uuid.uuid4().hex[:6]}"

        # Only send devices that are actually missing one of the 3 enrichment fields
        needs_enrichment = [
            i for i, s in enumerate(specs)
            if s.antutu_score is None or s.cooling_system is None or s.price_inr is None
        ]

        if not needs_enrichment:
            return specs

        enrich_devices = [
            {
                "brand": specs[i].device_manufacturer or "",
                "model": specs[i].device_model or "",
                "chipset": specs[i].chipset or "",
            }
            for i in needs_enrichment
        ]

        messages = build_enrichment_messages(enrich_devices)

        try:
            raw_items, _ = await _llm_extract(
                self._llm,
                messages,
                # We pass dummy DeviceInput list just to satisfy the signature;
                # repair logic uses the device list only for logging context.
                [DeviceInput(brand=d["brand"], model=d["model"]) for d in enrich_devices],
                request_id,
                json_mode=False,  # json_mode forces an object {}; we need an array []
            )
        except Exception as exc:
            logger.warning("Enrichment pass failed for batch %d: %s", batch_index, exc)
            return specs

        # Merge enriched fields back
        result = list(specs)
        for list_pos, original_idx in enumerate(needs_enrichment):
            if list_pos >= len(raw_items):
                break
            item = raw_items[list_pos]
            if not isinstance(item, dict):
                continue
            spec = result[original_idx]
            data = spec.model_dump()

            if spec.antutu_score is None and item.get("antutu_score") is not None:
                try:
                    data["antutu_score"] = int(float(str(item["antutu_score"])))
                except (ValueError, TypeError):
                    pass

            if spec.cooling_system is None and item.get("cooling_system"):
                cs = str(item["cooling_system"]).strip()
                if cs:
                    data["cooling_system"] = cs

            if spec.price_inr is None and item.get("price_inr") is not None:
                try:
                    data["price_inr"] = int(float(str(item["price_inr"])))
                except (ValueError, TypeError):
                    pass

            # Rebuild with validated data
            try:
                result[original_idx] = DeviceSpec(**{
                    k: v for k, v in data.items() if k in DeviceSpec.model_fields
                })
            except Exception as exc:
                logger.debug("Enrichment merge failed for index %d: %s", original_idx, exc)

        return result

    async def extract_batch(
        self,
        devices: list[DeviceInput],
        *,
        batch_index: int = 0,
    ) -> BatchResult:
        """
        Extract specifications for a batch of devices using a 2-pass strategy.

        Pass 1 — Brand + Model (standard):
            All devices are queried with brand + model together.

        Pass 2 — Model-Only retry (automatic):
            Triggered for any spec that is still blank after pass 1, AND for
            devices whose brand is a generic placeholder (e.g. "generic", "oem").
            The retry query uses ``brand model`` combined as a single identifier
            for known brands, or just ``model`` for generic/unknown brands.
            A spec is only replaced if the retry returned *more* filled fields.

        Parameters
        ----------
        devices:
            List of DeviceInput objects (typically 5–20 per batch).
        batch_index:
            Index of this batch within the overall run (for logging).

        Returns
        -------
        BatchResult
        """
        request_id = f"batch-{batch_index:04d}-{uuid.uuid4().hex[:6]}"
        t_start = time.perf_counter()
        total_input_tokens = 0
        total_output_tokens = 0

        # ------------------------------------------------------------------ #
        # Pass 1 — Brand + Model (standard)                                  #
        # ------------------------------------------------------------------ #

        key = _cache_key(devices)
        pass1_from_cache = False

        if self._settings.enable_cache and key in self._cache:
            logger.info("Cache hit (pass-1) for batch %d", batch_index)
            raw_items = self._cache[key]
            pass1_from_cache = True
        else:
            try:
                raw_items, resp = await _llm_extract(
                    self._llm,
                    build_messages(devices),
                    devices,
                    request_id,
                    json_mode=False,  # json_mode forces {} object; we need [] array
                )
                if resp:
                    total_input_tokens += resp.input_tokens
                    total_output_tokens += resp.output_tokens
                if self._settings.enable_cache:
                    self._cache[key] = raw_items
            except Exception as exc:
                latency_ms = (time.perf_counter() - t_start) * 1000
                logger.error("Batch %d pass-1 failed (%s): %s", batch_index, request_id, exc)
                return BatchResult(
                    batch_index=batch_index,
                    devices_requested=devices,
                    success=False,
                    error=str(exc),
                    latency_ms=latency_ms,
                )

        specs: list[DeviceSpec] = _parse_and_normalize(raw_items, devices, request_id)

        # ------------------------------------------------------------------ #
        # Pass 2 — Model-Only retry for blank / generic-brand specs           #
        # ------------------------------------------------------------------ #

        # Collect indices that need a retry:
        #   • spec is blank (< _BLANK_THRESHOLD key fields filled)  OR
        #   • brand is a meaningless generic placeholder
        retry_indices = [
            i
            for i, (dev, spec) in enumerate(zip(devices, specs))
            if _is_blank(spec) or is_generic_brand(dev.brand)
        ]

        if retry_indices and not pass1_from_cache:
            retry_devices = [devices[i] for i in retry_indices]
            retry_id = f"{request_id}-retry"
            retry_key = _cache_key_model_only(retry_devices)

            logger.info(
                "Batch %d: triggering model-only retry for %d blank/generic spec(s): %s",
                batch_index,
                len(retry_indices),
                [f"{devices[i].brand} {devices[i].model}" for i in retry_indices],
            )

            if self._settings.enable_cache and retry_key in self._cache:
                retry_raw = self._cache[retry_key]
            else:
                try:
                    retry_raw, retry_resp = await _llm_extract(
                        self._llm,
                        build_model_only_messages(retry_devices),
                        retry_devices,
                        retry_id,
                    )
                    if retry_resp:
                        total_input_tokens += retry_resp.input_tokens
                        total_output_tokens += retry_resp.output_tokens
                    if self._settings.enable_cache:
                        self._cache[retry_key] = retry_raw
                except Exception as exc:
                    logger.warning(
                        "Batch %d model-only retry failed (%s): %s — keeping pass-1 results",
                        batch_index,
                        retry_id,
                        exc,
                    )
                    retry_raw = []

            retry_specs = _parse_and_normalize(retry_raw, retry_devices, retry_id)

            improved = 0
            for list_pos, original_idx in enumerate(retry_indices):
                if list_pos >= len(retry_specs):
                    break
                retry_spec = retry_specs[list_pos]
                if _count_filled(retry_spec) > _count_filled(specs[original_idx]):
                    specs[original_idx] = retry_spec
                    improved += 1
                    logger.debug(
                        "Model-only retry improved spec for '%s %s' "
                        "(%d → %d key fields filled)",
                        retry_spec.device_manufacturer,
                        retry_spec.device_model,
                        _count_filled(specs[original_idx]),
                        _count_filled(retry_spec),
                    )

            if improved:
                logger.info(
                    "Batch %d: model-only retry improved %d/%d blank spec(s)",
                    batch_index,
                    improved,
                    len(retry_indices),
                )

        latency_ms = (time.perf_counter() - t_start) * 1000

        return BatchResult(
            batch_index=batch_index,
            devices_requested=devices,
            devices_extracted=specs,
            success=True,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            latency_ms=latency_ms,
            from_cache=pass1_from_cache,
        )
