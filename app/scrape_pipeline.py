"""
Multi-layer device extraction pipeline (scrape-first strategy).

Layer 1 — Pre-filter & brand normalize
    Rule-based classifier skips test builds, TV models, garbled strings.
    Normalizes brand codes (lge → LG, tct → Alcatel, etc.).

Layer 2 — GSMArena scraper (primary, free)
    Async scraper with SQLite cache; 3-attempt search strategy.
    Covers ~60-70% of real mobile devices.

Layer 3 — LLM enrichment for GSMArena-found devices
    Fills in the 3 fields not on GSMArena: antutu_score, cooling_system, price_inr.
    Uses a focused enrichment prompt + JSON mode for reliability.
    Very cheap: ~100 tokens per device.

Layer 4 — LLM fallback for devices not found on GSMArena
    Individual-device calls (batch_size=1) to eliminate batch contamination.
    JSON mode enabled for all single-device calls.

Results are checkpointed after every scrape batch so the run is resumable.
"""

from __future__ import annotations

import asyncio
import csv
import json
import time
from pathlib import Path
from typing import Optional

from tqdm.asyncio import tqdm as async_tqdm

from app.batch_processor import split_into_batches
from app.config import get_settings
from app.device_filter import filter_devices
from app.extractor import DeviceExtractor
from app.gsmarena.scraper import GSMArenaScraper
from app.logger import get_logger
from app.models import DeviceInput, DeviceSpec, RunSummary
from app.normalizer import derive_missing_fields
from app.utils import (
    load_checkpoint,
    save_checkpoint,
    write_csv,
    write_json,
)

logger = get_logger(__name__)

# How many GSMArena-found specs to batch together for the LLM enrichment pass
_ENRICH_BATCH_SIZE = 10

# Checkpoint key for LLM fallback completed batches
_CK_FALLBACK_DONE = "llm_fallback_done"  # dict: batch_index -> list[spec dicts]

# Checkpoint key names
_CK_SCRAPE_DONE = "scrape_done"     # set of device keys already processed by scraper
_CK_SPECS = "specs"                  # list of serialized DeviceSpec dicts


class ScrapePipelineResult:
    """Summary statistics from a ScrapePipeline run."""

    def __init__(self) -> None:
        self.total_input: int = 0
        self.skipped: int = 0
        self.scrape_found: int = 0
        self.scrape_not_found: int = 0
        self.llm_fallback_filled: int = 0
        self.llm_fallback_null: int = 0
        self.total_output: int = 0
        self.elapsed_s: float = 0.0

    def print(self) -> None:
        sep = "=" * 56
        print(sep)
        print("  SCRAPE PIPELINE SUMMARY")
        print(sep)
        print(f"  Total input          : {self.total_input}")
        print(f"  Skipped (invalid)    : {self.skipped}")
        print(f"  GSMArena found       : {self.scrape_found}")
        print(f"  GSMArena not found   : {self.scrape_not_found}")
        print(f"  LLM fallback filled  : {self.llm_fallback_filled}")
        print(f"  LLM fallback null    : {self.llm_fallback_null}")
        print(f"  Total output rows    : {self.total_output}")
        print(f"  Elapsed              : {self.elapsed_s:.0f}s")
        print(sep)


class ScrapePipeline:
    """
    Orchestrates the 4-layer extraction pipeline.

    Usage
    -----
    pipeline = ScrapePipeline()
    specs, skipped, summary = await pipeline.run(devices)
    """

    def __init__(
        self,
        checkpoint_file: Optional[Path] = None,
        resume: bool = False,
        skip_enrich: bool = False,
        skip_llm_fallback: bool = False,
        skip_scrape: bool = False,
    ) -> None:
        self._settings = get_settings()
        self._checkpoint_file = checkpoint_file or (
            self._settings.output_dir / "scrape_checkpoint.json"
        )
        self._resume = resume
        self._skip_enrich = skip_enrich
        self._skip_llm_fallback = skip_llm_fallback
        self._skip_scrape = skip_scrape

    async def run(
        self,
        devices: list[DeviceInput],
    ) -> tuple[list[DeviceSpec], list[dict], ScrapePipelineResult]:
        """
        Run the full 4-layer pipeline.

        Parameters
        ----------
        devices:
            Raw input devices (may contain invalids, test builds, etc.).

        Returns
        -------
        tuple[list[DeviceSpec], list[dict], ScrapePipelineResult]
            - specs: all extracted DeviceSpec objects (one per valid input device)
            - skipped_records: list of dicts with brand/model/skip_reason
            - summary: statistics
        """
        t_start = time.perf_counter()
        summary = ScrapePipelineResult()
        summary.total_input = len(devices)

        # ------------------------------------------------------------------ #
        # Layer 1: Pre-filter & Brand Normalize                               #
        # ------------------------------------------------------------------ #
        logger.info("Layer 1: filtering %d devices", len(devices))
        valid_devices, skipped_records = filter_devices(devices)
        summary.skipped = len(skipped_records)
        logger.info(
            "Layer 1 done: %d valid, %d skipped",
            len(valid_devices),
            summary.skipped,
        )

        if not valid_devices:
            summary.elapsed_s = time.perf_counter() - t_start
            return [], skipped_records, summary

        # Load checkpoint
        checkpoint = {}
        if self._resume and self._checkpoint_file.exists():
            checkpoint = load_checkpoint(self._checkpoint_file)
            logger.info(
                "Resuming: %d devices already processed in checkpoint",
                len(checkpoint.get(_CK_SCRAPE_DONE, {})),
            )

        done_keys: set[str] = set(checkpoint.get(_CK_SCRAPE_DONE, {}).keys())

        # Rebuild already-done specs from checkpoint
        spec_store: dict[str, dict] = dict(checkpoint.get(_CK_SPECS, {}))

        # Determine which devices still need scraping
        pending = [
            d for d in valid_devices
            if _device_key(d) not in done_keys
        ]
        already_done = [
            d for d in valid_devices
            if _device_key(d) in done_keys
        ]
        logger.info(
            "%d devices pending scrape, %d already done",
            len(pending),
            len(already_done),
        )

        # ------------------------------------------------------------------ #
        # Layer 2: GSMArena Scraper  (skipped if --skip-scrape)              #
        # ------------------------------------------------------------------ #
        scrape_found: list[tuple[DeviceInput, DeviceSpec]] = []   # (input_device, spec)
        scrape_not_found: list[DeviceInput] = []

        if not self._skip_scrape and pending:
            logger.info("Layer 2: scraping %d devices from GSMArena", len(pending))
            scrape_found, scrape_not_found = await self._run_scraper(
                pending, spec_store, done_keys, checkpoint
            )
            logger.info(
                "Layer 2 done: %d found, %d not found",
                len(scrape_found),
                len(scrape_not_found),
            )
        elif self._skip_scrape:
            logger.info(
                "Layer 2: GSMArena scraping skipped — %d devices go to LLM fallback",
                len(pending),
            )
            scrape_not_found = list(pending)

        # Reconstruct specs for already-done devices from checkpoint
        for dev in already_done:
            key = _device_key(dev)
            raw_data = spec_store.get(key)
            if raw_data is not None:
                try:
                    # Normalize 5g field name
                    if "5g_supported" in raw_data:
                        raw_data["five_g_supported"] = raw_data.pop("5g_supported")
                    spec = DeviceSpec(**{
                        k: v for k, v in raw_data.items()
                        if k in DeviceSpec.model_fields
                    })
                    scrape_found.append((dev, spec))
                except Exception as exc:
                    logger.warning("Failed to restore checkpoint spec for %s: %s", key, exc)
                    scrape_not_found.append(dev)
            else:
                # Previously checked but not found on GSMArena → LLM fallback
                scrape_not_found.append(dev)

        summary.scrape_found = len(scrape_found)
        summary.scrape_not_found = len(scrape_not_found)

        # ------------------------------------------------------------------ #
        # Layer 3: LLM Enrichment for GSMArena-found devices                 #
        # ------------------------------------------------------------------ #
        enriched_specs: list[DeviceSpec] = [s for _, s in scrape_found]

        if not self._skip_enrich and scrape_found:
            logger.info(
                "Layer 3: enriching %d GSMArena specs (antutu, cooling, price_inr)",
                len(enriched_specs),
            )
            enriched_specs = await self._run_enrichment(enriched_specs)

        # Apply deterministic field derivation regardless
        enriched_specs = [derive_missing_fields(s) for s in enriched_specs]

        # ------------------------------------------------------------------ #
        # Layer 4: LLM Fallback for devices not found on GSMArena            #
        # ------------------------------------------------------------------ #
        fallback_specs: list[DeviceSpec] = []

        if not self._skip_llm_fallback and scrape_not_found:
            logger.info(
                "Layer 4: LLM fallback for %d devices not found on GSMArena",
                len(scrape_not_found),
            )
            fallback_specs = await self._run_llm_fallback(
                scrape_not_found, checkpoint
            )
            summary.llm_fallback_filled = sum(
                1 for s in fallback_specs if _spec_has_data(s)
            )
            summary.llm_fallback_null = len(fallback_specs) - summary.llm_fallback_filled
        elif scrape_not_found:
            # Emit null specs to keep row count consistent
            fallback_specs = [
                DeviceSpec(
                    device_manufacturer=d.brand,
                    device_model=d.model,
                )
                for d in scrape_not_found
            ]

        all_specs = enriched_specs + fallback_specs
        summary.total_output = len(all_specs)
        summary.elapsed_s = time.perf_counter() - t_start

        return all_specs, skipped_records, summary

    # ------------------------------------------------------------------
    # Layer 2 helpers
    # ------------------------------------------------------------------

    async def _run_scraper(
        self,
        devices: list[DeviceInput],
        spec_store: dict[str, dict],
        done_keys: set[str],
        checkpoint: dict,
    ) -> tuple[list[tuple[DeviceInput, DeviceSpec]], list[DeviceInput]]:
        """
        Run the GSMArena scraper over all pending devices.

        Saves progress to the checkpoint file after each device so the
        run is resumable even if interrupted mid-way.
        """
        found: list[tuple[DeviceInput, DeviceSpec]] = []
        not_found: list[DeviceInput] = []

        async with GSMArenaScraper() as scraper:
            for device in async_tqdm(
                devices,
                desc="GSMArena scrape",
                unit="device",
                dynamic_ncols=True,
            ):
                key = _device_key(device)
                spec = await scraper.lookup(brand=device.brand, model=device.model)

                if spec is not None:
                    found.append((device, spec))
                    spec_store[key] = spec.to_flat_dict()
                else:
                    not_found.append(device)
                    spec_store[key] = None  # type: ignore[assignment]

                done_keys.add(key)
                checkpoint[_CK_SCRAPE_DONE] = {k: True for k in done_keys}
                checkpoint[_CK_SPECS] = spec_store
                save_checkpoint(self._checkpoint_file, checkpoint)

        return found, not_found

    # ------------------------------------------------------------------
    # Layer 3 helpers
    # ------------------------------------------------------------------

    async def _run_enrichment(self, specs: list[DeviceSpec]) -> list[DeviceSpec]:
        """
        Run LLM enrichment in batches to fill antutu_score, cooling_system, price_inr.
        """
        batches = [
            specs[i: i + _ENRICH_BATCH_SIZE]
            for i in range(0, len(specs), _ENRICH_BATCH_SIZE)
        ]
        enriched: list[DeviceSpec] = []

        async with DeviceExtractor() as extractor:
            for i, batch in enumerate(
                async_tqdm(batches, desc="LLM enrich", unit="batch", dynamic_ncols=True)
            ):
                result = await extractor.enrich_batch(batch, batch_index=i)
                enriched.extend(result)

        return enriched

    # ------------------------------------------------------------------
    # Layer 4 helpers
    # ------------------------------------------------------------------

    async def _run_llm_fallback(
        self,
        devices: list[DeviceInput],
        checkpoint: dict,
    ) -> list[DeviceSpec]:
        """
        Run LLM extraction for devices not found on GSMArena.

        Devices are grouped into batches (respecting batch_size + token_budget
        from settings) so that a single LLM call covers many devices rather
        than one per device.  Progress is checkpointed after each batch so the
        run is resumable after interruption.
        """
        settings = self._settings
        batches = split_into_batches(
            devices,
            batch_size=settings.batch_size,
            token_budget=settings.token_budget_per_batch,
        )

        # Load already-completed fallback batches from checkpoint
        done_batches: dict[str, list[dict]] = checkpoint.get(_CK_FALLBACK_DONE, {})

        # Pre-fill results from checkpoint
        all_specs: list[Optional[DeviceSpec]] = [None] * len(devices)
        device_offset = 0
        for b_idx, batch in enumerate(batches):
            key = str(b_idx)
            if key in done_batches:
                for j, spec_dict in enumerate(done_batches[key]):
                    abs_idx = device_offset + j
                    if abs_idx < len(devices):
                        try:
                            all_specs[abs_idx] = DeviceSpec(**{
                                k: v for k, v in spec_dict.items()
                                if k in DeviceSpec.model_fields
                            })
                        except Exception:
                            all_specs[abs_idx] = DeviceSpec(
                                device_manufacturer=devices[abs_idx].brand,
                                device_model=devices[abs_idx].model,
                            )
            device_offset += len(batch)

        pending_batch_indices = [
            i for i in range(len(batches)) if str(i) not in done_batches
        ]
        logger.info(
            "LLM fallback: %d/%d batches pending (batch_size=%d, ~%d requests total)",
            len(pending_batch_indices),
            len(batches),
            settings.batch_size,
            len(pending_batch_indices),
        )

        semaphore = asyncio.Semaphore(settings.max_concurrency)

        # Map batch index → device offset in `devices` list
        offsets: list[int] = []
        off = 0
        for batch in batches:
            offsets.append(off)
            off += len(batch)

        async def _extract_batch(b_idx: int) -> None:
            batch = batches[b_idx]
            async with semaphore:
                try:
                    result = await extractor.extract_batch(batch, batch_index=b_idx)
                    specs = (
                        [derive_missing_fields(s) for s in result.devices_extracted]
                        if result.success and result.devices_extracted
                        else [
                            DeviceSpec(device_manufacturer=d.brand, device_model=d.model)
                            for d in batch
                        ]
                    )
                except Exception as exc:
                    logger.warning("LLM fallback batch %d failed: %s", b_idx, exc)
                    specs = [
                        DeviceSpec(device_manufacturer=d.brand, device_model=d.model)
                        for d in batch
                    ]

            start = offsets[b_idx]
            for j, spec in enumerate(specs):
                if start + j < len(all_specs):
                    all_specs[start + j] = spec

            # Checkpoint this batch
            done_batches[str(b_idx)] = [s.to_flat_dict() for s in specs]
            checkpoint[_CK_FALLBACK_DONE] = done_batches
            save_checkpoint(self._checkpoint_file, checkpoint)

        async with DeviceExtractor() as extractor:
            tasks = [_extract_batch(i) for i in pending_batch_indices]
            for coro in async_tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="LLM fallback",
                unit="batch",
                dynamic_ncols=True,
            ):
                await coro

        return [
            s if s is not None else DeviceSpec(
                device_manufacturer=devices[i].brand,
                device_model=devices[i].model,
            )
            for i, s in enumerate(all_specs)
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device_key(device: DeviceInput) -> str:
    """Stable string key for a DeviceInput."""
    return f"{device.brand.lower()}|{device.model.lower()}"


def _spec_has_data(spec: DeviceSpec) -> bool:
    """Return True if the spec has at least 2 of the 5 key fields populated."""
    key_fields = ("ram_gb", "storage_gb", "battery_mah", "display_size_inch", "cpu_gpu")
    filled = sum(1 for f in key_fields if getattr(spec, f) is not None)
    return filled >= 2


def write_skipped_csv(skipped: list[dict], path: Path) -> None:
    """Write the skipped-devices audit CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not skipped:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["brand", "model", "skip_reason"])
        writer.writeheader()
        writer.writerows(skipped)
    logger.info("Skipped devices written to %s (%d rows)", path, len(skipped))
