"""
Batch processor: splits the device list into token-safe batches and
runs them concurrently with a semaphore-controlled concurrency limit.

Supports:
- Dynamic batch sizing based on estimated token budget.
- Resume from checkpoint (skip already-processed device indices).
- Failed-batch replay.
- Live progress reporting via tqdm.
- Aggregated RunSummary statistics.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import tiktoken
from tqdm.asyncio import tqdm as async_tqdm

from app.config import get_settings
from app.extractor import DeviceExtractor
from app.logger import get_logger
from app.models import BatchResult, DeviceInput, DeviceSpec, RunSummary
from app.prompts import SYSTEM_PROMPT, build_user_message
from app.utils import load_checkpoint, save_checkpoint

logger = get_logger(__name__)

# Approximate chars-per-token ratio used as a fast fallback when tiktoken is slow
_CHARS_PER_TOKEN = 4


def _estimate_tokens(devices: list[DeviceInput]) -> int:
    """
    Estimate token count for a batch using tiktoken (cl100k_base).

    Falls back to character-count heuristic on encoding failure.
    """
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        text = SYSTEM_PROMPT + build_user_message(devices)
        return len(enc.encode(text))
    except Exception:
        text = SYSTEM_PROMPT + build_user_message(devices)
        return len(text) // _CHARS_PER_TOKEN


def split_into_batches(
    devices: list[DeviceInput],
    *,
    batch_size: int,
    token_budget: int,
) -> list[list[DeviceInput]]:
    """
    Split devices into batches, respecting both batch_size and token_budget.

    Reduces the batch if it would exceed the token budget, ensuring each
    sub-batch stays within context limits.

    Parameters
    ----------
    devices:
        Full list of DeviceInput objects.
    batch_size:
        Maximum number of devices per batch.
    token_budget:
        Soft token ceiling per batch (prompt tokens only).

    Returns
    -------
    list[list[DeviceInput]]
        Ordered list of batches.
    """
    batches: list[list[DeviceInput]] = []
    i = 0
    while i < len(devices):
        # Start with requested batch size
        size = min(batch_size, len(devices) - i)
        candidate = devices[i : i + size]

        # Shrink until within token budget
        while size > 1 and _estimate_tokens(candidate) > token_budget:
            size -= 1
            candidate = devices[i : i + size]

        batches.append(candidate)
        i += size

    logger.info(
        "Split %d devices into %d batches (batch_size=%d, token_budget=%d)",
        len(devices),
        len(batches),
        batch_size,
        token_budget,
    )
    return batches


class BatchProcessor:
    """
    Orchestrates concurrent extraction across all batches.

    Usage
    -----
    processor = BatchProcessor()
    specs, summary = await processor.run(devices)
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    async def run(
        self,
        devices: list[DeviceInput],
        *,
        failed_only: bool = False,
    ) -> tuple[list[DeviceSpec], RunSummary]:
        """
        Run the full extraction pipeline.

        Parameters
        ----------
        devices:
            All devices to process.
        failed_only:
            If True, only replay batches that previously failed
            (loaded from checkpoint).

        Returns
        -------
        tuple[list[DeviceSpec], RunSummary]
            All extracted specs and a run summary with token/cost stats.
        """
        settings = self._settings
        checkpoint = load_checkpoint(settings.checkpoint_file)

        batches = split_into_batches(
            devices,
            batch_size=settings.batch_size,
            token_budget=settings.token_budget_per_batch,
        )

        # Filter to only unprocessed / failed batches
        pending_indices = _get_pending_indices(
            batches, checkpoint, failed_only=failed_only
        )
        logger.info(
            "%d/%d batches pending extraction",
            len(pending_indices),
            len(batches),
        )

        semaphore = asyncio.Semaphore(settings.max_concurrency)
        all_results: list[Optional[BatchResult]] = [None] * len(batches)

        # Populate already-done results from checkpoint
        for idx, batch_data in checkpoint.get("completed", {}).items():
            batch_idx = int(idx)
            if batch_idx < len(batches) and batch_idx not in pending_indices:
                all_results[batch_idx] = BatchResult(
                    batch_index=batch_idx,
                    devices_requested=batches[batch_idx],
                    devices_extracted=[
                        DeviceSpec(**d) for d in batch_data.get("specs", [])
                    ],
                    success=True,
                    from_cache=True,
                )

        async with DeviceExtractor() as extractor:
            tasks = [
                _extract_with_semaphore(
                    extractor, batches[i], i, semaphore
                )
                for i in pending_indices
            ]

            with_progress = async_tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Extracting batches",
                unit="batch",
            )

            for coro in with_progress:
                result: BatchResult = await coro
                all_results[result.batch_index] = result
                _update_checkpoint(checkpoint, result)
                save_checkpoint(settings.checkpoint_file, checkpoint)

        summary = _build_summary(all_results, settings)
        all_specs = _collect_specs(all_results)

        logger.info(
            "Run complete: %d specs extracted, %d failed batches, "
            "%.4f USD estimated cost",
            summary.successful_extractions,
            summary.failed_batches,
            summary.estimated_cost_usd,
        )

        return all_specs, summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _extract_with_semaphore(
    extractor: DeviceExtractor,
    batch: list[DeviceInput],
    batch_index: int,
    semaphore: asyncio.Semaphore,
) -> BatchResult:
    async with semaphore:
        return await extractor.extract_batch(batch, batch_index=batch_index)


def _get_pending_indices(
    batches: list[list[DeviceInput]],
    checkpoint: dict,
    *,
    failed_only: bool,
) -> list[int]:
    """Return batch indices that still need processing."""
    completed = set(int(k) for k in checkpoint.get("completed", {}).keys())
    if failed_only:
        failed = set(int(k) for k in checkpoint.get("failed", {}).keys())
        return sorted(failed)
    return [i for i in range(len(batches)) if i not in completed]


def _update_checkpoint(checkpoint: dict, result: BatchResult) -> None:
    """Update in-memory checkpoint dict after a batch finishes."""
    if result.success:
        checkpoint.setdefault("completed", {})[str(result.batch_index)] = {
            "specs": [s.to_flat_dict() for s in result.devices_extracted],
        }
        checkpoint.get("failed", {}).pop(str(result.batch_index), None)
    else:
        checkpoint.setdefault("failed", {})[str(result.batch_index)] = {
            "error": result.error,
            "devices": [
                {"brand": d.brand, "model": d.model}
                for d in result.devices_requested
            ],
        }


def _build_summary(
    results: list[Optional[BatchResult]],
    settings,
) -> RunSummary:
    summary = RunSummary(total_devices=0, total_batches=len(results))
    for r in results:
        if r is None:
            continue
        summary.total_devices += len(r.devices_requested)
        summary.total_input_tokens += r.input_tokens
        summary.total_output_tokens += r.output_tokens
        summary.total_latency_ms += r.latency_ms
        if r.success:
            summary.successful_extractions += len(r.devices_extracted)
        else:
            summary.failed_batches += 1
            summary.failed_extractions += len(r.devices_requested)
        if r.from_cache:
            summary.cache_hits += 1

    cost_in = (summary.total_input_tokens / 1000) * settings.cost_per_1k_input_tokens
    cost_out = (
        (summary.total_output_tokens / 1000) * settings.cost_per_1k_output_tokens
    )
    summary.estimated_cost_usd = round(cost_in + cost_out, 6)
    return summary


def _collect_specs(results: list[Optional[BatchResult]]) -> list[DeviceSpec]:
    specs: list[DeviceSpec] = []
    for r in results:
        if r is not None and r.success:
            specs.extend(r.devices_extracted)
    return specs
