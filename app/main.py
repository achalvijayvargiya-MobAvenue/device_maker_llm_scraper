"""
CLI entry point for the mobile device specification collector.

Usage examples — LLM-only strategy (original):
    python app/main.py --input data/input/devices.csv --batch-size 5
    python app/main.py --input data/input/devices.json --batch-size 8 --resume
    python app/main.py --input data/input/devices.csv --replay-failed

Usage examples — scrape-first strategy (recommended for pending_device.csv):
    python app/main.py --input data/input/pending_device.csv --strategy scrape-first
    python app/main.py --input data/input/pending_device.csv --strategy scrape-first --resume
    python app/main.py --input data/input/pending_device.csv --strategy scrape-first --skip-enrich
    python app/main.py --input data/input/pending_device.csv --strategy scrape-first --skip-llm-fallback
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Use uvloop on Linux/macOS (SageMaker runs Amazon Linux) for a faster event loop.
# Gracefully skipped on Windows or if uvloop is not installed.
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

# When invoked as `python app/main.py`, the project root is not automatically
# on sys.path. Insert it so that `from app.*` imports resolve correctly
# regardless of how the script is launched.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import click

from app.batch_processor import BatchProcessor
from app.config import get_settings
from app.logger import get_logger, setup_logging
from app.scrape_pipeline import ScrapePipeline, write_skipped_csv
from app.utils import load_devices, print_summary, write_csv, write_json


@click.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to input file (.csv or .json) with brand/model columns.",
)
@click.option(
    "--batch-size",
    default=None,
    type=click.IntRange(1, 64),
    show_default=True,
    help="Number of devices per LLM call (overrides .env value).",
)
@click.option(
    "--concurrency",
    default=None,
    type=click.IntRange(1, 32),
    help="Max parallel LLM requests (overrides .env value).",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Directory for output files (overrides .env value).",
)
@click.option(
    "--resume/--no-resume",
    default=False,
    help="Resume from last checkpoint (skip already completed batches).",
)
@click.option(
    "--replay-failed",
    is_flag=True,
    default=False,
    help="Only replay batches that failed in the previous run.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable LLM response caching.",
)
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Override log level.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print batch plan without calling the API.",
)
@click.option(
    "--strategy",
    default="llm-only",
    type=click.Choice(["llm-only", "scrape-first"], case_sensitive=False),
    show_default=True,
    help=(
        "Extraction strategy. "
        "'scrape-first' uses GSMArena as primary source with LLM fallback. "
        "'llm-only' uses the original pure-LLM batch approach."
    ),
)
@click.option(
    "--skip-enrich",
    is_flag=True,
    default=False,
    help="[scrape-first] Skip the LLM enrichment pass for GSMArena-found devices.",
)
@click.option(
    "--skip-llm-fallback",
    is_flag=True,
    default=False,
    help="[scrape-first] Skip LLM fallback; emit null rows for devices not on GSMArena.",
)
@click.option(
    "--skip-scrape",
    is_flag=True,
    default=False,
    help=(
        "[scrape-first] Skip GSMArena scraping entirely. "
        "Checkpoint found-specs are preserved; all other devices go straight to LLM. "
        "Use this if GSMArena is rate-limiting."
    ),
)
def main(
    input_path: Path,
    batch_size: int | None,
    concurrency: int | None,
    output_dir: Path | None,
    resume: bool,
    replay_failed: bool,
    no_cache: bool,
    log_level: str | None,
    dry_run: bool,
    strategy: str,
    skip_enrich: bool,
    skip_llm_fallback: bool,
    skip_scrape: bool,
) -> None:
    """
    Mobile Device Specification Collector

    Reads a list of devices from INPUT and extracts structured specs.

    \b
    Strategies:
      llm-only     — original pure-LLM batch approach (default)
      scrape-first — GSMArena scraping as primary source, LLM as fallback
                     (recommended for large pending lists)
    """
    settings = get_settings()

    # Apply CLI overrides
    if batch_size is not None:
        settings.batch_size = batch_size
    if concurrency is not None:
        settings.max_concurrency = concurrency
    if output_dir is not None:
        settings.output_dir = output_dir
        settings.ensure_dirs()
    if no_cache:
        settings.enable_cache = False

    effective_log_level = log_level or settings.log_level
    setup_logging(effective_log_level, settings.log_file)
    logger = get_logger(__name__)

    logger.info("Mobile Device Collector starting")
    logger.info("Input : %s", input_path)
    logger.info("Model : %s | Batch size: %d | Concurrency: %d",
                settings.openai_model, settings.batch_size, settings.max_concurrency)

    # Load devices
    try:
        devices = load_devices(input_path)
    except Exception as exc:
        click.echo(f"ERROR loading input: {exc}", err=True)
        sys.exit(1)

    if not devices:
        click.echo("No devices found in input file.", err=True)
        sys.exit(1)

    click.echo(f"Loaded {len(devices)} devices from {input_path}")

    if dry_run:
        from app.batch_processor import split_into_batches
        batches = split_into_batches(
            devices,
            batch_size=settings.batch_size,
            token_budget=settings.token_budget_per_batch,
        )
        click.echo(f"\nDry run: {len(batches)} batches would be sent:")
        for i, b in enumerate(batches):
            names = ", ".join(f"{d.brand} {d.model}" for d in b)
            click.echo(f"  Batch {i+1:3d} ({len(b):2d} devices): {names}")
        return

    stem = input_path.stem
    json_path = settings.output_dir / f"{stem}_output.json"
    csv_path = settings.output_dir / f"{stem}_output.csv"

    # ------------------------------------------------------------------ #
    # SCRAPE-FIRST strategy                                               #
    # ------------------------------------------------------------------ #
    if strategy.lower() == "scrape-first":
        pipeline = ScrapePipeline(
            resume=resume,
            skip_enrich=skip_enrich,
            skip_llm_fallback=skip_llm_fallback,
            skip_scrape=skip_scrape,
        )
        try:
            specs, skipped_records, pipe_summary = asyncio.run(
                pipeline.run(devices)
            )
        except KeyboardInterrupt:
            click.echo("\nInterrupted. Partial progress saved to checkpoint.", err=True)
            sys.exit(130)
        except Exception as exc:
            logger.exception("Fatal error in scrape pipeline: %s", exc)
            sys.exit(1)

        write_json(specs, json_path)
        write_csv(specs, csv_path)

        # Write skipped-devices audit file
        skipped_path = settings.output_dir / f"{stem}_skipped.csv"
        write_skipped_csv(skipped_records, skipped_path)

        pipe_summary.print()
        click.echo(f"\nJSON         → {json_path}")
        click.echo(f"CSV          → {csv_path}")
        click.echo(f"Skipped audit→ {skipped_path}")
        return

    # ------------------------------------------------------------------ #
    # LLM-ONLY strategy (original)                                        #
    # ------------------------------------------------------------------ #

    # If not resuming, clear existing checkpoint
    if not resume and not replay_failed:
        _clear_checkpoint(settings)

    processor = BatchProcessor()
    try:
        specs, summary = asyncio.run(
            processor.run(devices, failed_only=replay_failed)
        )
    except KeyboardInterrupt:
        click.echo("\nInterrupted. Progress saved to checkpoint.", err=True)
        sys.exit(130)
    except Exception as exc:
        logger.exception("Fatal error during extraction: %s", exc)
        sys.exit(1)

    write_json(specs, json_path)
    write_csv(specs, csv_path)

    print_summary(summary)
    click.echo(f"\nJSON → {json_path}")
    click.echo(f"CSV  → {csv_path}")


def _clear_checkpoint(settings) -> None:
    """Remove stale checkpoint file so a fresh run starts clean."""
    if settings.checkpoint_file.exists():
        settings.checkpoint_file.unlink()
        get_logger(__name__).info("Checkpoint cleared for fresh run.")


if __name__ == "__main__":
    main()
