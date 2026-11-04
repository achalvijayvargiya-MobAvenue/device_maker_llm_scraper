# Mobile Device Specification Collector

A production-ready Python pipeline that takes a list of mobile device brand/model pairs and uses **OpenAI LLM APIs** to extract, normalize, and output structured device specifications.

---

## Features

| Feature | Details |
|---|---|
| **LLM Extraction** | OpenAI GPT-4o-mini (configurable), temperature=0 |
| **Batch Processing** | 5–10 devices per call, token-safe dynamic splitting |
| **Async Concurrency** | Semaphore-controlled parallel LLM calls |
| **Retry Handling** | Tenacity exponential backoff for rate limits & timeouts |
| **JSON Self-Repair** | Asks the LLM to fix its own malformed JSON output |
| **Pydantic Validation** | Strict schema validation + graceful null coercion |
| **Normalization** | Chipset tier, GPU class, WiFi, resolution, months-since-launch |
| **Caching** | Disk-backed SHA256 cache — skip re-calls for seen batches |
| **Resumable Execution** | JSON checkpoint — restart interrupted runs from last batch |
| **Output Formats** | CSV + JSON |
| **Cost Estimation** | Per-run USD estimate printed in summary |
| **CLI** | `click`-based CLI with `--resume`, `--replay-failed`, `--dry-run` |

---

## Project Structure

```
mobile_device_collector/
├── app/
│   ├── main.py            # CLI entry point
│   ├── config.py          # Settings (pydantic-settings + .env)
│   ├── models.py          # Pydantic models: DeviceInput, RawDeviceSpec, DeviceSpec
│   ├── prompts.py         # System + user prompt builders
│   ├── llm_client.py      # Async OpenAI wrapper (LLMClient)
│   ├── normalizer.py      # Field normalization & classification
│   ├── extractor.py       # Single-batch extraction orchestrator
│   ├── batch_processor.py # Multi-batch concurrency + checkpoint
│   ├── retry_handler.py   # Tenacity retry + JSON repair
│   ├── utils.py           # I/O writers, cache, checkpoint helpers
│   └── logger.py          # Logging setup (console + rotating file)
├── data/
│   ├── input/
│   │   ├── devices.csv    # Sample input CSV
│   │   └── devices.json   # Sample input JSON
│   └── output/
│       └── sample_output.json
├── tests/
│   ├── test_normalizer.py
│   ├── test_retry_handler.py
│   ├── test_models.py
│   └── test_utils.py
├── .env.example
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

---

## Quick Start

### 1. Clone & Install

Python 3.11 or newer is required (`pyproject.toml` and the header of `requirements.txt`). Run the commands below from this repository's root. In the ML Data Collection Portal that directory is `device-data`.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Run

```bash
# Basic run from project root
python app/main.py --input data/input/devices.csv

# Custom batch size and concurrency
python app/main.py --input data/input/devices.csv --batch-size 8 --concurrency 6

# Resume an interrupted run
python app/main.py --input data/input/devices.csv --resume

# Replay only failed batches
python app/main.py --input data/input/devices.csv --replay-failed

# Preview batch plan without making any API calls
python app/main.py --input data/input/devices.csv --dry-run

# JSON input
python app/main.py --input data/input/devices.json
```

`--strategy` defaults to `llm-only`. `app/main.py` also accepts `scrape-first`, which uses GSMArena first and the LLM as a fallback:

```bash
python app/main.py --input data/input/pending_device.csv --strategy scrape-first
python app/main.py --input data/input/pending_device.csv --strategy scrape-first --resume
python app/main.py --input data/input/pending_device.csv --strategy scrape-first --skip-enrich
python app/main.py --input data/input/pending_device.csv --strategy scrape-first --skip-llm-fallback
```

`--skip-scrape` skips GSMArena and sends devices that are not already checkpointed to the LLM. `--output-dir` overrides the default `data/output` directory.

`sagemaker_run.sh` runs the same collector. It requires `OPENAI_API_KEY` in the environment and writes under `data/output`. It does not upload to S3.

```bash
chmod +x sagemaker_run.sh
./sagemaker_run.sh --input data/input/new_device_models.csv
```

---

## Input Format

### CSV (`devices.csv`)

```csv
brand,model
Samsung,Galaxy S24 Ultra
Xiaomi,Redmi Note 13 Pro
OnePlus,OnePlus 12
```

### JSON (`devices.json`)

```json
[
  {"brand": "Samsung", "model": "Galaxy S24 Ultra"},
  {"brand": "Xiaomi", "model": "Redmi Note 13 Pro"}
]
```

---

## Output Format

Results are written to `data/output/<input_stem>_output.csv` and `data/output/<input_stem>_output.json`.

### JSON Sample

```json
[
  {
    "brand": "Samsung",
    "model": "Galaxy S24 Ultra",
    "ram_gb": 12,
    "storage_gb": 256,
    "chipset_tier": "high",
    "cpu_cores": 8,
    "gpu_class": "high",
    "screen_refresh_rate": 120,
    "battery_capacity": 5000,
    "launch_year": 2024,
    "months_since_launch": 28,
    "5g_supported": 1,
    "price_inr": 129999,
    "screen_size": 6.8,
    "screen_resolution": 1440,
    "chipset": "Snapdragon 8 Gen 3",
    "wifi": "wifi 7",
    "nfc": 1,
    "antutu_score": 2100000
  }
]
```

---

## Normalization Rules

| Field | Rule |
|---|---|
| `ram_gb` | Integer GB (strips units) |
| `storage_gb` | Base/primary storage integer |
| `chipset_tier` | `low` / `mid` / `high` (keyword-classified from chipset name) |
| `gpu_class` | `weak` / `mid` / `high` (keyword-classified) |
| `screen_refresh_rate` | Integer Hz |
| `battery_capacity` | Integer mAh |
| `5g_supported` | `0` or `1` |
| `nfc` | `0` or `1` |
| `screen_resolution` | Bucketed to `720` / `1080` / `1440` / `2160` |
| `wifi` | `wifi 5` / `wifi 6` / `wifi 6e` / `wifi 7` |
| `months_since_launch` | Calculated from `launch_year` to May 2026 |
| `price_inr` | Integer (INR) |

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model to use |
| `OPENAI_TEMPERATURE` | `0.0` | Sampling temperature |
| `OPENAI_MAX_TOKENS` | `6000` | Max output tokens per call |
| `OPENAI_TIMEOUT` | `120.0` | HTTP timeout in seconds |
| `OPENAI_MAX_RETRIES` | `3` | Retry attempts for API errors |
| `BATCH_SIZE` | `20` | Devices per LLM call |
| `MAX_CONCURRENCY` | `8` | Parallel LLM requests |
| `TOKEN_BUDGET_PER_BATCH` | `12000` | Soft token ceiling per batch |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `ENABLE_CACHE` | `true` | Cache LLM responses to disk |
| `COST_PER_1K_INPUT_TOKENS` | `0.00015` | USD per 1K input tokens used for the run cost estimate |
| `COST_PER_1K_OUTPUT_TOKENS` | `0.0006` | USD per 1K output tokens used for the run cost estimate |

Defaults match `app/config.py` and `.env.example`. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` there. Do not commit `.env`.

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest
# With coverage:
pytest --cov=app --cov-report=term-missing
```

---

## Architecture Notes

### Low Hallucination Design
- `temperature=0` on all API calls.
- System prompt explicitly forbids guessing and instructs `null` for unknowns.
- Normalization layer re-classifies chipset/GPU tier from known keyword patterns, overriding any hallucinated tier value from the LLM.

### Resumable Execution
- After each batch succeeds, a `checkpoint.json` is written atomically (write-to-temp then rename).
- Re-running with `--resume` skips completed batches.
- `--replay-failed` only re-runs batches that previously errored.

### Caching
- Each batch is hashed (SHA256 of sorted brand+model list).
- Cache stored in `data/output/cache.json`.
- Disable with `--no-cache` or `ENABLE_CACHE=false`.

### Cost Control
- Token estimation via `tiktoken` before sending — oversized batches are split further.
- Cost printed per run using configurable per-1K-token pricing.

---

## License

MIT
