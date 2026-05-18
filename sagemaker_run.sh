#!/bin/bash
# =============================================================================
# Mobile Device Collector — SageMaker Setup & Run Script
#
# Works on:
#   - SageMaker Studio (JupyterLab terminal)
#   - SageMaker Notebook Instance terminal
#   - Any Amazon Linux 2 / AL2023 EC2 instance
#
# Usage:
#   chmod +x sagemaker_run.sh
#   ./sagemaker_run.sh --input data/input/new_device_models.csv
#
# To run in background (survives SSH disconnect):
#   nohup ./sagemaker_run.sh --input data/input/new_device_models.csv > run.log 2>&1 &
#   tail -f run.log
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. CONFIGURATION — edit these or pass as environment variables
# ---------------------------------------------------------------------------

PYTHON=${PYTHON:-python3}
INPUT_FILE=${INPUT_FILE:-"data/input/new_device_models.csv"}
BATCH_SIZE=${BATCH_SIZE:-20}
CONCURRENCY=${CONCURRENCY:-8}
OUTPUT_DIR=${OUTPUT_DIR:-"data/output"}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}

# OpenAI API Key — set via environment variable, NOT hard-coded here.
# Export before running:
#   export OPENAI_API_KEY="sk-proj-..."
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set."
    echo "Run:  export OPENAI_API_KEY='sk-proj-your-key-here'"
    exit 1
fi

# Parse optional --input flag from command line
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)   INPUT_FILE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --resume)  RESUME="--resume"; shift ;;
        --replay-failed) REPLAY="--replay-failed"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

RESUME=${RESUME:-""}
REPLAY=${REPLAY:-""}

# ---------------------------------------------------------------------------
# 2. MOVE TO PROJECT ROOT (script lives in project root)
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Working directory: $(pwd)"

# ---------------------------------------------------------------------------
# 3. PYTHON VERSION CHECK (requires 3.11+)
# ---------------------------------------------------------------------------

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

echo "Python version: $PY_VERSION"

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 11 ]]; then
    echo "ERROR: Python 3.11+ required. Found $PY_VERSION"
    echo ""
    echo "On SageMaker Notebook Instance, use:"
    echo "  conda activate python3    # usually 3.10"
    echo "  conda install python=3.11 # or create a new env:"
    echo "  conda create -n py311 python=3.11 -y && conda activate py311"
    echo ""
    echo "On SageMaker Studio, select 'Python 3.11' kernel or set PYTHON=python3.11"
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. VIRTUAL ENVIRONMENT SETUP (idempotent)
# ---------------------------------------------------------------------------

VENV_DIR=".venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
echo "Activated venv: $(which python)"

# ---------------------------------------------------------------------------
# 5. INSTALL DEPENDENCIES (only if not already installed)
# ---------------------------------------------------------------------------

if ! python -c "import openai" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    echo "Dependencies installed."
else
    echo "Dependencies already installed — skipping."
fi

# ---------------------------------------------------------------------------
# 6. CONFIRM uvloop IS ACTIVE
# ---------------------------------------------------------------------------

python -c "
import asyncio
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    print('uvloop: ACTIVE (faster event loop)')
except ImportError:
    print('uvloop: not found — using default asyncio loop')
"

# ---------------------------------------------------------------------------
# 7. CREATE OUTPUT DIRECTORY
# ---------------------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# 8. PRINT RUN PLAN
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  MOBILE DEVICE COLLECTOR — RUN PLAN"
echo "============================================================"
echo "  Input file   : $INPUT_FILE"
echo "  Batch size   : $BATCH_SIZE"
echo "  Concurrency  : $CONCURRENCY"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Log level    : $LOG_LEVEL"
echo "  Resume       : ${RESUME:-no}"
echo "  Replay failed: ${REPLAY:-no}"
echo "  API key      : ${OPENAI_API_KEY:0:12}... (masked)"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# 9. DRY RUN FIRST (shows batch plan, no API calls)
# ---------------------------------------------------------------------------

echo "Dry run (batch plan preview):"
python app/main.py \
    --input "$INPUT_FILE" \
    --batch-size "$BATCH_SIZE" \
    --concurrency "$CONCURRENCY" \
    --output-dir "$OUTPUT_DIR" \
    --log-level "$LOG_LEVEL" \
    --dry-run

echo ""
echo "Starting extraction in 3 seconds... (Ctrl+C to abort)"
sleep 3

# ---------------------------------------------------------------------------
# 10. RUN EXTRACTION
# ---------------------------------------------------------------------------

python app/main.py \
    --input "$INPUT_FILE" \
    --batch-size "$BATCH_SIZE" \
    --concurrency "$CONCURRENCY" \
    --output-dir "$OUTPUT_DIR" \
    --log-level "$LOG_LEVEL" \
    ${RESUME} \
    ${REPLAY}

echo ""
echo "Run complete. Outputs in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
