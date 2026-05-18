"""
Configuration management using pydantic-settings and python-dotenv.
All runtime parameters are sourced from environment variables or .env file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central settings object; values loaded from .env then environment."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field("gpt-4o-mini", description="Model to use for extraction")
    openai_temperature: float = Field(0.0, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(6000, ge=256, le=16384)
    openai_timeout: float = Field(120.0, ge=5.0, description="HTTP timeout in seconds")
    openai_max_retries: int = Field(3, ge=0, le=10)

    # --- Batching ---
    # Sweet spot for gpt-4o-mini: 20 devices ≈ 2100 input + 3400 output tokens,
    # ~30s/call. Main throughput lever is max_concurrency, not batch_size.
    batch_size: int = Field(20, ge=1, le=64, description="Devices per LLM call")
    max_concurrency: int = Field(8, ge=1, le=32, description="Parallel LLM requests")
    token_budget_per_batch: int = Field(
        12000, description="Soft token cap per batch to avoid overflow"
    )

    # --- Paths ---
    input_dir: Path = Field(ROOT_DIR / "data" / "input")
    output_dir: Path = Field(ROOT_DIR / "data" / "output")
    checkpoint_file: Path = Field(
        ROOT_DIR / "data" / "output" / "checkpoint.json",
        description="Resume state file",
    )

    # --- Logging ---
    log_level: str = Field("INFO", description="DEBUG | INFO | WARNING | ERROR")
    log_file: Path = Field(ROOT_DIR / "data" / "output" / "run.log")

    # --- Cost tracking ---
    cost_per_1k_input_tokens: float = Field(
        0.00015, description="USD per 1K input tokens (gpt-4o-mini default)"
    )
    cost_per_1k_output_tokens: float = Field(
        0.0006, description="USD per 1K output tokens (gpt-4o-mini default)"
    )

    # --- Caching ---
    enable_cache: bool = Field(True, description="Cache LLM responses to disk")
    cache_file: Path = Field(ROOT_DIR / "data" / "output" / "cache.json")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v

    def ensure_dirs(self) -> None:
        """Create output directory and parents if missing."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
