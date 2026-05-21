#!/usr/bin/env python3
"""
Quick check that GEMINI_API_KEY in .env works for a Gemini LLM call.

Usage (from mobile_device_collector/):
    python test_gemini_api.py

Optional env vars:
    GEMINI_MODEL   — default: gemini-2.5-flash
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "gemini-2.5-flash"
PROMPT = "Reply with exactly one word: OK"


def main() -> int:
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("FAIL: GEMINI_API_KEY is missing or empty in .env")
        return 1

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )

    body = json.dumps(
        {
            "contents": [{"parts": [{"text": PROMPT}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 32},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
    )

    print(f"Model: {model}")
    print(f"Prompt: {PROMPT!r}")
    print("Calling Gemini API...")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        print(f"FAIL: HTTP {exc.code} {exc.reason}")
        if exc.code in (401, 403):
            print("The API key is invalid or lacks permission for this model.")
        elif exc.code == 429:
            print(
                "The key is recognized, but quota is exhausted for this model. "
                "Try another model:  set GEMINI_MODEL=gemini-2.5-flash"
            )
        try:
            err_json = json.loads(err_body)
            print(json.dumps(err_json, indent=2))
        except json.JSONDecodeError:
            print(err_body[:2000])
        return 1
    except urllib.error.URLError as exc:
        print(f"FAIL: Network error — {exc.reason}")
        return 1

    try:
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        print("FAIL: Unexpected response shape")
        print(json.dumps(payload, indent=2)[:2000])
        print(f"Parse error: {exc}")
        return 1

    usage = payload.get("usageMetadata", {})
    print("SUCCESS: API key is valid and the model responded.")
    print(f"Response: {text.strip()!r}")
    if usage:
        print(
            "Tokens - "
            f"prompt: {usage.get('promptTokenCount', '?')}, "
            f"output: {usage.get('candidatesTokenCount', '?')}, "
            f"total: {usage.get('totalTokenCount', '?')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
