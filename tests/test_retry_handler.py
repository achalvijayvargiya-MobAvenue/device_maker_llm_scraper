"""
Unit tests for app.retry_handler.

Tests cover JSON parsing, markdown fence stripping, and repair logic.
LLM/OpenAI calls are fully mocked — no network access required.
"""

from __future__ import annotations

import pytest

from app.retry_handler import _strip_markdown_fences, extract_json_with_repair


class TestStripMarkdownFences:
    def test_json_fence(self):
        text = "```json\n[{\"a\": 1}]\n```"
        assert _strip_markdown_fences(text) == '[{"a": 1}]'

    def test_plain_fence(self):
        text = "```\n[{\"a\": 1}]\n```"
        assert _strip_markdown_fences(text) == '[{"a": 1}]'

    def test_no_fence(self):
        text = '[{"a": 1}]'
        assert _strip_markdown_fences(text) == '[{"a": 1}]'

    def test_whitespace_stripped(self):
        text = "  \n[{}]\n  "
        assert _strip_markdown_fences(text) == "[{}]"


class TestExtractJsonWithRepair:
    @pytest.mark.asyncio
    async def test_valid_json_on_first_call(self):
        async def llm_call(messages):
            return '[{"brand": "Samsung", "model": "S24"}]'

        async def repair_call(mal, err):
            raise AssertionError("Should not call repair for valid JSON")

        result = await extract_json_with_repair(
            llm_call=llm_call,
            repair_call=repair_call,
            messages=[],
        )
        assert result == [{"brand": "Samsung", "model": "S24"}]

    @pytest.mark.asyncio
    async def test_repair_on_invalid_json(self):
        call_count = {"n": 0}

        async def llm_call(messages):
            return "This is not JSON at all"

        async def repair_call(malformed, error_detail):
            call_count["n"] += 1
            return '[{"brand": "Xiaomi", "model": "Note 13"}]'

        result = await extract_json_with_repair(
            llm_call=llm_call,
            repair_call=repair_call,
            messages=[],
            max_repair_attempts=2,
        )
        assert call_count["n"] == 1
        assert result[0]["brand"] == "Xiaomi"

    @pytest.mark.asyncio
    async def test_raises_after_max_repairs(self):
        async def llm_call(messages):
            return "still not json"

        async def repair_call(mal, err):
            return "still not json either"

        with pytest.raises(ValueError, match="Could not parse LLM JSON"):
            await extract_json_with_repair(
                llm_call=llm_call,
                repair_call=repair_call,
                messages=[],
                max_repair_attempts=1,
            )

    @pytest.mark.asyncio
    async def test_not_array_raises(self):
        async def llm_call(messages):
            return '{"brand": "Samsung"}'

        async def repair_call(mal, err):
            return '{"brand": "Samsung"}'

        with pytest.raises(ValueError):
            await extract_json_with_repair(
                llm_call=llm_call,
                repair_call=repair_call,
                messages=[],
                max_repair_attempts=0,
            )
