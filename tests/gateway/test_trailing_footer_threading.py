import pytest
from unittest.mock import AsyncMock

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_trailing_footer_preserves_source_thread_metadata():
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="572506117",
        chat_type="dm",
        thread_id="257392",
        user_id="572506117",
    )

    await runner._send_trailing_footer(source, "gpt-5.5 · 26%")

    adapter.send.assert_awaited_once_with(
        "572506117",
        "gpt-5.5 · 26%",
        metadata={
            "thread_id": "257392",
            "telegram_dm_topic_reply_fallback": True,
        },
    )


@pytest.mark.asyncio
async def test_trailing_footer_omits_metadata_without_thread():
    runner = GatewayRunner.__new__(GatewayRunner)
    adapter = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="572506117",
        chat_type="dm",
        thread_id=None,
    )

    await runner._send_trailing_footer(source, "gpt-5.5")

    adapter.send.assert_awaited_once_with(
        "572506117",
        "gpt-5.5",
        metadata=None,
    )
