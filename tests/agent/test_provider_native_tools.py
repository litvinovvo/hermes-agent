from agent.provider_native_tools import (
    NativeToolSpec,
    filter_managed_tools_for_native_tools,
    resolve_provider_native_tools,
)


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_resolve_codex_native_web_search_from_config():
    specs = resolve_provider_native_tools(
        {"codex": {"web_search": "live"}},
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert specs == [
        NativeToolSpec(
            logical_name="web_search",
            provider_tool_type="web_search",
            mode="live",
            suppress_managed_tools=("web_search",),
            include=("web_search_call.action.sources",),
        )
    ]


def test_resolve_codex_native_web_search_from_generic_config():
    specs = resolve_provider_native_tools(
        {"provider_native_tools": {"openai-codex": {"web_search": "cached"}}},
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert len(specs) == 1
    assert specs[0].logical_name == "web_search"
    assert specs[0].mode == "cached"


def test_resolve_codex_native_image_generation_from_config():
    specs = resolve_provider_native_tools(
        {"codex": {"image_generation": "enabled"}},
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert specs == [
        NativeToolSpec(
            logical_name="image_generate",
            provider_tool_type="image_generation",
            mode="enabled",
            suppress_managed_tools=("image_generate",),
        )
    ]


def test_resolve_native_tools_ignores_disabled_and_non_codex_providers():
    assert resolve_provider_native_tools(
        {"codex": {"web_search": "disabled"}},
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
    ) == []
    assert resolve_provider_native_tools(
        {"codex": {"web_search": "live"}},
        provider="openrouter",
        api_mode="chat_completions",
        base_url="https://openrouter.ai/api/v1",
    ) == []


def test_filter_managed_tools_suppresses_only_conflicting_native_tools():
    specs = [
        NativeToolSpec(
            logical_name="web_search",
            provider_tool_type="web_search",
            mode="live",
            suppress_managed_tools=("web_search",),
        )
    ]
    tools = [_tool("web_search"), _tool("web_extract"), _tool("terminal")]

    filtered = filter_managed_tools_for_native_tools(tools, specs)

    assert [tool["function"]["name"] for tool in filtered] == ["web_extract", "terminal"]
    assert [tool["function"]["name"] for tool in tools] == ["web_search", "web_extract", "terminal"]


def test_filter_managed_tools_can_suppress_native_image_generation():
    specs = [
        NativeToolSpec(
            logical_name="image_generate",
            provider_tool_type="image_generation",
            mode="enabled",
            suppress_managed_tools=("image_generate",),
        )
    ]
    tools = [_tool("image_generate"), _tool("vision_analyze"), _tool("web_search")]

    filtered = filter_managed_tools_for_native_tools(tools, specs)

    assert [tool["function"]["name"] for tool in filtered] == ["vision_analyze", "web_search"]


def test_codex_native_tool_choice_for_freshness_request():
    from agent.provider_native_tools import codex_native_tool_choice_for_request

    specs = resolve_provider_native_tools(
        {"codex": {"web_search": "live", "web_search_tool_choice": "auto"}},
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert codex_native_tool_choice_for_request(
        specs,
        [{"role": "user", "content": "Поищи актуальные новости на сегодня"}],
    ) == {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "web_search"}],
    }


def test_codex_native_tool_choice_auto_ignores_non_freshness_request():
    from agent.provider_native_tools import codex_native_tool_choice_for_request

    specs = resolve_provider_native_tools(
        {"codex": {"web_search": "live", "web_search_tool_choice": "auto"}},
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert codex_native_tool_choice_for_request(
        specs,
        [{"role": "user", "content": "Explain binary search."}],
    ) is None


def test_codex_native_tool_choice_required_forces_every_request():
    from agent.provider_native_tools import codex_native_tool_choice_for_request

    specs = resolve_provider_native_tools(
        {"codex": {"web_search": "live", "web_search_tool_choice": "required"}},
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert codex_native_tool_choice_for_request(
        specs,
        [{"role": "user", "content": "Explain binary search."}],
    )["mode"] == "required"


def test_codex_native_tool_choice_none_never_forces():
    from agent.provider_native_tools import codex_native_tool_choice_for_request

    specs = resolve_provider_native_tools(
        {"codex": {"web_search": "live", "web_search_tool_choice": "none"}},
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert codex_native_tool_choice_for_request(
        specs,
        [{"role": "user", "content": "latest news today"}],
    ) is None


def test_codex_native_tool_choice_for_image_generation_request():
    from agent.provider_native_tools import codex_native_tool_choice_for_request

    specs = resolve_provider_native_tools(
        {"codex": {"image_generation": "enabled", "image_generation_tool_choice": "auto"}},
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert codex_native_tool_choice_for_request(
        specs,
        [{"role": "user", "content": "Сгенерируй маленькую тестовую картинку: синий квадрат"}],
    ) == {
        "type": "allowed_tools",
        "mode": "required",
        "tools": [{"type": "image_generation"}],
    }


def test_codex_native_tool_choice_prefers_image_generation_over_freshness_when_prompt_asks_for_image():
    from agent.provider_native_tools import codex_native_tool_choice_for_request

    specs = resolve_provider_native_tools(
        {
            "codex": {
                "web_search": "live",
                "web_search_tool_choice": "auto",
                "image_generation": "enabled",
                "image_generation_tool_choice": "auto",
            }
        },
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert codex_native_tool_choice_for_request(
        specs,
        [{"role": "user", "content": "Сгенерируй актуальную картинку для новости сегодня"}],
    )["tools"] == [{"type": "image_generation"}]


def test_codex_native_image_generation_tool_choice_none_never_forces():
    from agent.provider_native_tools import codex_native_tool_choice_for_request

    specs = resolve_provider_native_tools(
        {"codex": {"image_generation": "enabled", "image_generation_tool_choice": "none"}},
        provider="openai-codex",
        api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex",
    )

    assert codex_native_tool_choice_for_request(
        specs,
        [{"role": "user", "content": "Generate an avatar image"}],
    ) is None
