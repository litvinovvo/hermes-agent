"""Provider-native tool capability helpers.

Hermes managed tools are normal function tools implemented by Hermes. Some
providers also expose hosted/native tools that must be declared on the model
request itself (for example OpenAI/Codex Responses ``web_search``).  This module
keeps that policy separate from provider transports so transports only need to
map a small ``NativeToolSpec`` into their wire format.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class NativeToolSpec:
    """Logical description of a provider-native tool exposed for one request."""

    logical_name: str
    provider_tool_type: str
    mode: str
    suppress_managed_tools: tuple[str, ...] = ()
    include: tuple[str, ...] = ()
    tool_choice: str = "auto"


def _normalize_mode(value: Any) -> str | None:
    if isinstance(value, bool):
        return "enabled" if value else None
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", "-")
    if not text or text in {"0", "false", "off", "no", "none", "disabled", "disable"}:
        return None
    if text in {"1", "true", "on", "yes", "enabled", "enable"}:
        return "enabled"
    if text == "live":
        return "live"
    if text in {"cached", "cache", "no-live", "no-live-web", "offline"}:
        return "cached"
    return None


def _normalize_tool_choice_mode(value: Any) -> str:
    if value is None:
        return "auto"
    text = str(value).strip().lower().replace("_", "-")
    if not text or text in {"0", "false", "off", "no", "none", "disabled", "disable"}:
        return "none"
    if text in {"1", "true", "on", "yes", "required", "require", "force", "forced", "always"}:
        return "required"
    if text in {"auto", "automatic", "detect", "fresh", "freshness"}:
        return "auto"
    return "auto"


_FRESHNESS_PATTERN = re.compile(
    r"\b("
    r"today|latest|current|currently|recent|breaking|news|headline|headlines|"
    r"now|live|this week|this month|as of|update|updates|price|prices|"
    r"weather|forecast|score|scores|schedule|release|version|"
    r"сегодня|сейчас|актуальн|последн|свеж|новост|заголов|"
    r"курс|цен[аы]|погод|прогноз|сч[её]т|расписан|релиз|верси"
    r")\b",
    re.IGNORECASE,
)


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                value = part.get("text") or part.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    return ""


def latest_user_text(messages: Sequence[Mapping[str, Any]] | None) -> str:
    for message in reversed(messages or []):
        if isinstance(message, Mapping) and message.get("role") == "user":
            return _message_content_text(message.get("content"))
    return ""


def is_freshness_sensitive_request(messages: Sequence[Mapping[str, Any]] | None) -> bool:
    return bool(_FRESHNESS_PATTERN.search(latest_user_text(messages)))


def codex_native_tool_choice_for_request(
    native_tools: Sequence[NativeToolSpec] | None,
    messages: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    """Return a Codex Responses tool_choice override for native hosted tools.

    Hosted tools remain normal ``auto`` tools for ordinary prompts.  For native
    web search, ``tool_choice=auto`` on the API call can still let the model
    answer freshness-sensitive prompts from its parametric knowledge without a
    visible ``web_search_call`` trace.  When configured as ``auto`` we require
    the hosted web_search tool only for requests that look freshness-sensitive;
    ``required`` forces it for every request, and ``none`` disables forcing.
    """

    for spec in native_tools or ():
        if spec.provider_tool_type != "web_search":
            continue
        if spec.tool_choice == "none":
            continue
        if spec.tool_choice == "required" or is_freshness_sensitive_request(messages):
            return {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "web_search"}],
            }
    return None


def _is_codex_responses_backend(*, provider: str | None, api_mode: str | None, base_url: str | None) -> bool:
    if api_mode and str(api_mode).strip() != "codex_responses":
        return False
    provider_name = (provider or "").strip().lower()
    if provider_name == "openai-codex":
        return True
    base = (base_url or "").strip().lower()
    return "chatgpt.com" in base and "/backend-api/codex" in base


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def resolve_provider_native_tools(
    config: Mapping[str, Any] | None,
    *,
    provider: str | None,
    api_mode: str | None,
    base_url: str | None = None,
) -> list[NativeToolSpec]:
    """Resolve configured provider-native tools for the active provider.

    Supports the established short Codex config shape::

        codex:
          web_search: live   # disabled | cached | live

    and a generic future-friendly shape::

        provider_native_tools:
          openai-codex:
            web_search: live
    """

    cfg = _mapping(config)
    if not _is_codex_responses_backend(provider=provider, api_mode=api_mode, base_url=base_url):
        return []

    generic_cfg = _mapping(_mapping(cfg.get("provider_native_tools")).get("openai-codex"))
    codex_cfg = _mapping(cfg.get("codex"))

    specs: list[NativeToolSpec] = []

    web_search_value = generic_cfg.get("web_search", codex_cfg.get("web_search"))
    web_search_mode = _normalize_mode(web_search_value)
    web_search_tool_choice = _normalize_tool_choice_mode(
        generic_cfg.get("web_search_tool_choice", codex_cfg.get("web_search_tool_choice"))
    )
    if web_search_mode == "enabled":
        web_search_mode = "live"
    if web_search_mode in {"live", "cached"}:
        specs.append(
            NativeToolSpec(
                logical_name="web_search",
                provider_tool_type="web_search",
                mode=web_search_mode,
                suppress_managed_tools=("web_search",),
                include=("web_search_call.action.sources",),
                tool_choice=web_search_tool_choice,
            )
        )

    image_generation_value = generic_cfg.get(
        "image_generation",
        codex_cfg.get("image_generation", codex_cfg.get("image_generate")),
    )
    image_generation_mode = _normalize_mode(image_generation_value)
    if image_generation_mode == "live":
        image_generation_mode = "enabled"
    if image_generation_mode == "enabled":
        specs.append(
            NativeToolSpec(
                logical_name="image_generate",
                provider_tool_type="image_generation",
                mode="enabled",
                suppress_managed_tools=("image_generate",),
            )
        )

    return specs


def filter_managed_tools_for_native_tools(
    tools: Sequence[Mapping[str, Any]] | None,
    native_tools: Sequence[NativeToolSpec] | None,
) -> list[Mapping[str, Any]] | None:
    """Return managed function tools with native-tool conflicts removed."""

    if tools is None:
        return None
    suppress = {
        name
        for spec in native_tools or ()
        for name in spec.suppress_managed_tools
        if name
    }
    if not suppress:
        return list(tools)

    filtered: list[Mapping[str, Any]] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        if name in suppress:
            continue
        filtered.append(tool)
    return filtered


def codex_responses_tool_for_native_tool(spec: NativeToolSpec) -> dict[str, Any] | None:
    """Map a native tool spec to an OpenAI/Codex Responses hosted tool."""

    if spec.provider_tool_type == "web_search":
        return {
            "type": "web_search",
            "external_web_access": spec.mode == "live",
        }
    if spec.provider_tool_type == "image_generation":
        return {"type": "image_generation"}
    return None


def codex_responses_tools_for_native_tools(native_tools: Iterable[NativeToolSpec]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for spec in native_tools:
        tool = codex_responses_tool_for_native_tool(spec)
        if tool is not None:
            tools.append(tool)
    return tools


def merge_include_values(existing: Any, additions: Iterable[str]) -> list[str]:
    merged: list[str] = []
    if isinstance(existing, list):
        for item in existing:
            if isinstance(item, str) and item not in merged:
                merged.append(item)
    for item in additions:
        if isinstance(item, str) and item and item not in merged:
            merged.append(item)
    return merged
