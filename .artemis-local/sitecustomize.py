"""Local-only Artemis bootstrap for free CI/emulator runs.

Enabled only when ARTEMIS_LOCAL_ONLY=1. Normal Artemis provider behavior is
unchanged when the flag is absent.
"""

import os

if os.environ.get("ARTEMIS_LOCAL_ONLY") == "1":
    from artemis.llm.router import ModelEndpoint, ModelFactory, ModelProvider
    import artemis.config.llm as config_llm
    import artemis.services.llm as service_llm

    def _local_model_name() -> str:
        return os.environ.get("ARTEMIS_LOCAL_MODEL", "qwen3-vl:2b-instruct")

    def _local_endpoint(*, temperature: float = 0.0, timeout: float = 60.0):
        return ModelEndpoint(
            provider=ModelProvider.OLLAMA,
            model_name=_local_model_name(),
            temperature=temperature,
            timeout_seconds=timeout,
            thinking_budget=None,
            thinking_level=None,
            reasoning_effort=None,
            include_thoughts=None,
            enable_grounding=False,
        )

    def _validate_provider(self, name: str) -> None:
        return None

    config_llm.LLM.validate_provider = _validate_provider

    _original_resolve_endpoint = service_llm._resolve_endpoint

    def _resolve_endpoint_local(ctx, name, is_utils=False, use_fallback=False):
        endpoint = _original_resolve_endpoint(
            ctx, name, is_utils=is_utils, use_fallback=use_fallback
        )
        return endpoint.model_copy(
            update={
                "provider": ModelProvider.OLLAMA,
                "model_name": _local_model_name(),
                "thinking_budget": None,
                "thinking_level": None,
                "reasoning_effort": None,
                "enable_grounding": False,
            }
        )

    service_llm._resolve_endpoint = _resolve_endpoint_local

    def _get_google_llm_local(
        model_name: str = "gemini-3.8-flash",
        temperature: float | None = None,
        timeout: float | None = None,
        **_kwargs,
    ):
        return ModelFactory.create_model(
            _local_endpoint(temperature=temperature or 0.0, timeout=timeout or 60.0)
        )

    service_llm.get_google_llm = _get_google_llm_local
    service_llm.get_vertex_llm = _get_google_llm_local

    def _get_cached_raw_model_local(
        provider: str,
        model_name: str,
        temperature: float | None = None,
        timeout: float | None = None,
        **_kwargs,
    ):
        return ModelFactory.get_model(
            _local_endpoint(temperature=temperature or 0.0, timeout=timeout or 60.0)
        )

    service_llm.get_cached_raw_model = _get_cached_raw_model_local

    # Force structured tool selection for local OpenAI-compatible Qwen endpoints.
    from langchain_core.language_models.chat_models import BaseChatModel
    from artemis.agents.flash.runner import FlashRunner
    import json
    import re

    _original_bind_tools = BaseChatModel.bind_tools
    def _bind_tools_local(self, tools, *args, **kwargs):
        if tools and "tool_choice" not in kwargs:
            kwargs["tool_choice"] = "required"
        return _original_bind_tools(self, tools, *args, **kwargs)
    BaseChatModel.bind_tools = _bind_tools_local

    _original_init = FlashRunner.__init__
    def _flash_init_local(self, ctx, goal, max_turns=None):
        if max_turns is None or max_turns <= 0:
            max_turns = 6
        return _original_init(self, ctx, goal, max_turns)
    FlashRunner.__init__ = _flash_init_local

    _original_resolve = FlashRunner._resolve_tool_calls
    def _resolve_tool_calls_local(self, response, raw_text):
        calls = _original_resolve(self, response, raw_text)
        if calls:
            return calls
        found = []
        for match in re.finditer(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
            raw_text or "",
            flags=re.DOTALL,
        ):
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            function = payload.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                args = function.get("arguments", {})
            else:
                name = payload.get("name")
                args = payload.get("arguments", payload.get("args", {}))
            if not isinstance(name, str) or not isinstance(args, (dict, str)):
                continue
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if not isinstance(args, dict):
                continue
            found.append({"name": name, "args": args, "id": str(payload.get("id") or os.urandom(8).hex())})
        return found
    FlashRunner._resolve_tool_calls = _resolve_tool_calls_local

    _original_prompt = FlashRunner._render_system_prompt
    def _render_system_prompt_local(self, tools_declaration):
        prompt = _original_prompt(self, tools_declaration)
        prompt += (
            "\n\nLOCAL QWEN EXECUTION RULES:\n"
            "Every action turn MUST use one of the provided tools. "
            "Use manage_app for device actions. Use report_task_status only "
            "after the requested task is actually complete. Do not answer "
            "an action turn with plain prose. Native tool metadata or a "
            "tool_call XML block is required."
        )
        return prompt
    FlashRunner._render_system_prompt = _render_system_prompt_local
