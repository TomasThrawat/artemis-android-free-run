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
