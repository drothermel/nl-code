from __future__ import annotations

import os
from typing import Any

import dspy
from pydantic import BaseModel, ConfigDict

DEFAULT_CODE_GENERATION_INSTRUCTIONS = (
    "Implement the requested function using the provided specification. "
    "Return only executable Python code. Do not include explanations or Markdown."
)
DEFAULT_ENCODER_INSTRUCTIONS = (
    "Provide a concise natural language description of the code. "
    "Do not output anything else."
)
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_CONFIG_ID = "openrouter/xiaomi/mimo-v2-flash/off/v1"
DEFAULT_REASONING_EFFORT = "none"


class DspyLmCatalogConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    reasoning: dict[str, str | bool] | None = None


SUPPORTED_OPENROUTER_LLM_CONFIGS: dict[str, DspyLmCatalogConfig] = {
    "openrouter/deepseek/deepseek-chat-v3.1/off/v1": DspyLmCatalogConfig(
        model="openrouter/deepseek/deepseek-chat-v3.1",
        reasoning={"enabled": False},
    ),
    "openrouter/xiaomi/mimo-v2-flash/off/v1": DspyLmCatalogConfig(
        model="openrouter/xiaomi/mimo-v2-flash",
        reasoning={"enabled": False},
    ),
    "openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5/off/v1": DspyLmCatalogConfig(
        model="openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5",
        reasoning={"enabled": False},
    ),
    "openrouter/baidu/ernie-4.5-21b-a3b/na/v1": DspyLmCatalogConfig(
        model="openrouter/baidu/ernie-4.5-21b-a3b",
    ),
    "openrouter/bytedance-seed/seed-2.0-mini/off/v1": DspyLmCatalogConfig(
        model="openrouter/bytedance-seed/seed-2.0-mini",
        reasoning={"enabled": False},
    ),
    "openrouter/mistralai/devstral-small/na/v1": DspyLmCatalogConfig(
        model="openrouter/mistralai/devstral-small",
    ),
    "openrouter/meta-llama/llama-4-scout/na/v1": DspyLmCatalogConfig(
        model="openrouter/meta-llama/llama-4-scout",
    ),
    "openrouter/qwen/qwen3-coder-30b-a3b-instruct/na/v1": DspyLmCatalogConfig(
        model="openrouter/qwen/qwen3-coder-30b-a3b-instruct",
    ),
}


DEFAULT_DSPY_MODEL = SUPPORTED_OPENROUTER_LLM_CONFIGS[DEFAULT_LLM_CONFIG_ID].model


class ResolvedDspyLmSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    llm_config_id: str | None
    reasoning_effort: str | None
    reasoning_config: dict[str, str | bool] | None


def resolve_dspy_lm_settings(
    *,
    model: str | None = None,
    llm_config_id: str | None = None,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    reasoning_config: dict[str, str | bool] | None = None,
) -> ResolvedDspyLmSettings:
    if llm_config_id is not None:
        catalog = resolve_openrouter_llm_config(llm_config_id)
        if reasoning_config is not None and reasoning_config != catalog.reasoning:
            raise ValueError(
                "reasoning_config conflicts with llm_config_id catalog entry"
            )
        if reasoning_effort not in (None, "none") and catalog.reasoning is not None:
            raise ValueError(
                "reasoning_effort cannot be combined with llm_config_id catalog entry"
            )
        return ResolvedDspyLmSettings(
            model=catalog.model,
            llm_config_id=llm_config_id,
            reasoning_effort=None,
            reasoning_config=catalog.reasoning,
        )

    effective_model = model if model is not None else DEFAULT_DSPY_MODEL
    if effective_model != DEFAULT_DSPY_MODEL:
        return ResolvedDspyLmSettings(
            model=effective_model,
            llm_config_id=None,
            reasoning_effort=reasoning_effort,
            reasoning_config=reasoning_config,
        )

    default_catalog = resolve_openrouter_llm_config(DEFAULT_LLM_CONFIG_ID)
    return ResolvedDspyLmSettings(
        model=default_catalog.model,
        llm_config_id=DEFAULT_LLM_CONFIG_ID,
        reasoning_effort=None,
        reasoning_config=default_catalog.reasoning,
    )


def supported_openrouter_llm_config_ids() -> tuple[str, ...]:
    return tuple(SUPPORTED_OPENROUTER_LLM_CONFIGS)


def resolve_openrouter_llm_config(config_id: str) -> DspyLmCatalogConfig:
    try:
        return SUPPORTED_OPENROUTER_LLM_CONFIGS[config_id]
    except KeyError as exc:
        supported = ", ".join(supported_openrouter_llm_config_ids())
        raise ValueError(
            f"unsupported --llm-config-id {config_id!r}; supported values: {supported}"
        ) from exc


def reasoning_from_effort(effort: str | None) -> dict[str, str] | None:
    if effort is None or effort == "none":
        return None
    return {"effort": effort}


class CompleteCodeFromStub(dspy.Signature):
    code_stub: str = dspy.InputField(
        desc=(
            "Partial Python source containing imports, function signature, "
            "and any available docstring or comments to complete."
        )
    )
    completed_code: str = dspy.OutputField(
        desc="Complete executable Python source code."
    )


CompleteCodeFromStub = CompleteCodeFromStub.with_instructions(  # ty: ignore[invalid-assignment]
    DEFAULT_CODE_GENERATION_INSTRUCTIONS
)


class DirectCodeGenerator(dspy.Module):
    def __init__(self, signature: type[dspy.Signature] = CompleteCodeFromStub) -> None:
        super().__init__()
        self.complete = dspy.Predict(signature)

    def forward(self, code_stub: str) -> Any:
        return self.complete(code_stub=code_stub)


class EncodeCodeSpec(dspy.Signature):
    input_code: str = dspy.InputField(desc="Complete Python source code to describe.")
    code_spec: str = dspy.OutputField(
        desc="Concise natural-language behavior specification for the code."
    )


EncodeCodeSpec = EncodeCodeSpec.with_instructions(  # ty: ignore[invalid-assignment]
    DEFAULT_ENCODER_INSTRUCTIONS
)


class DecodeCodeSpec(dspy.Signature):
    code_spec: str = dspy.InputField(
        desc="Natural-language behavior specification for the requested function."
    )
    function_stub: str = dspy.InputField(
        desc=(
            "Python imports and function signature to complete; comments and "
            "docstrings are intentionally omitted."
        )
    )
    completed_code: str = dspy.OutputField(
        desc="Complete executable Python source code."
    )


DecodeCodeSpec = DecodeCodeSpec.with_instructions(  # ty: ignore[invalid-assignment]
    DEFAULT_CODE_GENERATION_INSTRUCTIONS
)


class CodeSpecEncoder(dspy.Module):
    def __init__(self, signature: type[dspy.Signature] = EncodeCodeSpec) -> None:
        super().__init__()
        self.encode = dspy.Predict(signature)

    def forward(self, input_code: str) -> Any:
        return self.encode(input_code=input_code)


class CodeSpecDecoder(dspy.Module):
    def __init__(self, signature: type[dspy.Signature] = DecodeCodeSpec) -> None:
        super().__init__()
        self.decode = dspy.Predict(signature)

    def forward(self, code_spec: str, function_stub: str) -> Any:
        return self.decode(code_spec=code_spec, function_stub=function_stub)


class EncoderDecoderCodeGenerator(dspy.Module):
    def __init__(
        self,
        encoder: CodeSpecEncoder | None = None,
        decoder: CodeSpecDecoder | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder or CodeSpecEncoder()
        self.decoder = decoder or CodeSpecDecoder()

    def forward(self, input_code: str, function_stub: str) -> Any:
        encoded = self.encoder(input_code=input_code)
        decoded = self.decoder(
            code_spec=encoded.code_spec,
            function_stub=function_stub,
        )
        return dspy.Prediction(
            code_spec=encoded.code_spec,
            completed_code=decoded.completed_code,
        )


def configure_dspy_lm(
    *,
    model: str | None = None,
    llm_config_id: str | None = None,
    api_key: str,
    api_base: str = DEFAULT_OPENROUTER_BASE_URL,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    reasoning: dict[str, str | bool] | None = None,
    disable_cache: bool = True,
) -> Any:
    resolved = resolve_dspy_lm_settings(
        model=model,
        llm_config_id=llm_config_id,
        reasoning_effort=reasoning_effort,
        reasoning_config=reasoning,
    )
    os.environ["OPENROUTER_API_KEY"] = api_key
    os.environ["OPENROUTER_API_BASE"] = api_base
    os.environ["OPENAI_API_KEY"] = api_key
    lm = dspy.LM(
        resolved.model,
        api_key=api_key,
        api_base=api_base,
        reasoning=resolved.reasoning_config
        if resolved.reasoning_config is not None
        else reasoning_from_effort(resolved.reasoning_effort),
    )
    dspy.configure(lm=lm)
    if disable_cache:
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    return lm
