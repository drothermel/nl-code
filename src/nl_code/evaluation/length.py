from pydantic import BaseModel

from nl_code.evaluation.tokenizer import tokenize


class LengthMetrics(BaseModel):
    """Length measurements for a text."""

    char_count: int
    token_count: int


class CompressionRatio(BaseModel):
    """Ratio of description length to code length."""

    description: LengthMetrics
    code: LengthMetrics
    char_ratio: float
    token_ratio: float


def measure_length(text: str) -> LengthMetrics:
    return LengthMetrics(
        char_count=len(text),
        token_count=len(tokenize(text)),
    )


def compression_ratio(description: str, code: str) -> CompressionRatio:
    desc_metrics = measure_length(description)
    code_metrics = measure_length(code)
    return CompressionRatio(
        description=desc_metrics,
        code=code_metrics,
        char_ratio=(
            desc_metrics.char_count / code_metrics.char_count
            if code_metrics.char_count
            else 0.0
        ),
        token_ratio=(
            desc_metrics.token_count / code_metrics.token_count
            if code_metrics.token_count
            else 0.0
        ),
    )
