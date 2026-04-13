from pydantic import BaseModel, ConfigDict

from nl_code.evaluation.tokenizer import token_set


class OverlapResult(BaseModel):
    """Lexical overlap between two texts."""

    model_config = ConfigDict(frozen=True)

    tokens_a: frozenset[str]
    tokens_b: frozenset[str]
    shared: frozenset[str]
    only_a: frozenset[str]
    only_b: frozenset[str]
    jaccard: float
    overlap_a: float  # fraction of A's tokens that appear in B
    overlap_b: float  # fraction of B's tokens that appear in A


def lexical_overlap(text_a: str, text_b: str, *, min_length: int = 2) -> OverlapResult:
    """Compute lexical overlap between two texts.

    Works for code<->description and description<->description comparisons.
    """
    if not isinstance(min_length, int) or min_length < 1:
        raise ValueError("min_length must be a positive integer")
    set_a = frozenset(token_set(text_a, min_length=min_length))
    set_b = frozenset(token_set(text_b, min_length=min_length))
    shared = set_a & set_b
    union = set_a | set_b

    return OverlapResult(
        tokens_a=set_a,
        tokens_b=set_b,
        shared=shared,
        only_a=set_a - set_b,
        only_b=set_b - set_a,
        jaccard=len(shared) / len(union) if union else 0.0,
        overlap_a=len(shared) / len(set_a) if set_a else 0.0,
        overlap_b=len(shared) / len(set_b) if set_b else 0.0,
    )
