import re

_SPLIT_PATTERN = re.compile(r"[^a-zA-Z0-9]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str, *, min_length: int = 2) -> list[str]:
    """Split text into lowercase tokens on non-alphanumeric boundaries.

    Also splits camelCase and PascalCase identifiers. Filters tokens
    shorter than min_length.
    """
    if not isinstance(min_length, int) or min_length < 1:
        raise ValueError("min_length must be a positive integer")
    raw_tokens = _SPLIT_PATTERN.split(text)
    tokens: list[str] = []
    for raw in raw_tokens:
        sub_tokens = _CAMEL_BOUNDARY.split(raw)
        tokens.extend(t.lower() for t in sub_tokens if t and len(t) >= min_length)
    return tokens


def token_set(text: str, *, min_length: int = 2) -> set[str]:
    """Return the unique set of tokens from text."""
    return set(tokenize(text, min_length=min_length))
