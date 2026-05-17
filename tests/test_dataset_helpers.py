import pytest

from nl_code.datasets.text import strip_surrounding_empty_lines
from nl_code.datasets.validation import require_string


def test_require_string_returns_string() -> None:
    assert require_string("value", name="field") == "value"


def test_require_string_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="field must be a string"):
        require_string(1, name="field")


def test_strip_surrounding_empty_lines() -> None:
    assert strip_surrounding_empty_lines("\n\n  hello\n\n") == "  hello"
