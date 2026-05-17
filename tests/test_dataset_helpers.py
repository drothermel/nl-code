import pytest

from nl_code.datasets.collections import normalize_sequence_index
from nl_code.datasets.text import strip_surrounding_empty_lines
from nl_code.datasets.validation import require_string


def test_require_string_returns_string() -> None:
    assert require_string("value", name="field") == "value"


def test_require_string_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="field must be a string"):
        require_string(1, name="field")


def test_strip_surrounding_empty_lines() -> None:
    assert strip_surrounding_empty_lines("\n\n  hello\n\n") == "  hello"


def test_normalize_sequence_index() -> None:
    assert normalize_sequence_index(1, 3, collection_name="item") == 1
    assert normalize_sequence_index(-1, 3, collection_name="item") == 2


def test_normalize_sequence_index_rejects_out_of_range() -> None:
    with pytest.raises(IndexError, match="item index 3 out of range for 3 items"):
        normalize_sequence_index(3, 3, collection_name="item")
