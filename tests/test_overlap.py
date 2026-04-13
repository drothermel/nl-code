import pytest

from nl_code.evaluation.overlap import lexical_overlap


class TestLexicalOverlap:
    def test_identical_texts(self) -> None:
        result = lexical_overlap("hello world", "hello world")
        assert result.jaccard == 1.0
        assert result.overlap_a == 1.0
        assert result.overlap_b == 1.0
        assert result.only_a == frozenset()
        assert result.only_b == frozenset()

    def test_disjoint_texts(self) -> None:
        result = lexical_overlap("hello world", "foo bar")
        assert result.jaccard == 0.0
        assert result.overlap_a == 0.0
        assert result.overlap_b == 0.0
        assert result.shared == frozenset()

    def test_partial_overlap(self) -> None:
        result = lexical_overlap("hello world foo", "hello bar foo")
        assert result.shared == frozenset({"hello", "foo"})
        assert result.only_a == frozenset({"world"})
        assert result.only_b == frozenset({"bar"})
        assert result.jaccard == pytest.approx(2 / 4)

    def test_asymmetric_overlap(self) -> None:
        result = lexical_overlap("hello", "hello world")
        assert result.overlap_a == 1.0  # all of A's tokens in B
        assert result.overlap_b == 0.5  # half of B's tokens in A

    def test_code_vs_description(self) -> None:
        code = "def has_close_elements(numbers, threshold):\n    return True\n"
        desc = "Check if any two elements in the numbers list are close"
        result = lexical_overlap(code, desc)
        assert "numbers" in result.shared
        assert "close" in result.shared
        assert "elements" in result.shared

    def test_empty_inputs(self) -> None:
        result = lexical_overlap("", "")
        assert result.jaccard == 0.0
        assert result.overlap_a == 0.0
        assert result.overlap_b == 0.0

    def test_result_is_frozen(self) -> None:
        result = lexical_overlap("hello world", "hello")
        with pytest.raises(Exception):
            result.jaccard = 0.5  # type: ignore[misc]
