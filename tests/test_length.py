import pytest

from nl_code.evaluation.length import compression_ratio, measure_length


class TestMeasureLength:
    def test_char_count(self) -> None:
        result = measure_length("hello")
        assert result.char_count == 5

    def test_token_count(self) -> None:
        result = measure_length("hello world foo")
        assert result.token_count == 3

    def test_empty_string(self) -> None:
        result = measure_length("")
        assert result.char_count == 0
        assert result.token_count == 0


class TestCompressionRatio:
    def test_basic_ratio(self) -> None:
        desc = "a short description here"
        code = "def foo():\n    return bar\n"
        result = compression_ratio(description=desc, code=code)
        assert result.description.char_count == len(desc)
        assert result.code.char_count == len(code)
        assert result.char_ratio == pytest.approx(len(desc) / len(code))

    def test_empty_code(self) -> None:
        result = compression_ratio(description="some text", code="")
        assert result.char_ratio == 0.0
        assert result.token_ratio == 0.0

    def test_description_longer_than_code(self) -> None:
        result = compression_ratio(
            description="a very long description of what this does",
            code="x = 1",
        )
        assert result.char_ratio > 1.0
