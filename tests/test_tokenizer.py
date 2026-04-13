from nl_code.evaluation.tokenizer import token_set, tokenize


class TestTokenize:
    def test_basic_words(self) -> None:
        assert tokenize("hello world") == ["hello", "world"]

    def test_snake_case(self) -> None:
        result = tokenize("has_close_elements")
        assert result == ["has", "close", "elements"]

    def test_camel_case(self) -> None:
        result = tokenize("hasCloseElements")
        assert result == ["has", "close", "elements"]

    def test_pascal_case(self) -> None:
        result = tokenize("HasCloseElements")
        assert result == ["has", "close", "elements"]

    def test_acronym_split(self) -> None:
        result = tokenize("parseHTTPResponse")
        assert result == ["parse", "http", "response"]

    def test_min_length_filters_short(self) -> None:
        result = tokenize("a b cd ef")
        assert result == ["cd", "ef"]

    def test_min_length_one(self) -> None:
        result = tokenize("a b cd", min_length=1)
        assert result == ["a", "b", "cd"]

    def test_code_with_operators(self) -> None:
        result = tokenize("def foo(x): return x + 1")
        assert "def" in result
        assert "foo" in result
        assert "return" in result

    def test_empty_string(self) -> None:
        assert tokenize("") == []

    def test_only_punctuation(self) -> None:
        assert tokenize("!@#$%") == []


class TestTokenSet:
    def test_returns_unique(self) -> None:
        result = token_set("hello hello world")
        assert result == {"hello", "world"}

    def test_returns_set_type(self) -> None:
        result = token_set("hello")
        assert isinstance(result, set)
