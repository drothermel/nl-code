import ast
import io
import tokenize
from typing import Any


def _parse_source(source: str, *, allow_stub_body: bool = False) -> ast.Module:
    if allow_stub_body:
        return parse_source_with_stub_body(source)
    return ast.parse(source)


def merge_code_components(*components: str) -> str:
    return "\n".join(component.rstrip() for component in components) + "\n"


def parse_source_with_stub_body(source: str) -> ast.Module:
    try:
        return ast.parse(source)
    except SyntaxError:
        stripped_source = source.rstrip()
        if stripped_source.endswith(":"):
            return ast.parse(stripped_source + "\n    pass\n")
        if stripped_source.endswith("..."):
            return ast.parse(stripped_source + "\n")
        raise


def find_docstring_expr(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Expr | None:
    if not node.body:
        return None

    first_stmt = node.body[0]
    if not isinstance(first_stmt, ast.Expr):
        return None
    if not isinstance(first_stmt.value, ast.Constant):
        return None
    if not isinstance(first_stmt.value.value, str):
        return None
    return first_stmt


class _DocstringStripper(ast.NodeTransformer):
    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    @staticmethod
    def _strip_docstring(
        body: list[ast.stmt],
    ) -> list[ast.stmt]:
        body_node = ast.Module(body=body, type_ignores=[])
        if find_docstring_expr(body_node) is None:
            return body
        return body[1:]


def remove_docstrings_and_comments(source: str) -> str:
    tree = ast.parse(source)
    stripped_tree = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(stripped_tree)
    return ast.unparse(stripped_tree).rstrip() + "\n"


def remove_docstrings_preserving_comments(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source.rstrip() + "\n" if source.strip() else ""
    line_numbers_to_remove: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            docstring_expr = find_docstring_expr(node)
            if docstring_expr is None:
                continue
            if docstring_expr.lineno is None or docstring_expr.end_lineno is None:
                raise RuntimeError("docstring node is missing line numbers")
            line_numbers_to_remove.update(
                range(docstring_expr.lineno, docstring_expr.end_lineno + 1)
            )

    remaining_lines = [
        line
        for line_number, line in enumerate(source.splitlines(keepends=True), start=1)
        if line_number not in line_numbers_to_remove
    ]
    if not remaining_lines:
        return ""

    return "".join(remaining_lines).rstrip() + "\n"


def remove_full_line_comments(source: str) -> str:
    uncommented_lines = [
        line
        for line in source.splitlines(keepends=True)
        if not line.lstrip().startswith("#")
    ]
    uncommented = "".join(uncommented_lines)
    if not uncommented.strip():
        return ""
    return uncommented.rstrip() + "\n"


def line_col_to_index(source: str, line_number: int, col_offset: int) -> int:
    lines = source.splitlines(keepends=True)
    line = lines[line_number - 1]
    col_text = line.encode("utf-8")[:col_offset].decode("utf-8")
    return sum(len(line) for line in lines[: line_number - 1]) + len(col_text)


def node_span(source: str, node: ast.AST) -> tuple[int, int]:
    lineno = getattr(node, "lineno", None)
    col_offset = getattr(node, "col_offset", None)
    end_lineno = getattr(node, "end_lineno", None)
    end_col_offset = getattr(node, "end_col_offset", None)
    if (
        not isinstance(lineno, int)
        or not isinstance(col_offset, int)
        or not isinstance(end_lineno, int)
        or not isinstance(end_col_offset, int)
    ):
        raise ValueError(f"{type(node).__name__} node does not have source positions")
    return (
        line_col_to_index(source, lineno, col_offset),
        line_col_to_index(source, end_lineno, end_col_offset),
    )


def replace_source_spans(
    source: str,
    replacements: list[tuple[tuple[int, int], str]],
) -> str:
    updated = source
    for (start, end), replacement in sorted(replacements, reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    return updated


def single_item_list_source(source: str, item_node: ast.AST) -> str:
    start, end = node_span(source, item_node)
    return f"[{source[start:end]}]"


def node_references_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id == name for child in ast.walk(node)
    )


def find_first_function_name(source: str) -> str:
    node = find_first_function_node(source)
    return node.name


def find_first_function_node(
    source: str,
    *,
    allow_stub_body: bool = False,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = _parse_source(source, allow_stub_body=allow_stub_body)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    raise ValueError("no function definition found in source")


def find_named_function_in_module(
    tree: ast.Module, function_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ):
            return node
    raise ValueError(f"function {function_name!r} not found")


def find_named_function(
    source: str, function_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    return find_named_function_in_module(ast.parse(source), function_name)


def find_named_assignment_in_body(body: list[ast.stmt], name: str) -> ast.Assign | None:
    for stmt in body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name
        ):
            return stmt
    return None


def literal_eval_assignment_value(assign: ast.Assign) -> object:
    return ast.literal_eval(assign.value)


def literal_list_assignment_in_body(
    body: list[ast.stmt],
    name: str,
) -> tuple[ast.Assign, list[Any], list[ast.expr]]:
    assign = find_named_assignment_in_body(body, name)
    if assign is None:
        raise ValueError(f"no `{name} = ...` assignment found")
    if not isinstance(assign.value, ast.List):
        raise TypeError(f"`{name}` assignment must be a list literal")
    value = literal_eval_assignment_value(assign)
    if not isinstance(value, list):
        raise TypeError(f"`{name}` assignment must evaluate to a list")
    return assign, value, assign.value.elts


def get_docstring(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    return (ast.get_docstring(node) or "").strip()


def get_first_function_docstring(
    source: str,
    *,
    allow_stub_body: bool = False,
) -> str:
    return get_docstring(
        find_first_function_node(source, allow_stub_body=allow_stub_body)
    )


def get_comments(source: str, *, strip_hash: bool = False) -> str | None:
    comments = [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    ]
    if not comments:
        return None
    if strip_hash:
        comments = [
            comment[1:].lstrip() if comment.startswith("#") else comment
            for comment in comments
        ]
    return "\n".join(comments)


def get_docstrings_and_comments(
    source: str,
    *,
    strip_hash: bool = True,
    allow_stub_body: bool = False,
    joiner: str = "\n\n",
) -> str:
    parts = [
        part.strip()
        for part in (
            get_comments(source, strip_hash=strip_hash) or "",
            get_first_function_docstring(source, allow_stub_body=allow_stub_body),
        )
        if part.strip()
    ]
    return joiner.join(parts)


def extract_top_level_import_source(
    source: str,
    *,
    allow_stub_body: bool = False,
) -> str:
    lines = source.splitlines(keepends=True)
    tree = _parse_source(source, allow_stub_body=allow_stub_body)

    import_blocks = [
        "".join(lines[node.lineno - 1 : node.end_lineno])
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and node.lineno is not None
        and node.end_lineno is not None
    ]
    if not import_blocks:
        return ""

    return "".join(import_blocks).rstrip() + "\n"
