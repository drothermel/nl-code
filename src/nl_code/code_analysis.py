import ast
import io
import logging
import re
import tokenize
from typing import Any

from pydantic import BaseModel, Field

from nl_code.code_parsing import find_named_function, get_docstring

logger = logging.getLogger(__name__)

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ControlStructureCounts(BaseModel):
    for_loops: int = 0
    while_loops: int = 0
    if_statements: int = 0
    elif_branches: int = 0
    match_statements: int = 0
    try_blocks: int = 0
    with_statements: int = 0
    list_comprehensions: int = 0
    dict_comprehensions: int = 0
    set_comprehensions: int = 0
    generator_expressions: int = 0


class CodeStyleMetrics(BaseModel):
    total_lines: int = 0
    longest_line_length: int = 0
    avg_line_length: float = 0.0
    variable_names: list[str] = Field(default_factory=list)
    variable_name_lengths: list[int] = Field(default_factory=list)
    num_variables: int = 0
    num_snake_case: int = 0
    num_camel_case: int = 0
    num_single_char: int = 0
    max_nesting_depth: int = 0
    num_magic_numbers: int = 0


class FunctionAnalysis(BaseModel):
    function_name: str
    has_return: bool
    has_print: bool
    has_raise: bool
    has_assert: bool
    return_type_annotation: str | None
    parameter_names: list[str]
    has_docstring: bool
    docstring_text: str
    inline_comments: list[str]
    string_literals: list[str]
    control_structures: ControlStructureCounts
    code_style: CodeStyleMetrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _LocalBodyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes: list[ast.AST] = []

    def visit(self, node: ast.AST) -> Any:
        self.nodes.append(node)
        return super().visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # noqa: ARG002
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:  # noqa: ARG002
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:  # noqa: ARG002
        return None

    def visit_Lambda(self, node: ast.Lambda) -> Any:  # noqa: ARG002
        return None


def _iter_local_body_nodes(node: _FunctionNode) -> list[ast.AST]:
    visitor = _LocalBodyVisitor()
    for stmt in node.body:
        visitor.visit(stmt)
    return visitor.nodes


def _collect_target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in target.elts:
            names.update(_collect_target_names(elt))
        return names
    if isinstance(target, ast.Starred):
        return _collect_target_names(target.value)
    return set()


def _find_function_node(tree: ast.Module, function_name: str) -> _FunctionNode | None:
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ):
            return node
    return None


def _is_snake_case(name: str) -> bool:
    stripped = name.lstrip("_")
    if not stripped:
        return False
    return stripped.islower()


def _is_camel_case(name: str) -> bool:
    if not name:
        return False
    return name[0].islower() and any(c.isupper() for c in name)


def _get_nesting_depth(
    node: ast.AST, current_depth: int = 0, *, is_elif: bool = False
) -> int:
    nesting_nodes = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Match,
    )

    if isinstance(node, nesting_nodes) and not is_elif:
        current_depth += 1

    max_depth = current_depth

    if isinstance(node, ast.If):
        for body_node in node.body:
            max_depth = max(max_depth, _get_nesting_depth(body_node, current_depth))
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                max_depth = max(
                    max_depth,
                    _get_nesting_depth(node.orelse[0], current_depth, is_elif=True),
                )
            else:
                for else_node in node.orelse:
                    max_depth = max(
                        max_depth, _get_nesting_depth(else_node, current_depth)
                    )
    else:
        for child in ast.iter_child_nodes(node):
            max_depth = max(max_depth, _get_nesting_depth(child, current_depth))

    return max_depth


# ---------------------------------------------------------------------------
# Standalone utilities (not function-scoped)
# ---------------------------------------------------------------------------


def extract_from_code_fences(text: str) -> tuple[str, bool]:
    """Extract code from markdown code fences. Returns (code, had_fences)."""
    pattern = r"```(?:\w+)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip(), True
    return text.strip(), False


def check_python_syntax(code: str) -> tuple[bool, str | None]:
    """Check if code is valid Python. Returns (valid, error_message)."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"
    return True, None


def check_function_exists(code: str, function_name: str) -> bool:
    """Check whether a top-level function with the given name exists in the code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return _find_function_node(tree, function_name) is not None


# ---------------------------------------------------------------------------
# Per-function structural checks
# ---------------------------------------------------------------------------


def check_has_return(code: str, function_name: str) -> bool | None:
    try:
        node = find_named_function(code, function_name)
    except ValueError:
        return None
    for child in _iter_local_body_nodes(node):
        if isinstance(child, ast.Return):
            return True
    return False


def check_has_print(code: str, function_name: str) -> bool | None:
    try:
        node = find_named_function(code, function_name)
    except ValueError:
        return None
    for child in _iter_local_body_nodes(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "print"
        ):
            return True
    return False


def check_has_raise(code: str, function_name: str) -> bool | None:
    try:
        node = find_named_function(code, function_name)
    except ValueError:
        return None
    for child in _iter_local_body_nodes(node):
        if isinstance(child, ast.Raise):
            return True
    return False


def check_has_assert(code: str, function_name: str) -> bool | None:
    try:
        node = find_named_function(code, function_name)
    except ValueError:
        return None
    for child in _iter_local_body_nodes(node):
        if isinstance(child, ast.Assert):
            return True
    return False


def get_return_type_annotation(code: str, function_name: str) -> str | None:
    try:
        node = find_named_function(code, function_name)
    except ValueError:
        return None
    if node.returns is not None:
        return ast.unparse(node.returns)
    return None


def get_parameter_names(code: str, function_name: str) -> list[str] | None:
    try:
        node = find_named_function(code, function_name)
    except ValueError:
        return None
    names: list[str] = []
    for arg in node.args.posonlyargs:
        names.append(arg.arg)
    for arg in node.args.args:
        names.append(arg.arg)
    if node.args.vararg:
        names.append(f"*{node.args.vararg.arg}")
    for arg in node.args.kwonlyargs:
        names.append(arg.arg)
    if node.args.kwarg:
        names.append(f"**{node.args.kwarg.arg}")
    return names


def extract_inline_comments(code: str, function_name: str) -> list[str] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    func_node = _find_function_node(tree, function_name)
    if func_node is None:
        return None

    start_line = func_node.lineno
    end_line = func_node.end_lineno or start_line

    comments: list[str] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT and start_line <= tok.start[0] <= end_line:
                comments.append(tok.string.lstrip("#").strip())
    except tokenize.TokenError:
        logger.warning(
            "Skipping inline comment extraction for %s due to tokenization error",
            function_name,
        )

    return comments


def extract_string_literals(code: str, function_name: str) -> list[str] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    func_node = _find_function_node(tree, function_name)
    if func_node is None:
        return None

    has_docstring = (
        func_node.body
        and isinstance(func_node.body[0], ast.Expr)
        and isinstance(func_node.body[0].value, ast.Constant)
        and isinstance(func_node.body[0].value.value, str)
    )

    parent_map: dict[int, ast.AST] = {}
    for parent in ast.walk(func_node):
        for child_node in ast.iter_child_nodes(parent):
            parent_map[id(child_node)] = parent

    strings_with_pos: list[tuple[int, int, str]] = []
    for child in ast.walk(func_node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if isinstance(parent_map.get(id(child)), ast.JoinedStr):
                continue
            first_stmt = func_node.body[0]
            if (
                has_docstring
                and isinstance(first_stmt, ast.Expr)
                and child is first_stmt.value
            ):
                continue
            strings_with_pos.append((child.lineno, child.col_offset, child.value))
        elif isinstance(child, ast.JoinedStr):
            for part in child.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    strings_with_pos.append((part.lineno, part.col_offset, part.value))

    strings_with_pos.sort(key=lambda x: (x[0], x[1]))
    return [s[2] for s in strings_with_pos]


def count_control_structures(
    code: str, function_name: str
) -> ControlStructureCounts | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    node = _find_function_node(tree, function_name)
    if node is None:
        return None

    local_nodes = _iter_local_body_nodes(node)
    counts = ControlStructureCounts()

    elif_nodes: set[int] = set()
    for child in local_nodes:
        if isinstance(child, ast.If):
            current = child
            while (
                current.orelse
                and len(current.orelse) == 1
                and isinstance(current.orelse[0], ast.If)
            ):
                elif_nodes.add(id(current.orelse[0]))
                current = current.orelse[0]

    for child in local_nodes:
        if isinstance(child, (ast.For, ast.AsyncFor)):
            counts.for_loops += 1
        elif isinstance(child, ast.While):
            counts.while_loops += 1
        elif isinstance(child, ast.If):
            if id(child) in elif_nodes:
                counts.elif_branches += 1
            else:
                counts.if_statements += 1
        elif isinstance(child, ast.Match):
            counts.match_statements += 1
        elif isinstance(child, ast.Try):
            counts.try_blocks += 1
        elif isinstance(child, (ast.With, ast.AsyncWith)):
            counts.with_statements += 1
        elif isinstance(child, ast.ListComp):
            counts.list_comprehensions += 1
        elif isinstance(child, ast.DictComp):
            counts.dict_comprehensions += 1
        elif isinstance(child, ast.SetComp):
            counts.set_comprehensions += 1
        elif isinstance(child, ast.GeneratorExp):
            counts.generator_expressions += 1

    return counts


def analyze_code_style(code: str, function_name: str) -> CodeStyleMetrics | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    func_node = _find_function_node(tree, function_name)
    if func_node is None:
        return None

    metrics = CodeStyleMetrics()

    start_line = func_node.lineno
    end_line = func_node.end_lineno or start_line
    lines = code.split("\n")[start_line - 1 : end_line]
    metrics.total_lines = len(lines)
    if lines:
        line_lengths = [len(line) for line in lines]
        metrics.longest_line_length = max(line_lengths)
        metrics.avg_line_length = sum(line_lengths) / len(line_lengths)

    variable_names: set[str] = set()
    for child in _iter_local_body_nodes(func_node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            variable_names.add(child.id)
        elif isinstance(child, ast.For):
            variable_names.update(_collect_target_names(child.target))
        elif isinstance(child, ast.NamedExpr) and isinstance(child.target, ast.Name):
            variable_names.add(child.target.id)

    metrics.variable_names = sorted(variable_names)
    metrics.variable_name_lengths = [len(name) for name in metrics.variable_names]
    metrics.num_variables = len(variable_names)
    metrics.num_snake_case = sum(1 for name in variable_names if _is_snake_case(name))
    metrics.num_camel_case = sum(1 for name in variable_names if _is_camel_case(name))
    metrics.num_single_char = sum(1 for name in variable_names if len(name) == 1)

    metrics.max_nesting_depth = _get_nesting_depth(func_node)

    # Build parent map to avoid double-counting negated literals.
    # Without this, `-42` would count both the UnaryOp and the inner Constant.
    negated_operand_ids: set[int] = set()
    local_nodes = _iter_local_body_nodes(func_node)
    for child in local_nodes:
        if isinstance(child, ast.UnaryOp) and isinstance(child.op, ast.USub):
            negated_operand_ids.add(id(child.operand))

    for child in local_nodes:
        if isinstance(child, ast.UnaryOp) and isinstance(child.op, ast.USub):
            if isinstance(child.operand, ast.Constant) and isinstance(
                child.operand.value, int | float
            ):
                signed_value = -child.operand.value
                if signed_value not in (0, 1, -1, 0.0, 1.0, -1.0):
                    metrics.num_magic_numbers += 1
        elif isinstance(child, ast.Constant) and isinstance(child.value, int | float):
            if id(child) not in negated_operand_ids and child.value not in (
                0,
                1,
                -1,
                0.0,
                1.0,
                -1.0,
            ):
                metrics.num_magic_numbers += 1

    return metrics


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def analyze_function(code: str, function_name: str) -> FunctionAnalysis:
    """Parse once and run all structural analyses on a function."""
    node = find_named_function(code, function_name)

    names: list[str] = []
    for arg in node.args.posonlyargs:
        names.append(arg.arg)
    for arg in node.args.args:
        names.append(arg.arg)
    if node.args.vararg:
        names.append(f"*{node.args.vararg.arg}")
    for arg in node.args.kwonlyargs:
        names.append(arg.arg)
    if node.args.kwarg:
        names.append(f"**{node.args.kwarg.arg}")

    local_nodes = _iter_local_body_nodes(node)

    docstring_text = get_docstring(node)
    comments = extract_inline_comments(code, function_name) or []
    strings = extract_string_literals(code, function_name) or []
    ctrl = count_control_structures(code, function_name) or ControlStructureCounts()
    style = analyze_code_style(code, function_name) or CodeStyleMetrics()

    return FunctionAnalysis(
        function_name=function_name,
        has_return=any(isinstance(n, ast.Return) for n in local_nodes),
        has_print=any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "print"
            for n in local_nodes
        ),
        has_raise=any(isinstance(n, ast.Raise) for n in local_nodes),
        has_assert=any(isinstance(n, ast.Assert) for n in local_nodes),
        return_type_annotation=ast.unparse(node.returns) if node.returns else None,
        parameter_names=names,
        has_docstring=bool(docstring_text),
        docstring_text=docstring_text,
        inline_comments=comments,
        string_literals=strings,
        control_structures=ctrl,
        code_style=style,
    )
