import ast
import io
import tokenize


def merge_code_components(*components: str) -> str:
    return "\n".join(component.rstrip() for component in components) + "\n"


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
        if not body:
            return body

        first_stmt = body[0]
        if not isinstance(first_stmt, ast.Expr):
            return body
        if not isinstance(first_stmt.value, ast.Constant):
            return body
        if not isinstance(first_stmt.value.value, str):
            return body
        return body[1:]


def remove_docstrings_and_comments(source: str) -> str:
    tree = ast.parse(source)
    stripped_tree = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(stripped_tree)
    return ast.unparse(stripped_tree).rstrip() + "\n"


def find_first_function_name(source: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
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


def get_docstring(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    return (ast.get_docstring(node) or "").strip()


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
