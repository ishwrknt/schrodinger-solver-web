from __future__ import annotations

import ast
from collections.abc import Callable

import numpy as np


ALLOWED_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "log": np.log,
}
ALLOWED_NAMES = {"x", "pi", "e"}
ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


class PotentialExpressionError(ValueError):
    """Raised when a potential expression is invalid or unsafe."""


def parse_potential_expression(expression: str) -> ast.AST:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise PotentialExpressionError(f"Invalid analytic expression: {exc.msg}.") from exc
    _validate_node(tree)
    return tree


def evaluate_potential_expression(expression: str, x: np.ndarray) -> np.ndarray:
    tree = parse_potential_expression(expression)
    return _evaluate_node(tree.body, np.asarray(x, dtype=float))


def _validate_node(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate_node(node.body)
        return
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise PotentialExpressionError("Only numeric constants are allowed.")
        return
    if isinstance(node, ast.Name):
        if node.id not in ALLOWED_NAMES:
            raise PotentialExpressionError(f"Unsupported symbol '{node.id}'.")
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, ALLOWED_BINOPS):
            raise PotentialExpressionError("Unsupported operator in analytic expression.")
        _validate_node(node.left)
        _validate_node(node.right)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ALLOWED_UNARYOPS):
            raise PotentialExpressionError("Unsupported unary operator in analytic expression.")
        _validate_node(node.operand)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise PotentialExpressionError("Unsupported function in analytic expression.")
        if len(node.keywords) > 0:
            raise PotentialExpressionError("Keyword arguments are not supported.")
        for arg in node.args:
            _validate_node(arg)
        return
    raise PotentialExpressionError(f"Unsupported syntax '{type(node).__name__}'.")


def _evaluate_node(node: ast.AST, x: np.ndarray) -> np.ndarray:
    if isinstance(node, ast.Constant):
        return np.full_like(x, float(node.value), dtype=float)
    if isinstance(node, ast.Name):
        if node.id == "x":
            return x
        if node.id == "pi":
            return np.full_like(x, np.pi, dtype=float)
        if node.id == "e":
            return np.full_like(x, np.e, dtype=float)
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, x)
        right = _evaluate_node(node.right, x)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_node(node.operand, x)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = ALLOWED_FUNCTIONS[node.func.id]
        args = [_evaluate_node(arg, x) for arg in node.args]
        return fn(*args)
    raise PotentialExpressionError("Unable to evaluate analytic expression.")

