"""Evaluate a small, data-only expression language without Python execution."""

from __future__ import annotations

import ast
import operator
from typing import Any


class SafeExpressionError(ValueError):
    """Raised when an expression is outside the supported safe subset."""


_MAX_NODES = 128
_MAX_COLLECTION_ITEMS = 64
_MAX_STRING_LENGTH = 4096
_MAX_INTEGER_BITS = 4096

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.invert,
}
_COMPARISON_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: operator.contains,
    ast.NotIn: lambda left, right: not operator.contains(left, right),
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}


def _validate_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_STRING_LENGTH:
        raise SafeExpressionError("string literal exceeds the safe expression limit")
    if isinstance(value, int) and value.bit_length() > _MAX_INTEGER_BITS:
        raise SafeExpressionError("integer exceeds the safe expression limit")
    if isinstance(value, (list, tuple, set, dict)) and len(value) > _MAX_COLLECTION_ITEMS:
        raise SafeExpressionError("collection exceeds the safe expression limit")
    return value


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (str, int, float, bool, type(None))):
            raise SafeExpressionError("literal type is not allowed")
        return _validate_value(node.value)

    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        if len(node.elts) > _MAX_COLLECTION_ITEMS:
            raise SafeExpressionError("collection exceeds the safe expression limit")
        values = [_evaluate(item) for item in node.elts]
        constructor = {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)]
        result = constructor(values)
        return _validate_value(result)

    if isinstance(node, ast.Dict):
        if len(node.keys) > _MAX_COLLECTION_ITEMS:
            raise SafeExpressionError("dictionary exceeds the safe expression limit")
        if any(key is None for key in node.keys):
            raise SafeExpressionError("dictionary unpacking is not allowed")
        try:
            result = {_evaluate(key): _evaluate(value) for key, value in zip(node.keys, node.values)}
        except TypeError as error:
            raise SafeExpressionError("dictionary keys must be hashable") from error
        return _validate_value(result)

    if isinstance(node, ast.UnaryOp):
        function = next((fn for kind, fn in _UNARY_OPERATORS.items() if isinstance(node.op, kind)), None)
        if function is None:
            raise SafeExpressionError("unary operator is not allowed")
        return _validate_value(function(_evaluate(node.operand)))

    if isinstance(node, ast.BinOp):
        function = next((fn for kind, fn in _BINARY_OPERATORS.items() if isinstance(node.op, kind)), None)
        if function is None:
            raise SafeExpressionError("binary operator is not allowed")
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        try:
            return _validate_value(function(left, right))
        except (ArithmeticError, TypeError) as error:
            raise SafeExpressionError(f"expression evaluation failed: {type(error).__name__}") from error

    if isinstance(node, ast.BoolOp):
        if not node.values:
            raise SafeExpressionError("boolean expression is empty")
        if isinstance(node.op, ast.And):
            result = True
            for value in node.values:
                result = _evaluate(value)
                if not result:
                    break
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for value in node.values:
                result = _evaluate(value)
                if result:
                    break
            return result
        raise SafeExpressionError("boolean operator is not allowed")

    if isinstance(node, ast.Compare):
        left = _evaluate(node.left)
        for operation, comparator in zip(node.ops, node.comparators):
            function = next((fn for kind, fn in _COMPARISON_OPERATORS.items() if isinstance(operation, kind)), None)
            if function is None:
                raise SafeExpressionError("comparison operator is not allowed")
            right = _evaluate(comparator)
            try:
                if not function(left, right):
                    return False
            except (TypeError, ValueError) as error:
                raise SafeExpressionError(f"comparison failed: {type(error).__name__}") from error
            left = right
        return True

    raise SafeExpressionError(f"expression node `{type(node).__name__}` is not allowed")


def evaluate_safe_expression(expression: str) -> Any:
    """Evaluate a bounded expression containing no names, calls, or access."""
    if not isinstance(expression, str) or not expression.strip():
        raise SafeExpressionError("expression must be a non-empty string")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise SafeExpressionError("expression is not valid syntax") from error
    if sum(1 for _ in ast.walk(tree)) > _MAX_NODES:
        raise SafeExpressionError("expression contains too many nodes")
    return _evaluate(tree.body)


def safe_expression(expression: str) -> str:
    """Tool wrapper for bounded arithmetic and data-only expressions."""
    try:
        value = evaluate_safe_expression(expression)
    except SafeExpressionError as error:
        return f"安全表达式错误：{error}"
    return repr(value)


safe_expression.tool_info = {
    "tool_name": "safe_expression",
    "tool_title": "安全表达式计算",
    "tool_description": "计算不访问文件、网络、进程或 Python 运行时的受限数据表达式",
    "tool_params": [
        {
            "name": "expression",
            "description": "受限表达式，例如 45 * 9827 或 [1, 2, 3]",
            "type": "string",
            "required": True,
        },
    ],
}


__all__ = ["SafeExpressionError", "evaluate_safe_expression", "safe_expression"]
