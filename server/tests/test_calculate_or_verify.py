"""Unit tests for app/agent/tools/calculate_or_verify.py -- direct calls to
the async tool function, no agent loop or LLM involved.
"""

import ast
import inspect

import pytest

from app.agent.tools import calculate_or_verify as calc_module
from app.agent.tools.calculate_or_verify import calculate_or_verify


async def test_valid_simple_expression():
    result = await calculate_or_verify("2 + 3", "simple addition")
    assert result == "Result: 5 (context: simple addition)"


async def test_valid_expression_respects_operator_precedence_and_parens():
    # Without correct precedence/parens support this would be 20, not 14.
    result = await calculate_or_verify("2 + 3 * 4", "precedence check")
    assert result == "Result: 14 (context: precedence check)"

    result = await calculate_or_verify("(2 + 3) * 4", "parens check")
    assert result == "Result: 20 (context: parens check)"


async def test_valid_expression_matches_the_real_overtime_formula():
    # The exact shape of expression the agent sends in practice (Section 5.5).
    result = await calculate_or_verify("(19800000 / 176) * 1.5 * 8", "overtime pay")
    assert result == "Result: 1350000.0 (context: overtime pay)"


async def test_invalid_expression_raises_value_error_with_clear_message():
    with pytest.raises(ValueError, match=r"Invalid expression"):
        await calculate_or_verify("2 +", "syntax error")


async def test_undefined_variable_raises_value_error():
    with pytest.raises(ValueError, match=r"Invalid expression"):
        await calculate_or_verify("undefined_variable + 1", "undefined name")


def test_never_uses_python_builtin_eval():
    """Static check backing the module's own "NOT Python eval()" docstring
    claim. Walks the actual AST (not a text/regex search, which would
    false-positive on the word "eval()" appearing in the module's own
    docstring, or on "aeval(" -- asteval's interpreter call, which is
    exactly what this module is supposed to use instead) looking for a Call
    node whose callee is literally the builtin name `eval`.
    """
    source = inspect.getsource(calc_module)
    tree = ast.parse(source)
    eval_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval"
    ]
    assert not eval_calls, "found a call to the builtin eval() in calculate_or_verify.py"
    assert "aeval(" in source, "expected the module to actually use asteval's Interpreter"
