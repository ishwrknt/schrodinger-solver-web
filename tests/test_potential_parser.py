import numpy as np
import pytest

from schrodinger_solver_web.solver.potential_parser import PotentialExpressionError, evaluate_potential_expression


def test_allowed_expression_evaluates_on_arrays():
    x = np.array([-1.0, 0.0, 1.0])
    values = evaluate_potential_expression("0.5 * x**2 + sin(x)", x)
    assert values.shape == x.shape
    assert np.isfinite(values).all()


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hacked')",
        "open('file.txt')",
        "lambda y: y",
        "x if x > 0 else 0",
    ],
)
def test_unsafe_expression_is_rejected(expression):
    with pytest.raises(PotentialExpressionError):
        evaluate_potential_expression(expression, np.array([0.0]))


def test_unsupported_function_is_rejected():
    with pytest.raises(PotentialExpressionError, match="Unsupported function"):
        evaluate_potential_expression("arctan(x)", np.array([0.0]))

