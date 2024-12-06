import pytest
import numpy as np
from pySDC.implementations.problem_classes.DuffingEquation import duffingequation, duffing_first_model
from qmat.lagrange import LagrangeApproximation

import pytest
import numpy as np
from pySDC.implementations.problem_classes.DuffingEquation import duffingequation
from pySDC.core.errors import ProblemError
from pySDC.implementations.datatype_classes.particles import particles, acceleration


@pytest.mark.parametrize(
    "u_pos, t, expected_acceleration",
    [
        (np.array([1.0]), 0, -1.1),  # default parameters
        (np.array([0.0]), 0, 0.0),  # zero position
        (np.array([-1.0]), 0, 1.1),  # negative position
    ],
    ids=[
        "default_parameters",
        "zero_position",
        "negative_position",
    ],
)
def test_eval_f(u_pos, t, expected_acceleration):

    # Arrange
    problem = duffingequation()
    # Act
    u = problem.u_init()
    u.pos = u_pos
    f = problem.eval_f(u, t)

    # Assert
    assert np.isclose(f[0], expected_acceleration)


@pytest.mark.base
def test_exact_solution():
    """
    Test for the exact solution of the reduced order models
    """
    P_duffing = duffingequation(epsilon=0.001)
    P_reduced_model = duffing_first_model(epsilon=0.001)

    num_nodes = 5
    nodes = np.sort(np.random.rand(num_nodes) / 4)
    scipy_solution = []
    asymp_solution = []
    for ii in nodes:
        u_scipy = P_duffing.scipy_solution(ii)
        u_asymp = P_reduced_model.u_asymptotic_expansion(ii)
        scipy_solution = np.append(scipy_solution, u_scipy.pos)
        asymp_solution = np.append(asymp_solution, u_asymp.pos)
    assert np.allclose(scipy_solution, asymp_solution, atol=1e-6), 'Scipy solution and asymptotic solution are failed!'


@pytest.mark.parametrize(
    "u_pos, t, expected_acceleration",
    [
        (np.array([2.0]), 0, -10.0),  # Default parameters
        (np.array([0.0]), 0, -8.0),  # Zero position
        (np.array([-1.0]), 0, -7.0),  # Negative position
    ],
    ids=["default_parameters", "zero_position", "negative_position"],
)
def test_first_order_reduced_eval_f(u_pos, t, expected_acceleration):
    # Arrange
    problem = duffing_first_model()
    u = problem.u_init()
    u.pos = u_pos

    # Act
    f = problem.eval_f(u, t)

    # Assert
    assert np.isclose(
        f[0], expected_acceleration, atol=1e-5
    ), f"Expected acceleration: {expected_acceleration}, got: {f[0]}"


if __name__ == '__main__':
    # test_DuffingEquation()
    # test_reduced_models()
    test_exact_solution()
