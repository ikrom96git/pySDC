import pytest
import numpy as np
from pySDC.implementations.problem_classes.DuffingEquation import (
    duffingequation,
    duffing_first_model,
    duffing_zeros_model,
)
from qmat.lagrange import LagrangeApproximation


@pytest.mark.base
def test_DuffingEquation():
    """
    Testing eval_f for the Duffing equation

    Returns:
        _type_: _description_
    """
    P = duffingequation()

    def derivative_eval_f(u):
        return -P.omega**2 - 3 * P.epsilon * P.b * u**2

    num_nodes = 5
    nodes = np.sort(np.random.rand(num_nodes))
    approx = LagrangeApproximation(nodes)
    D1 = approx.getDerivationMatrix()
    u = P.u_init()
    func_vals = []
    for ii in nodes:
        u.pos = ii
        func_vals = np.append(func_vals, P.eval_f(u, 0.0))

    derivative_f = derivative_eval_f(nodes)
    approx_derivative = D1 @ func_vals
    assert np.allclose(derivative_f, approx_derivative, atol=1e-6), 'eval_f function derivative is failed'


@pytest.mark.base
def test_reduced_models():
    """
    Test for first order reduced model RHS funciton
    """
    P = duffing_first_model()

    def derivative_eval_f(t):
        z = P.omega * t
        return (0.25 * (P.duffing_zeros.u0[0] ** 3) * P.b * P.omega) * (3 * np.sin(3 * z) + 3 * np.sin(z))

    num_nodes = 5
    nodes = np.sort(np.random.rand(num_nodes) / 8)
    approx = LagrangeApproximation(nodes)
    D1 = approx.getDerivationMatrix()
    u = P.u_init()
    func_vals = []
    for ii in nodes:
        func_vals = np.append(func_vals, P.eval_f(u, ii))

    derivative_f = derivative_eval_f(nodes)
    approx_derivative = D1 @ func_vals
    assert np.allclose(derivative_f, approx_derivative, atol=1e-6), 'Approximate derivative and derivative of RHS function do not reat tolerence!'


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


if __name__ == '__main__':
    # test_DuffingEquation()
    # test_reduced_models()
    test_exact_solution()
