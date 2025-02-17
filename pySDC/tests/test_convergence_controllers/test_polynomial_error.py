import pytest


def get_controller(dt, num_nodes, quad_type, useMPI, useGPU, rel_error):
    """
    Get a controller prepared for polynomial test equation

    Args:
        dt (float): Step size
        num_nodes (int): Number of nodes
        quad_type (str): Type of quadrature
        useMPI (bool): Whether or not to use MPI

    Returns:
       (dict): Stats object generated during the run
       (pySDC.Controller.controller): Controller used in the run
    """
    from pySDC.implementations.problem_classes.polynomial_test_problem import polynomial_testequation
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.convergence_controller_classes.estimate_polynomial_error import (
        EstimatePolynomialError,
    )

    if useMPI:
        from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper_class
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        comm = None

    # initialize level parameters
    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = 1.0

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = quad_type
    sweeper_params['num_nodes'] = num_nodes
    sweeper_params['comm'] = comm

    problem_params = {
        'degree': 12,
        'useGPU': useGPU,
    }

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 0

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = polynomial_testequation
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = {EstimatePolynomialError: {'rel_error': rel_error}}

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    return controller


def single_test(**kwargs):
    """
    Run a single test where the solution is replaced by a polynomial and the nodes are changed.
    Because we know the polynomial going in, we can check if the interpolation based change was
    exact. If the solution is not a polynomial or a polynomial of higher degree then the number
    of nodes, the change in nodes does add some error, of course, but here it is on the order of
    machine precision.
    """
    import numpy as np

    args = {
        'num_nodes': 3,
        'quad_type': 'RADAU-RIGHT',
        'useMPI': False,
        'dt': 1.0,
        'useGPU': False,
        **kwargs,
    }

    # prepare variables
    controller = get_controller(**args)
    step = controller.MS[0]
    level = step.levels[0]
    prob = level.prob
    cont = controller.convergence_controllers[
        np.arange(len(controller.convergence_controllers))[
            [type(me).__name__ == 'EstimatePolynomialError' for me in controller.convergence_controllers]
        ][0]
    ]
    nodes = np.append([0], level.sweep.coll.nodes)

    # initialize variables
    step.status.slot = 0
    step.status.iter = 1
    level.status.time = 0.0
    level.status.residual = 0.0
    level.u[0] = prob.u_exact(t=0)
    level.sweep.predict()

    for i in range(len(level.u)):
        if level.u[i] is not None:
            level.u[i][:] = prob.u_exact(nodes[i] * level.dt)

    # perform the interpolation
    cont.reset_status_variables(controller)
    cont.post_iteration_processing(controller, step)
    error = level.status.error_embedded_estimate
    order = level.status.order_embedded_estimate

    return error, order


def multiple_runs(dts, **kwargs):
    """
    Make multiple runs of a specific problem and record vital error information

    Args:
        dts (list): The step sizes to run with
        num_nodes (int): Number of nodes
        quad_type (str): Type of nodes

    Returns:
        dict: Errors for multiple runs
        int: Order of the collocation problem
    """
    res = {}

    for dt in dts:
        res[dt] = {}
        res[dt]['e'], res[dt]['order'] = single_test(dt=dt, **kwargs)

    return res


def check_order(dts, **kwargs):
    """
    Check the order by calling `multiple_runs` and then `plot_and_compute_order`.

    Args:
        dts (list): The step sizes to run with
        num_nodes (int): Number of nodes
        quad_type (str): Type of nodes
    """
    import numpy as np

    res = multiple_runs(dts, **kwargs)
    dts = np.array(list(res.keys()))
    # keys = list(res[dts[0]].keys())

    expected_order = {
        'e': res[dts[0]]['order'],
    }

    for key in ['e']:
        errors = np.array([res[dt][key] for dt in dts])

        mask = np.logical_and(errors < 1e-0, errors > 1e-10)
        order = np.log(errors[mask][1:] / errors[mask][:-1]) / np.log(dts[mask][1:] / dts[mask][:-1])

        assert np.isclose(
            np.mean(order), expected_order[key], atol=0.5
        ), f'Expected order {expected_order[key]} for {key}, but got {np.mean(order):.2e}!'


@pytest.mark.base
@pytest.mark.parametrize('num_nodes', [2, 3, 4, 5])
@pytest.mark.parametrize('quad_type', ['RADAU-RIGHT', 'GAUSS'])
@pytest.mark.parametrize('rel_error', [True, False])
def test_interpolation_error(num_nodes, quad_type, rel_error):
    import numpy as np

    kwargs = {
        'num_nodes': num_nodes,
        'quad_type': quad_type,
        'useMPI': False,
        'rel_error': rel_error,
    }
    steps = np.logspace(-1, -4, 20)
    check_order(steps, **kwargs)


@pytest.mark.cupy
@pytest.mark.parametrize('num_nodes', [2, 3, 4, 5])
@pytest.mark.parametrize('quad_type', ['RADAU-RIGHT', 'GAUSS'])
def test_interpolation_error_GPU(num_nodes, quad_type):
    import numpy as np

    kwargs = {
        'num_nodes': num_nodes,
        'quad_type': quad_type,
        'useMPI': False,
        'useGPU': True,
        'rel_error': False,
    }
    steps = np.logspace(-1, -4, 20)
    check_order(steps, **kwargs)


@pytest.mark.mpi4py
@pytest.mark.parametrize('num_nodes', [2, 5])
@pytest.mark.parametrize('quad_type', ['RADAU-RIGHT', 'GAUSS'])
def test_interpolation_error_MPI(num_nodes, quad_type):
    import subprocess
    import os

    # Set python path once
    my_env = os.environ.copy()
    my_env['PYTHONPATH'] = '../../..:.'
    my_env['COVERAGE_PROCESS_START'] = 'pyproject.toml'

    cmd = f"mpirun -np {num_nodes} python {__file__} {num_nodes} {quad_type}".split()

    p = subprocess.Popen(cmd, env=my_env, cwd=".")

    p.wait()
    assert p.returncode == 0, 'ERROR: did not get return code 0, got %s with %2i processes' % (
        p.returncode,
        num_nodes,
    )


@pytest.mark.firedrake
def test_polynomial_error_firedrake(dt=1.0, num_nodes=3, useMPI=False):
    from pySDC.implementations.problem_classes.HeatFiredrake import Heat1DForcedFiredrake
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.convergence_controller_classes.estimate_polynomial_error import (
        EstimatePolynomialErrorFiredrake,
        LagrangeApproximation,
    )
    import numpy as np

    if useMPI:
        from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper_class
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        comm = None

    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = 1.0

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = num_nodes
    sweeper_params['comm'] = comm

    problem_params = {'n': 1}

    step_params = {}
    step_params['maxiter'] = 0

    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['mssdc_jac'] = False

    description = {}
    description['problem_class'] = Heat1DForcedFiredrake
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = {EstimatePolynomialErrorFiredrake: {}}

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    L = controller.MS[0].levels[0]

    cont = controller.convergence_controllers[
        np.arange(len(controller.convergence_controllers))[
            [type(me).__name__ == 'EstimatePolynomialErrorFiredrake' for me in controller.convergence_controllers]
        ][0]
    ]

    nodes = np.append(np.append(0, L.sweep.coll.nodes), 1.0)
    estimate_on_node = cont.params.estimate_on_node
    interpolator = LagrangeApproximation(points=[nodes[i] for i in range(num_nodes + 1) if i != estimate_on_node])
    cont.interpolation_matrix = np.array(interpolator.getInterpolationMatrix([nodes[estimate_on_node]]))

    for i in range(num_nodes + 1):
        L.u[i] = L.prob.u_init
        L.u[i].functionspace.assign(nodes[i])

    u_inter = cont.get_interpolated_solution(L)
    error = abs(u_inter - L.u[estimate_on_node])
    assert np.isclose(error, 0)


if __name__ == "__main__":
    import sys
    import numpy as np

    steps = np.logspace(-1, -4, 20)

    if len(sys.argv) > 1:
        kwargs = {
            'num_nodes': int(sys.argv[1]),
            'quad_type': sys.argv[2],
            'rel_error': False,
        }
        check_order(steps, useMPI=True, **kwargs)
    else:
        check_order(steps, useMPI=False, num_nodes=3, quad_type='RADAU-RIGHT', rel_error=False)
