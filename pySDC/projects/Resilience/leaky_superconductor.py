# script to run a quench problem
from pySDC.implementations.problem_classes.LeakySuperconductor import LeakySuperconductor, LeakySuperconductorIMEX
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.core.Hooks import hooks
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.Resilience.hook import hook_collection, LogData
import numpy as np

import matplotlib.pyplot as plt
from pySDC.core.Errors import ConvergenceError


class live_plot(hooks):  # pragma no cover
    """
    This hook plots the solution and the non-linear part of the right hand side after every step. Keep in mind that using adaptivity will result in restarts, which is not marked in these plots. Prepare to see the temperature profile jumping back again after a restart.
    """

    def _plot_state(self, step, level_number):  # pragma no cover
        """
        Plot the solution at all collocation nodes and the non-linear part of the right hand side

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): Number of current level

        Returns:
            None
        """
        L = step.levels[level_number]
        for ax in self.axs:
            ax.cla()
        [self.axs[0].plot(L.prob.xv, L.u[i], legend=f"node {i}") for i in range(len(L.u))]
        self.axs[0].axhline(L.prob.params.u_thresh, color='black')
        self.axs[1].plot(L.prob.xv, L.prob.eval_f_non_linear(L.u[-1], L.time))
        self.axs[0].set_ylim(0, 0.025)
        self.fig.suptitle(f"t={L.time:.2e}, k={step.status.iter}")
        plt.pause(1e-9)

    def pre_run(self, step, level_number):  # pragma no cover
        """
        Setup a figure to plot into

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): Number of current level

        Returns:
            None
        """
        self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 4))

    def post_step(self, step, level_number):  # pragma no cover
        """
        Call the plotting function after the step

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): Number of current level

        Returns:
            None
        """
        self._plot_state(step, level_number)


def run_leaky_superconductor(
    custom_description=None,
    num_procs=1,
    Tend=6e2,
    hook_class=LogData,
    fault_stuff=None,
    custom_controller_params=None,
    custom_problem_params=None,
    imex=False,
    u0=None,
    t0=None,
    **kwargs,
):
    """
    Run a toy problem of a superconducting magnet with a temperature leak with default parameters.

    Args:
        custom_description (dict): Overwrite presets
        num_procs (int): Number of steps for MSSDC
        Tend (float): Time to integrate to
        hook_class (pySDC.Hook): A hook to store data
        fault_stuff (dict): A dictionary with information on how to add faults
        custom_controller_params (dict): Overwrite presets
        custom_problem_params (dict): Overwrite presets
        imex (bool): Solve the problem IMEX or fully implicit
        u0 (dtype_u): Initial value
        t0 (float): Starting time

    Returns:
        dict: The stats object
        controller: The controller
        Tend: The time that was supposed to be integrated to
    """

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 10.0

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'
    sweeper_params['QE'] = 'PIC'

    problem_params = {}

    if custom_problem_params is not None:
        problem_params = {**problem_params, **custom_problem_params}

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 5

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_collection + (hook_class if type(hook_class) == list else [hook_class])
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = LeakySuperconductorIMEX if imex else LeakySuperconductor
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order if imex else generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    if custom_description is not None:
        for k in custom_description.keys():
            if k == 'sweeper_class':
                description[k] = custom_description[k]
                continue
            description[k] = {**description.get(k, {}), **custom_description.get(k, {})}

    # set time parameters
    t0 = 0.0 if t0 is None else t0

    # instantiate controller
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # insert faults
    if fault_stuff is not None:
        from pySDC.projects.Resilience.fault_injection import prepare_controller_for_faults

        rnd_args = {'iteration': 5, 'min_node': 1}
        args = {'time': 21.0, 'target': 0}
        prepare_controller_for_faults(controller, fault_stuff, rnd_args, args)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0) if u0 is None else u0

    # call main function to get things done...
    try:
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    except ConvergenceError:
        stats = controller.return_stats()
    return stats, controller, Tend


def plot_solution(stats, controller):  # pragma no cover
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    u_ax = ax
    dt_ax = u_ax.twinx()

    u = get_sorted(stats, type='u', recomputed=False)
    u_ax.plot([me[0] for me in u], [max(me[1]) for me in u], label=r'$T$')

    dt = get_sorted(stats, type='dt', recomputed=False)
    dt_ax.plot([me[0] for me in dt], [me[1] for me in dt], color='black', ls='--')
    u_ax.plot([None], [None], color='black', ls='--', label=r'$\Delta t$')

    P = controller.MS[0].levels[0].prob
    u_ax.axhline(P.params.u_thresh, color='grey', ls='-.', label=r'$T_\mathrm{thresh}$')
    u_ax.axhline(P.params.u_max, color='grey', ls=':', label=r'$T_\mathrm{max}$')

    u_ax.legend()
    u_ax.set_xlabel(r'$t$')
    u_ax.set_ylabel(r'$T$')
    dt_ax.set_ylabel(r'$\Delta t$')


def compare_imex_full(plotting=False):
    """
    Compare the results of IMEX and fully implicit runs. For IMEX we need to limit the step size in order to achieve convergence, but for fully implicit, adaptivity can handle itself better.

    Args:
        plotting (bool): Plot the solution or not
    """
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

    res = {}

    custom_description = {}
    custom_description['problem_params'] = {
        'newton_tol': 1e-10,
        'newton_iter': 20,
        'nvars': 2**9,
    }
    custom_controller_params = {'logger_level': 30}
    for imex in [False, True]:
        custom_description['convergence_controllers'] = {Adaptivity: {'e_tol': 1e-6, 'dt_max': 1e2}}
        stats, controller, _ = run_leaky_superconductor(
            custom_description=custom_description,
            custom_controller_params=custom_controller_params,
            imex=imex,
            Tend=5e2,
        )

        res[imex] = get_sorted(stats, type='u')[-1][1]
        if plotting:  # pragma no cover
            plot_solution(stats, controller)

    diff = abs(res[True] - res[False])
    thresh = 3e-3
    assert (
        diff < thresh
    ), f"Difference between IMEX and fully-implicit too large! Got {diff:.2e}, allowed is only {thresh:.2e}!"
    prob = controller.MS[0].levels[0].prob
    assert (
        max(res[True]) > prob.params.u_max
    ), f"Expected runaway to happen, but maximum temperature is {max(res[True]):.2e} < u_max={prob.params.u_max:.2e}!"


def compare_dirk():
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity, AdaptivityRK
    from pySDC.implementations.sweeper_classes.Runge_Kutta import DIRK34
    from pySDC.projects.Resilience.hook import hook_collection, LogNewtonIter, LogData, LogSpaceIter
    from pySDC.helpers.plot_helper import figsize_by_journal, setup_mpl
    from pySDC.implementations.hooks.log_work import LogWork

    setup_mpl(reset=True)
    fig, axs = plt.subplots(2, 2, figsize=figsize_by_journal('Springer_Numerical_Algorithms', 1, 0.82), sharex=True)

    problem_params = {}
    problem_params['direct_solver'] = False

    description = {}
    description['step_params'] = {'maxiter': 4}
    description['convergence_controllers'] = {Adaptivity: {'e_tol': 1e-5, 'dt_max': 50.0}}

    description_RK = {}
    description_RK['convergence_controllers'] = {AdaptivityRK: {'e_tol': 1e-5, 'dt_max': 5e1, 'update_order': 4}}
    description_RK['sweeper_class'] = DIRK34
    description_RK['step_params'] = {'maxiter': 1}

    description_res = {}
    description_res['step_params'] = {'maxiter': 99}
    description_res['level_params'] = {'restol': 1e-10}

    controller_params = {}
    controller_params['hook_class'] = hook_collection + [LogNewtonIter, LogData, LogSpaceIter, LogWork]
    controller_params['logger_level'] = 30
    Tend = 500

    stats_fixed, _, _ = run_leaky_superconductor(
        custom_description=description_res,
        imex=False,
        Tend=Tend,
        custom_controller_params=controller_params,
        custom_problem_params=problem_params,
    )
    stats, _, _ = run_leaky_superconductor(
        custom_description=description,
        imex=False,
        Tend=Tend,
        custom_controller_params=controller_params,
        custom_problem_params=problem_params,
    )
    stats_RK, _, _ = run_leaky_superconductor(
        custom_description=description_RK,
        imex=False,
        Tend=Tend,
        custom_controller_params=controller_params,
        custom_problem_params=problem_params,
    )

    labels = {
        0: 'SDC+adaptivity',
        1: 'DIRK4(3)',
        2: 'SDC',
    }
    ls = {
        0: '-',
        1: '--',
        2: ':',
    }
    S = [stats, stats_RK, stats_fixed]
    for i in range(len(S)):
        newton_iter = get_sorted(S[i], type='work_newton')
        axs[0, 0].plot(
            [me[0] for me in newton_iter], np.cumsum([me[1] for me in newton_iter]), label=labels[i], ls=ls[i]
        )

        space_iter = get_sorted(S[i], type='work_linear')
        axs[1, 1].plot([me[0] for me in space_iter], np.cumsum([me[1] for me in space_iter]), ls=ls[i])

        dt = get_sorted(S[i], type='dt')
        axs[0, 1].plot([me[0] for me in dt], [me[1] for me in dt], ls=ls[i])

        # u = get_sorted(S[i], type='u', recomputed=False)
        # axs[1,1].plot([me[0] for me in u], [max(me[1]) for me in u], ls=ls[i])

        k = get_sorted(S[i], type='sweeps', recomputed=None)
        axs[1, 0].plot([me[0] for me in k], np.cumsum([me[1] for me in k]), ls=ls[i])
    axs[0, 0].set_yscale('log')
    axs[1, 1].set_yscale('log')
    axs[0, 0].legend(frameon=False)
    axs[0, 0].set_ylabel('cumulative Newton iterations')
    axs[1, 0].set_ylabel('cumulative SDC iterations')
    axs[1, 0].set_xlabel('$t$')
    axs[0, 1].set_ylabel(r'$\Delta t$')
    axs[1, 1].set_ylabel('cumulative GMRES iterations')
    fig.tight_layout()
    fig.savefig('data/quench_newton.pdf')


if __name__ == '__main__':
    compare_dirk()
    # compare_imex_full(plotting=True)
    plt.show()
