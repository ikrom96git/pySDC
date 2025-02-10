import numpy as np
from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
# from pySDC.projects.Second_orderSDC.penningtrap_HookClass import particles_output # for convergence of MLSDC
# from pySDC.playgrounds.Boris.penningtrap_HookClass import particles_output
from pySDC.tutorial.step_3.HookClass_Particles import particle_hook
from pySDC.projects.Second_orderSDC.penningtrap_params import penningtrap_params_mlsdc, penningtrap_params
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.implementations.transfer_classes.TransferParticles_NoCoarse import particles_to_particles
import matplotlib.pyplot as plt
from pySDC.helpers.plot_helper import figsize_by_journal

def Error_mlsdc():
    transfer_params = dict()
    transfer_params['finter'] = False
    step_params = dict()
    step_params['maxiter'] = 10
    controller_params_mlsdc, description_mlsdc = penningtrap_params_mlsdc()
    controller_params_mlsdc['hook_class'] = particles_output
    description_mlsdc['space_transfer_class'] = particles_to_particles
    description_mlsdc['base_transfer_params'] = transfer_params
    description_mlsdc['step_params'] = step_params

    u_val, uex_val = dict(), dict()
    values, error = ['position', 'velocity'], dict()
    controller = controller_nonMPI(
        num_procs=1, controller_params=controller_params_mlsdc, description=description_mlsdc
    )
    t0, Tend = 0.0, 2
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()
    uend, stats_mlsdc = controller.run(u0=uinit, t0=t0, Tend=Tend)
    for nn in values:
        u_val[nn] = get_sorted(stats_mlsdc, type=nn, sortby="time")
        uex_val[nn] = get_sorted(stats_mlsdc, type=f'{nn}_exact', sortby='time')
        error[nn] = relative_error(uex_val[nn], u_val[nn])
        error[nn] = list(error[nn].T[0])


def residual_mlsdc():
    transfer_params = dict()
    transfer_params['finter'] = False
    step_params = dict()
    step_params['maxiter'] = 10
    controller_params_mlsdc, description_mlsdc = penningtrap_params_mlsdc()
    controller_params_mlsdc['hook_class'] = particle_hook
    description_mlsdc['space_transfer_class'] = particles_to_particles
    description_mlsdc['base_transfer_params'] = transfer_params
    description_mlsdc['step_params'] = step_params

    u_val, uex_val = dict(), dict()
    values, error = ['position', 'velocity'], dict()
    controller = controller_nonMPI(
        num_procs=1, controller_params=controller_params_mlsdc, description=description_mlsdc
    )
    t0, Tend = 0.0, 2
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()
    uend, stats_mlsdc = controller.run(u0=uinit, t0=t0, Tend=Tend)
    sortedlist_stats_mlsdc = get_sorted(stats_mlsdc, type='residual_post_sweep', sortby='level')
    sortedlist_stats_mlsdc_array = np.asarray(sortedlist_stats_mlsdc)
    fine_level_mlsdc, coarse_level_mlsdc = np.split(sortedlist_stats_mlsdc_array, 2)
    
    return fine_level_mlsdc[:,1]


def residual_sdc():
    step_params = dict()
    step_params['maxiter'] = 10
    controller_params_sdc, description_sdc=penningtrap_params()
    controller_params_sdc['hook_class'] = particle_hook

    description_sdc['step_params'] = step_params

    # instantiate the controller (no controller parameters used here)
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params_sdc, description=description_sdc)

    # set time parameters
    t0 = 0.0
    Tend = 128 * 0.015625

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()
    uend, stats_sdc = controller.run(u0=uinit, t0=t0, Tend=Tend)
    sortedlist_stats_sdc = get_sorted(stats_sdc, type='residual_post_sweep', sortby='iter')
    sortedlist_stats_sdc_array = np.asarray(sortedlist_stats_sdc)
    Residual = np.copy(sortedlist_stats_sdc_array)
    iteration = sortedlist_stats_sdc_array[:, 0]
    return Residual, iteration


def relative_error(uex_data, u_data):
    u_ex = np.array([entry[1] for entry in uex_data])
    u = np.array([entry[1] for entry in u_data])
    return np.linalg.norm(np.abs((u_ex - u)), np.inf, 0) / np.linalg.norm(u_ex, np.inf, 0)

def plot_residual_sdc_vs_mlsdc():
    figsize_by_journal("TUHH_thesis", 10, 6)
    Residual, iteration=residual_sdc()
    fine_residual=residual_mlsdc()
    
    Residual[:, 0]=fine_residual
    breakpoint()
    labels=['MLSDC', 'SDC']
    for ii in range(len(labels)):
        plt.semilogy(iteration, Residual[:, ii], label=labels[ii])
    plt.xlabel("Iteration")
    plt.ylabel('$\|R\|_{\infty}$')
    plt.legend()
    plt.tight_layout()
    plt.show()

    


if __name__ == '__main__':
    # Error_mlsdc()
    plot_residual_sdc_vs_mlsdc()
