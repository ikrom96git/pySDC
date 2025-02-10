import time
from pathlib import Path

import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.implementations.transfer_classes.TransferParticles_NoCoarse import particles_to_particles
from pySDC.tutorial.step_3.HookClass_Particles import particle_hook
from pySDC.tutorial.step_4.PenningTrap_3D_coarse import penningtrap_coarse
import matplotlib.pyplot as plt
from pySDC.projects.Second_orderSDC.plot_helper import set_fixed_plot_params

set_fixed_plot_params()
def main():
    """
    A simple test program to compare SDC with two flavors of MLSDC for particle dynamics
    """

    # run SDC, MLSDC and MLSDC plus f-interpolation and compare
    stats_sdc, time_sdc = run_penning_trap_simulation(mlsdc=False)
    stats_mlsdc, time_mlsdc = run_penning_trap_simulation(mlsdc=True)
    stats_mlsdc_finter, time_mlsdc_finter = run_penning_trap_simulation(mlsdc=True, finter=True)
    breakpoint()
    Path("data").mkdir(parents=True, exist_ok=True)
    f = open('data/step_4_D_out.txt', 'w')
    out = 'Timings for SDC, MLSDC and MLSDC+finter: %12.8f -- %12.8f -- %12.8f' % (
        time_sdc,
        time_mlsdc,
        time_mlsdc_finter,
    )
    f.write(out + '\n')
    print(out)

    # sort and convert stats to list, sorted by iteration numbers (only pre- and after-step are present here)
    energy_sdc = get_sorted(stats_sdc, type='etot', sortby='iter')
    energy_mlsdc = get_sorted(stats_mlsdc, type='etot', sortby='iter')
    energy_mlsdc_finter = get_sorted(stats_mlsdc_finter, type='etot', sortby='iter')

    # get base energy and show differences
    base_energy = energy_sdc[0][1]
    for item in energy_sdc:
        out = 'Total energy and relative deviation in iteration %2i: %12.10f -- %12.8e' % (
            item[0],
            item[1],
            abs(base_energy - item[1]) / base_energy,
        )
        f.write(out + '\n')
        print(out)
    for item in energy_mlsdc:
        out = 'Total energy and relative deviation in iteration %2i: %12.10f -- %12.8e' % (
            item[0],
            item[1],
            abs(base_energy - item[1]) / base_energy,
        )
        f.write(out + '\n')
        print(out)
    for item in energy_mlsdc_finter:
        out = 'Total energy and relative deviation in iteration %2i: %12.10f -- %12.8e' % (
            item[0],
            item[1],
            abs(base_energy - item[1]) / base_energy,
        )
        f.write(out + '\n')
        print(out)
    f.close()

    assert (
        abs(energy_sdc[-1][1] - energy_mlsdc[-1][1]) / base_energy < 6e-10
    ), 'ERROR: energy deviated too much between SDC and MLSDC, got %s' % (
        abs(energy_sdc[-1][1] - energy_mlsdc[-1][1]) / base_energy
    )
    assert (
        abs(energy_mlsdc[-1][1] - energy_mlsdc_finter[-1][1]) / base_energy < 8e-10
    ), 'ERROR: energy deviated too much after using finter, got %s' % (
        abs(energy_mlsdc[-1][1] - energy_mlsdc_finter[-1][1]) / base_energy
    )


def run_penning_trap_simulation(mlsdc, finter=False, iter=3):
    """
    A simple test program to run IMEX SDC for a single time step
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-16
    level_params['dt'] = 1.0 / 8

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'GAUSS'
    sweeper_params['num_nodes'] = 6

    sweeper_params_mlsdc = dict()
    sweeper_params_mlsdc['quad_type'] = 'GAUSS'
    sweeper_params_mlsdc['num_nodes'] = [6, iter]

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['omega_E'] = 4.9  # E-field frequency
    problem_params['omega_B'] = 25.0  # B-field frequency
    problem_params['u0'] = np.array([[10, 0, 0], [100, 0, 100], [1], [1]], dtype=object)  # initial center of positions
    problem_params['nparts'] = 1  # number of particles in the trap
    problem_params['sig'] = 0.1  # smoothing parameter for the forces

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 10

    # initialize controller parameters
    controller_params = dict()
    controller_params['hook_class'] = particle_hook  # specialized hook class for more statistics and output
    controller_params['logger_level'] = 30

    transfer_params = dict()
    transfer_params['finter'] = finter

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    if mlsdc:
        # MLSDC: provide list of two problem classes: one for the fine, one for the coarse level
        description['problem_class'] = [penningtrap, penningtrap]
        description['sweeper_params'] = sweeper_params_mlsdc
    else:
        # SDC: provide only one problem class
        description['problem_class'] = penningtrap
        description['sweeper_params'] = sweeper_params
    description['problem_params'] = problem_params
    description['sweeper_class'] = boris_2nd_order

    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = particles_to_particles
    description['base_transfer_params'] = transfer_params

    # instantiate the controller (no controller parameters used here)
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = level_params['dt']

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()

    # call and time main function to get things done...
    start_time = time.perf_counter()
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    end_time = time.perf_counter() - start_time

    return stats, end_time


def residual_sdc_mlsdc():
    stats_sdc, time_sdc = run_penning_trap_simulation(mlsdc=False)
    stats_mlsdc3, time_mlsdc = run_penning_trap_simulation(mlsdc=True, iter=3)
    stats_mlsdc4, *_=run_penning_trap_simulation(mlsdc=True, iter=4)
    stats_mlsdc5, *_=run_penning_trap_simulation(mlsdc=True, iter=5)

    sortedlist_stats_sdc = get_sorted(stats_sdc, type='residual_post_sweep', sortby='iter')
    sortedlist_stats_sdc_array = np.asarray(sortedlist_stats_sdc)
    sortedlist_stats_mlsdc3 = get_sorted(stats_mlsdc3, type='residual_post_sweep', sortby='level')
    sortedlist_stats_mlsdc4 = get_sorted(stats_mlsdc4, type='residual_post_sweep', sortby='level')
    sortedlist_stats_mlsdc5 = get_sorted(stats_mlsdc5, type='residual_post_sweep', sortby='level')

    sortedlist_stats_mlsdc_array3 = np.asarray(sortedlist_stats_mlsdc3)
    sortedlist_stats_mlsdc_array4 = np.asarray(sortedlist_stats_mlsdc4)
    sortedlist_stats_mlsdc_array5 = np.asarray(sortedlist_stats_mlsdc5)

    fine_level_mlsdc3, coarse_level_mlsdc = np.split(sortedlist_stats_mlsdc_array3, 2)
    fine_level_mlsdc4, coarse_level_mlsdc = np.split(sortedlist_stats_mlsdc_array4, 2)
    fine_level_mlsdc5, coarse_level_mlsdc = np.split(sortedlist_stats_mlsdc_array5, 2)
    Residual=np.zeros((np.shape(sortedlist_stats_sdc_array[:,1])[0], 4))

    Residual[:,0] = np.copy(sortedlist_stats_sdc_array)[:,1]
    Residual[:, 1] = fine_level_mlsdc3[:, 1]
    Residual[:, 2] = fine_level_mlsdc4[:, 1]
    Residual[:, 3] = fine_level_mlsdc5[:, 1]



    iteration = sortedlist_stats_sdc_array[:, 0]
    return Residual, iteration


def plot_residual(x, y, labels):
    # figsize_by_journal('TUHH_thesis', 10, 6)
    mark=['s', 'o', '.', '*']
    colors=['black', "red", "blue", "green"]
    for ii in range(len(labels)):
        plt.semilogy(x, y[:, ii], label=labels[ii], color=colors[ii], marker=mark[ii])

    plt.xlabel('Iteration')
    plt.ylabel('$\|R\|_{\infty}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/residual_SDC_vs_MLSDC.pdf")
    plt.show()



if __name__ == "__main__":
    # main()
    Residual, iteration = residual_sdc_mlsdc()
    labels = ['SDC', 'MLSDC $(6,3)$', 'MLSDC $(6,4)$', 'MLSDC $(6,5)$']
    plot_residual(iteration, Residual, labels=labels)
