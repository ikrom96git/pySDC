import matplotlib.pyplot as plt
import numpy as np

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
# from pySDC.playgrounds.Boris.penningtrap_HookClass import particles_output

from pySDC.tutorial.step_4.PenningTrap_3D_coarse import penningtrap_coarse
from pySDC.implementations.transfer_classes.TransferParticles_NoCoarse import particles_to_particles
def main():
    """
    Particle cloud in a penning trap, incl. live visualization

    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-16
    level_params['dt'] = 0.1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'GAUSS'
    sweeper_params['num_nodes'] = [5, 3]

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['omega_E'] = 1
    problem_params['omega_B'] = 0.0
    problem_params['u0'] = np.array([[0, 0, 1], [0, 0, 0], [1], [1]], dtype=object)
    problem_params['nparts'] = 1
    problem_params['sig'] = 0.1


    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 10

    # initialize controller parameters
    controller_params = dict()
    # controller_params['hook_class'] = particles_output  # specialized hook class for more statistics and output
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = penningtrap
    description['problem_params'] = problem_params
    description['sweeper_class'] = boris_2nd_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['space_transfer_class'] = particles_to_particles # this is only needed for more than 2 levels
    description['step_params'] = step_params

    # instantiate the controller (no controller parameters used here)
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    sortedlist_stats = get_sorted(stats, type='position', sortby='time')

    energy = [entry[1] for entry in sortedlist_stats]
    position=np.array(energy)
    positoin=position.reshape(np.shape(position)[0],3)
    plt.figure()
    plt.plot(position[:,2], 'b--')

    plt.xlabel('Time')
    plt.ylabel('Position')

    # plt.savefig('penningtrap_energy.png', transparent=True, bbox_inches='tight')


if __name__ == "__main__":
    main()
