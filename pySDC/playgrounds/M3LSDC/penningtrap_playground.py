import matplotlib.pyplot as plt
import numpy as np

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
# from pySDC.playgrounds.Boris.penningtrap_HookClass import particles_output
from pySDC.playgrounds.M3LSDC.NonUniformElectricField import non_uniformElectricField
from pySDC.playgrounds.M3LSDC.NonUniform_zeroth_order import non_uniform_zeroth_order
from pySDC.implementations.transfer_classes.TransferParticles_NoCoarse import particles_to_particles, mesh_to_mesh
from pySDC.playgrounds.M3LSDC.NonUniform_first_order import non_uniform_first_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh_2_particle import mesh_to_particles
from pySDC.playgrounds.M3LSDC.NonUniform_electric_field import non_uniformElectric

def main():
    """
    Particle cloud in a penning trap, incl. live visualization

    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-08
    level_params['dt'] = 0.00015625

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = [5, 3]

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['omega_E'] = 4.9
    problem_params['omega_B'] = 0.1
    problem_params['u0'] = np.array([1, 1, 1, 1, 1, 1])
    problem_params['nparts'] = 1
    problem_params['sig'] = 0.1
    # problem_params['Tend'] = 16.0

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    transfer_params = dict()
    transfer_params['finter'] = False




    # initialize controller parameters
    controller_params = dict()
    # controller_params['hook_class'] = particles_output  # specialized hook class for more statistics and output
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = [non_uniformElectric, non_uniform_first_order]
    description['problem_params'] = problem_params
    description['sweeper_class'] = [generic_implicit, generic_implicit]
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['space_transfer_class'] = mesh_to_mesh # this is only needed for more than 2 levels
    description['step_params'] = step_params
    description['transfer_params'] = transfer_params

    # instantiate the controller (no controller parameters used here)
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 128 * 0.015625

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    sortedlist_stats = get_sorted(stats, type='residual_post_sweep', sortby='level')
    sortedlist_array=np.array(sortedlist_stats)
    fine_level_residual, coarse_level_residual = np.split(sortedlist_array, 2, axis=1)
    re  
    print(fine_level_residual)
    breakpoint()

def plot_residual(x, y, labels):
    mark=['s', 'o', '.', '*']
    for ii in range(len(labels)):
        plt.semilogy(x, y[:, ii], label=labels[ii], marker=mark[ii])
    
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
