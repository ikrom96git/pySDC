import matplotlib.pyplot as plt
import numpy as np

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.playgrounds.M3LSDC.NonUniform_zeroth_order import non_uniform_zeroth_order, non_uniform_zeroth_order_generic_implicit
from pySDC.implementations.transfer_classes.TransferParticles_NoCoarse import particles_to_particles, mesh_to_mesh
from pySDC.playgrounds.M3LSDC.NonUniform_first_order import non_uniform_first_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.M3LSDC.NonUniform_electric_field import non_uniformElectric, non_uniformElectric_generic_implicit
from pySDC.playgrounds.M3LSDC.penningtrap_playground import dt, EPSILON, plot_residual, first_order_model
from pySDC.playgrounds.M3LSDC.NonUniform_first_order import non_uniform_first_order

def MLSDC_generic_implicit(zeroth_order=False):
    """
    Particle cloud in a penning trap, incl. live visualization

    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = -1
    level_params['dt'] = dt
    # level_params['nsweeps']= 1
    

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'GAUSS'
    sweeper_params['num_nodes'] = [5, 3]

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['omega_E'] = 4.9
    problem_params['omega_B'] = EPSILON
    problem_params['u0'] = np.array([1,1,1,1,1,1])
    problem_params['nparts'] = 1
    problem_params['sig'] = 0.1
    # problem_params['Tend'] = 16.0

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    transfer_params = dict()
    transfer_params['finter'] = False





    # initialize controller parameters
    controller_params = dict()
    # controller_params['hook_class'] = particles_output  # specialized hook class for more statistics and output
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = [non_uniformElectric_generic_implicit, non_uniformElectric_generic_implicit]
    if zeroth_order:
        description['problem_class'] = [non_uniformElectric_generic_implicit, non_uniform_zeroth_order_generic_implicit]
            
    
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
    Tend = dt#128 * 0.015625

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_init()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    sortedlist_stats = get_sorted(stats, type='residual_post_sweep', sortby='level')
    sortedlist_array=np.array(sortedlist_stats)
    fine_level, coarse_level=np.split(sortedlist_array, 2)
    residual=fine_level[:, 1]
    time=np.linspace(t0, Tend, len(residual))
    # breakpoint()
    return residual, time


if __name__=='__main__':
    mlsdc_residual, time=MLSDC_generic_implicit()
    m3lsdc_residual0, time=MLSDC_generic_implicit(zeroth_order=True)
    m3lsdc_residual1, time=first_order_model()
    Residual=[mlsdc_residual, m3lsdc_residual0, m3lsdc_residual1]
    breakpoint()
    labels=['MLSDC', r'M3LSDC $\mathcal{O}(\varepsilon^{0})$', r'M3LSDC $\mathcal{O}(\varepsilon^{1})$']
    plot_residual(time, Residual, labels)
