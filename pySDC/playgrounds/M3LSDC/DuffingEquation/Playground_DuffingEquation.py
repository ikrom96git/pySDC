import time
from pathlib import Path

import numpy as np

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.playgrounds.M3LSDC.DuffingEquation.DuffingEquation import duffingequation, duffingequation_D4
from pySDC.playgrounds.M3LSDC.DuffingEquation.DuffingEquation_zeroth_order import duffingequation_zeroth_order
from pySDC.playgrounds.M3LSDC.DuffingEquation.DuffingEquation_first_order import duffingequation_first_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh_NoCoarse import mesh_to_mesh
from pySDC.playgrounds.M3LSDC.plot_residual import plot_residual


dt=0.01
EPSILON=.1

def MLSDC_duffing_equation(zeroth_order=False):
    """
    A simple test program to run IMEX SDC for a single time step
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = -1
    level_params['dt'] = dt
    level_params['nsweeps']=[5,1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'GAUSS'
    sweeper_params['num_nodes'] = [5,5]

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['omega'] = 1.0  # E-field frequency
    problem_params['b'] = 1.0  # B-field frequency
    problem_params['epsilon']=EPSILON
    problem_params['u0'] = np.array([2.0, 0.0])  # initial center of positions
    
    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 5

    # initialize controller parameters
    controller_params = dict()
    # controller_params['hook_class'] = particle_hook  # specialized hook class for more statistics and output
    # controller_params['logger_level'] = 30

    transfer_params = dict()
    transfer_params['finter'] = False

    # Fill description dictionary for easy hierarchy creation
    description = dict()
        # MLSDC: provide list of two problem classes: one for the fine, one for the coarse level
    
    if zeroth_order:
        description['problem_class']=[duffingequation, duffingequation_zeroth_order]
    else:
        description['problem_class'] = [duffingequation, duffingequation]
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh
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
    residual_with_level=get_sorted(stats, type='residual_post_iteration', sortby='level')
    residual_with_level=get_sorted(stats, type='residual_post_iteration', sortby='level')
    residual=np.asarray(residual_with_level)
    iter=np.arange(0, step_params['maxiter'], 1)
    return residual[:,1], iter
    

def M3LSDC_first_order():
    """
    A simple test program to run IMEX SDC for a single time step
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = -1
    level_params['dt'] = dt
    level_params['nsweeps']=[1 ,5]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'GAUSS'
    sweeper_params['num_nodes'] = [5,5]

    # initialize problem parameters for the Penning trap
    problem_params = dict()
    problem_params['omega'] = 1.0  # E-field frequency
    problem_params['b'] = 1.0  # B-field frequency
    problem_params['epsilon']=EPSILON
    # problem_params['u0'] = np.array([2.0, 0.0, 0.0, 0.0])  # initial center of positions
    
    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 5

    # initialize controller parameters
    controller_params = dict()
    # controller_params['hook_class'] = particle_hook  # specialized hook class for more statistics and output
    # controller_params['logger_level'] = 30

    transfer_params = dict()
    transfer_params['finter'] = False

    # Fill description dictionary for easy hierarchy creation
    description = dict()
        # MLSDC: provide list of two problem classes: one for the fine, one for the coarse level
    description['problem_class'] = [duffingequation, duffingequation_first_order]
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh
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
    residual_with_level=get_sorted(stats, type='residual_post_iteration', sortby='level')
    residual=np.asarray(residual_with_level)
    iter=np.arange(0, step_params['maxiter'], 1)
    return residual[:,1], iter
    

if __name__ == "__main__":
    mlsdc_residual, iter=MLSDC_duffing_equation()
    m3lsdc0_residual, iter=MLSDC_duffing_equation(zeroth_order=True)
    m3lsdc1_residual, iter=M3LSDC_first_order()
    residual=[mlsdc_residual, m3lsdc0_residual, m3lsdc1_residual]
    labels=['MLSDC', r"M3LSDC $\mathcal{O}(\varepsilon^{0})$", r'M3LSDC $\mathcal{O}(\varepsilon^{1})$']
    title=rf'Duffing Equation, $\varepsilon={EPSILON}, \ \Delta t={dt}$'
    plot_residual(iter, residual, labels, title)
