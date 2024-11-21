import numpy as np
from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.Second_orderSDC.penningtrap_HookClass import particles_output
from pySDC.projects.Second_orderSDC.penningtrap_params import penningtrap_params_mlsdc
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.implementations.transfer_classes.TransferParticles_NoCoarse import particles_to_particles
def Error_mlsdc():
    transfer_params = dict()
    transfer_params['finter'] = False
    step_params = dict()
    step_params['maxiter'] = 10
    controller_params_mlsdc, description_mlsdc=penningtrap_params_mlsdc()
    controller_params_mlsdc['hook_class']=particles_output
    description_mlsdc['space_transfer_class'] = particles_to_particles
    description_mlsdc['base_transfer_params'] = transfer_params
    description_mlsdc['step_params']=step_params

    u_val, uex_val=dict(), dict()
    values, error=['position', 'velocity'], dict()
    controller=controller_nonMPI(num_procs=1, controller_params=controller_params_mlsdc, description=description_mlsdc)
    t0, Tend=0.0, 2
    P=controller.MS[0].levels[0].prob
    uinit=P.u_init()
    uend, stats_mlsdc=controller.run(u0=uinit, t0=t0, Tend=Tend)
    for nn in values:
        u_val[nn]=get_sorted(stats_mlsdc, type=nn, sortby="time")
        uex_val[nn]=get_sorted(stats_mlsdc, type=f'{nn}_exact', sortby='time')
        error[nn]=relative_error(uex_val[nn], u_val[nn])
        error[nn]=list(error[nn].T[0])
    breakpoint()


def relative_error(uex_data, u_data):
        u_ex = np.array([entry[1] for entry in uex_data])
        u = np.array([entry[1] for entry in u_data])
        return np.linalg.norm(np.abs((u_ex - u)), np.inf, 0) / np.linalg.norm(u_ex, np.inf, 0)


if __name__=='__main__':
     Error_mlsdc()