import matplotlib.pyplot as plt
import numpy as np

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.playgrounds.M3LSDC.DuffingEquation.problem_class_DuffingEquation import duffingequation
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.projects.Second_orderSDC.penningtrap_HookClass import particles_output

def main():
    level_params=dict()
    level_params['restol']=1e-10
    level_params['dt']=0.015625

    sweeper_params=dict()
    sweeper_params['quad_type']='GAUSS'
    sweeper_params['num_nodes']=5
    sweeper_params['initial_guess']='first_order_Model'
    problem_params=dict()
    problem_params['omega']=1.0
    problem_params['b']=1.0
    problem_params['epsilon']=0.1
    problem_params['u0']=np.array([[2, 0, 0], [0,0,0], [1], [1]], dtype=object)

    
    step_params=dict()
    step_params['maxiter']=10

    controller_params=dict()
    controller_params['hook_class']=particles_output

    controller_params['logger_level']=30

    description=dict()
    description['problem_class']=duffingequation
    description['problem_params']=problem_params
    description['sweeper_params']=sweeper_params
    description['level_params']=level_params
    description['step_params']=step_params
    description['sweeper_class']=boris_2nd_order

    controller=controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    t0=0.0
    Tend=128*0.015625

    P=controller.MS[0].levels[0].prob

    uinit=P.u_init()

    uend, stats=controller.run(u0=uinit, t0=t0, Tend=Tend)
    sortedlist_stats=get_sorted(stats, type='position', sortby='time')
    breakpoint()


if __name__=='__main__':
    main()
