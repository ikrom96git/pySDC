import numpy as np
import matplotlib.pyplot as plt
from pySDC.implementations.problem_classes.DuffingEquation import duffingequation, duffing_zeros_model, duffing_first_model
import matplotlib.pyplot as plt
def duffing_eqaution_default_params():
    problem_params=dict()
    problem_params['omega']=1.0
    problem_params['b']=1.0
    problem_params['epsilon']=0.1
    problem_params['u0']=np.array([1, 0])
    return problem_params

def solution_Duffing_equation():
    problem_params=duffing_eqaution_default_params()
    duffing=duffingequation()
    t0=0.0
    tend=2*np.pi 
    time=np.linspace(t0, tend, 1000)
    duffing_solution=duffing.scipy_solve_ivp(tend, time)
    return duffing_solution.pos
def solution_duffing_zeros_model():
    duffing_zeros=duffing_zeros_model()
    t0=0.0
    tend=2*np.pi 
    time=np.linspace(t0, tend, 1000)
    u_pos=[]
    for t in time:
        solution=duffing_zeros.u_exact(t)
        u_pos=np.append(u_pos, solution.pos)
    return u_pos, time
def solution_duffing_first_model():
    duffing_first=duffing_first_model()
    t0=0.0
    tend=2*np.pi 
    time=np.linspace(t0, tend, 1000)
    u_pos=[]
    for t in time:
        solution=duffing_first.u_asymptotic_solution(t)
        u_pos=np.append(u_pos, solution.pos)
    return u_pos, time

def plot_solution(Solution, time, label):
    for nn in range(len(label)):
        plt.plot(time, Solution[nn], label=label[nn])
    plt.legend()
    plt.tight_layout()
    plt.show()




if __name__=='__main__':
    solution_ivp=solution_Duffing_equation()
    zeros_model_sol, time=solution_duffing_zeros_model()
    first_models_sol, time=solution_duffing_first_model()
    Solution=[solution_ivp, zeros_model_sol, first_models_sol]
    label=['solution_ivp', 'zeros model', 'first model']
    plot_solution(Solution, time, label)





