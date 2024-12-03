import numpy as np
import matplotlib.pyplot as plt
from pySDC.implementations.problem_classes.DuffingEquation import duffingequation, duffing_zeros_model, duffing_first_model

def duffing_eqaution_default_params():
    problem_params=dict()
    problem_params['omega']=1.0
    problem_params['b']=1.0
    problem_params['epsilon']=0.1
    problem_params['u0']=np.array([1, 0])
    return problem_params

def solution_Duffing_equation():
    problem_params=duffing_eqaution_default_params()
    duffing=duffingequation(problem_params)
    
