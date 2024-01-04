import numpy as np
from pySDC.implementations.problem_classes.HarmonicOscillator import harmonic_oscillator
from pySDC.implementations.sweeper_classes.boris_2nd_order import boris_2nd_order
from pySDC.projects.Second_orderSDC.penningtrap_Simulation import Stability_implementation


def dampedharmonic_oscillator_params():
    """
    Routine to compute modules of the stability function

    Returns:
        description (dict): A dictionary containing parameters for the damped harmonic oscillator problem
    """

    # Initialize level parameters
    level_params = {'restol': 1e-16, 'dt': 1.0}

    # Initialize problem parameters for the Damped harmonic oscillator problem
    problem_params = {'k': 0, 'mu': 0, 'u0': np.array([1, 1])}

    # Initialize sweeper parameters
    sweeper_params = {'quad_type': 'GAUSS', 'num_nodes': 3, 'do_coll_update': True, 'picard_mats_sweep': True}

    # Initialize step parameters
    step_params = {'maxiter': 2}

    # Fill description dictionary for easy step instantiation
    description = {
        'problem_class': harmonic_oscillator,
        'problem_params': problem_params,
        'sweeper_class': boris_2nd_order,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    return description


if __name__ == '__main__':
    """
    Damped harmonic oscillator as a test problem for the stability plot:
        x' = v
        v' = -kappa * x - mu * v
        kappa: spring constant
        mu: friction
        Source: https://beltoforion.de/en/harmonic_oscillator/
    """
    # Execute the stability analysis for the damped harmonic oscillator
    description = dampedharmonic_oscillator_params()
    Stability = Stability_implementation(description, kappa_max=30, mu_max=30, Num_iter=(200, 200))

    Stability.run_SDC_stability()
    Stability.run_Picard_stability()
    Stability.run_RKN_stability()
    Stability.run_Ksdc()
    # Stability.run_Kpicard
