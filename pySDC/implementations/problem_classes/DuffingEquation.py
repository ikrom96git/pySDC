import numpy as np

from pySDC.core.problem import Problem
from pySDC.core.errors import ProblemError
from pySDC.implementations.problem_classes.HarmonicOscillator import harmonic_oscillator
from pySDC.implementations.datatype_classes.particles import particles, acceleration
from scipy.integrate import solve_ivp


class duffingequation(Problem):

    r"""
    Example implementing the Duffing equation 

    .. math::
        \ddot{x}+\omega^2 x+\varepsilon b x^{3}=0
    with initial conditions are 

    .. math::
        x(0)=a, \quad v(0)=\dot{x}(0)=0
    which is a second-order problem. The unknown function :math:`x` denotes the position of the mass, and the
    derivative is the velocity. :math:`\mu` defines the damping and :math:`k` is the spring constant.

    Parameters
    ----------
    k : float, optional
        Spring constant :math:`k`.
    mu : float, optional
        Damping parameter :math:`\mu`.
    u0 : tuple, optional
        Initial condition for the position, and the velocity. Should be a tuple, e.g. ``u0=(1, 0)``.
    phase : float, optional
        Phase of the oscillation.
    amp : float, optional
        Amplitude of the oscillation.
    Source: https://beltoforion.de/en/harmonic_oscillator/
    """

    dtype_u = particles
    dtype_f = acceleration

    def __init__(self, omega=1.0, b=1.0, epsilon=0.1, u0=(2, 0)):
        u0 = np.asarray(u0)
        super().__init__((1, None, np.dtype("float64")))
        self._makeAttributeAndRegister(
            'omega', 'b', 'u0', 'epsilon', localVars=locals(), readOnly=True)

    def eval_f(self, u, t):

        me = self.dtype_f(self.init)
        me[:] = -self.omega**2*u.pos-self.epsilon*self.b*u.pos**3

        return me

    def u_init(self):

        u0 = self.u0
        u = self.dtype_u(self.init)
        u.pos[0] = u0[0]
        u.vel[0] = u[1]
        return u

    def u_exact(self, t):
        raise ProblemError('Exact solution of Duffing equation does not exist')

    def duffing_rhs(self, t, u):
        return [u[1], -(self.omega**2)*u[0]-self.epsilon*self.omega*u[0]**3]

    def scipy_solve_ivp(self, Tend, t_span):
        u = self.dtype_u(self.init)
        solve = solve_ivp(self.duffing_rhs, [0, Tend], self.u0, t_eval=t_span)
        solution = solve.y
        u.pos = solution[0, :]
        u.vel = solution[1, :]
        return u


class duffing_zeros_model(harmonic_oscillator):
    def __init__(self, omega=1.0, u0=(2, 0)):
        super().__init__(k=omega**2, u0=u0)


class duffing_first_model(duffingequation):
    def __init__(self, omega=1, b=1, epsilon=0.1, u0=(0, 0)):
        super().__init__(omega, b, epsilon, u0)
        self.duffing_zeros = duffing_zeros_model()

    def eval_f(self, u, t):
        me = self.dtype_f(self.init)
        z = t*self.omega
        righthandside = -(0.25*self.u0[0]*self.b)*(np.cos(3*z)+3*np.cos(z))

        me[:] = -self.omega**2*u.pos+righthandside
        return me

    def u_init(self):

        u = self.dtype_u(self.init)
        u.pos[0] = 0.0
        u.vel[0] = 0.0
        return u

    def u_exact(self, t):

        me_first = self.dtype_u(self.init)
        M1 = -(self.duffing_zeros.u0[0]**3*self.b)/(32*self.omega**2)
        N1 = 0.0
        z = self.omega*t
        me_first.pos = M1 * np.cos(z) + N1 * np.sin(z) + \
            M1 * (12 * z * np.sin(z) - np.cos(3 * z))
        me_first.vel = (-M1 * self.omega * np.sin(z) + N1 * self.omega * np.cos(z) +
                        M1 * self.omega * (12 * np.sin(z) + 12 * z * np.cos(z) + 3 * np.sin(3 * z)))

        return me_first

    def u_asymptotic_solution(self, t):
        me = self.dtype_u(self.init)
        u_zeros = self.duffing_zeros.u_exact(t)
        u_first = self.u_exact(t)
        me.pos = u_zeros.pos+self.epsilon*u_first.pos
        me.vel = u_zeros.vel+self.epsilon*u_first.vel
        return me
