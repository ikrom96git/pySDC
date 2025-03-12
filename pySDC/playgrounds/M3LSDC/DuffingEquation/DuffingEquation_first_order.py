import numpy as np

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class duffingequation_first_order(ptype):
    r"""
    This class implements the stiff Van der Pol oscillator given by the equation

    .. math::
        \frac{d^2 u(t)}{d t^2} - \mu (1 - u(t)^2) \frac{d u(t)}{dt} + u(t) = 0.

    Parameters
    ----------
    u0 : sequence of array_like, optional
        Initial condition.
    mu : float, optional
        Stiff parameter :math:`\mu`.
    newton_maxiter : int, optional
        Maximum number of iterations for Newton's method to terminate.
    newton_tol : float, optional
        Tolerance for Newton to terminate.
    stop_at_nan : bool, optional
        Indicate whether Newton's method should stop if ``nan`` values arise.
    crash_at_maxiter : bool, optional
        Indicates whether Newton's method should stop if maximum number of iterations
        ``newton_maxiter`` is reached.

    Attributes
    ----------
    work_counters : WorkCounter
        Counts different things, here: Number of evaluations of the right-hand side in ``eval_f``
        and number of Newton calls in each Newton iterations are counted.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, u0=None, omega=1.0, b=1.0, epsilon=0.1):
        """Initialization routine"""
        nvars = 4

        if u0 is None:
            u0 = [2.0, 0.0, 0.0, 0.0]

        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'u0', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister(
            'omega', 'b', 'epsilon', localVars=locals()
        )
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()
        self.zeroth_order=False
        self.first_order=True

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine to approximate the exact solution at time t by ``SciPy`` or give initial conditions when called at :math:`t=0`.

        Parameters
        ----------
        t : float
            Current time.
        u_init : pySDC.problem.vanderpol.dtype_u
            Initial conditions for getting the exact solution.
        t_init : float
            The starting time.

        Returns
        -------
        me : dtype_u
            Approximate exact solution.
        """

        me = self.dtype_u(self.init)

        if t > 0.0:

            def eval_rhs(t, u):
                return self.eval_f(u, t)

            me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init)
        else:
            me[:] = self.u0
        return me
    def u_init(self):
        u=self.dtype_u(self.init)
        u[0]=self.u0[0]
        u[1]=self.u0[1]
        u[2]=0.0
        u[3]=0.0
        return u

    def eval_f(self, u, t):
        """
        Routine to compute the right-hand side for both components simultaneously.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time at which the numerical solution is computed (not used here).

        Returns
        -------
        f : dtype_f
            The right-hand side (contains 2 components).
        """

        x1 = u[0]
        x2 = u[1]
        x3=u[2]
        x4=u[3]
        f = self.f_init
        f[0] = x2
        f[1] = -self.omega**2*x1
        f[2]=x4
        f[3]=-self.omega**2*x3-self.b*x1**3
        self.work_counters['rhs']()
        return f

    def right_hand_side(self, x, rhs, dt):
        f0=x[1]
        f1 = -self.omega**2*x[0]
        f2=x[3]
        f3=-self.omega**2*x[2]-self.b*x[1]**3
        return x-dt*np.array([f0, f1, f2, f3])-rhs
        
        

    def solve_system(self, rhs, dt, u0, t):
        """
        Simple Newton solver for the nonlinear system.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        dt : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        u : dtype_u
            The solution u.
        """

        omega = self.omega

        # create new mesh object from u0 and set initial values for iteration
        u = self.dtype_u(u0)
        
        # start newton iteration
        from scipy.optimize import newton

        u_newton=newton(self.right_hand_side, x0=u0, args=(rhs, dt), tol=1e-14)

        np.copyto(u, u_newton)



        return u
