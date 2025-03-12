import numpy as np
from scipy.optimize import newton 

from pySDC.core.errors import ProblemError
from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class vanderpol(Problem):
   

    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self,
        u0=None,
        omega=1.0,
        epsilon=0.1,
        b=1.0,
        stop_at_nan=True,
        crash_at_maxiter=True,
        relative_tolerance=False,
    ):
        """Initialization routine"""
        nvars = 2

        if u0 is None:
            u0 = [2.0, 0.0]

        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('u0', 'omega', 'epsilon', 'b', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister(
            'mu',
            'newton_maxiter',
            'newton_tol',
            'stop_at_nan',
            'crash_at_maxiter',
            'relative_tolerance',
            localVars=locals(),
        )
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

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
        f = self.f_init
        f[0] = x2
        f[1] = -self.omega**2 * x1 - self.epsilon*self.b*x1**3
        self.work_counters['rhs']()
        return f

    def right_hand_side(self, u, rhs, dt, u0, t):
        x1=u[0]
        x2=u[1]
        f[0]=x2
        f[1]=-self.omega**2*x1-self.epsilon*self.b*x1**3
        return f

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

        mu = self.mu

        # create new mesh object from u0 and set initial values for iteration
        u = self.dtype_u(u0)
        x1 = u[0]
        x2 = u[1]
        root=newton(self.right_hand_side, args=(rhs, dt, u0, t))
        np.copyto(u, root)
        
        return u

