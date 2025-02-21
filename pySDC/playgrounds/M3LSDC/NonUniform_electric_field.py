import numpy as np
from numba import jit

from pySDC.core.errors import ProblemError
from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.particles import particles, fields, acceleration
from pySDC.implementations.datatype_classes.mesh import mesh

# noinspection PyUnusedLocal
class non_uniformElectric(Problem):
    r"""
    This class implements a standard Penning trap problem on the time interval :math:`[0, t_{end}]`
    fully investigated in [1]_. The equations are given by the following equation of motion

    .. math::
        \frac{dv}{dt}=f(x,v)=\alpha[E(x,t)+v\times B(x,t)],

    .. math::
        \frac{dx}{dt}=v

    with the particles :math:`x, v\in \mathbb{R}^{3}`. For the penning trap problem, the other parameters are given by
    the constant magnetic field :math:`B=\frac{\omega_{B}}{\alpha}\cdot \hat{e_{z}}\in \mathbb{R}^{3}`
    along the :math:`z`-axis with the particle's charge-to-mass ratio :math:`\alpha=\frac{q}{m}` so that

    .. math::
        v\times B=\frac{\omega_{B}}{\alpha}\left(
            \begin{matrix}
            0 & 1 & 0\\
                -1 & 0 & 0\\
                    0 & 0 & 0
            \end{matrix}
            \right)v.

    The electric field :math:`E(x_{i})=E_{ext}(x_{i})+E_{int}(x_{i})\in \mathbb{R}^{3}` where

    .. math::
        E_{ext}(x_{i})=-\epsilon\frac{\omega_{E}^{2}}{\alpha}\left(
            \begin{matrix}
            1 & 0 & 0\\
                0 & 1 & 0\\
                    0 & 0 & -2
            \end{matrix}
            \right)x_{i}

    and the inter-particle Coulomb interaction

    .. math::
        E_{int}(x_{i})=\sum_{k=1, k\neq i}^{N_{particles}}Q_{k}\frac{x_{i}-x_{k}}{(|x_{i}-x_{k}|^{2}+\lambda^{2})^{3/2}}

    with the smoothing parameter :math:`\lambda>0`.
    The exact solution also given for the single particle penning trap more detailed [1]_, [2]_.
    For to solve nonlinear equation of system, Boris trick is used (see [2]_).

    Parameters
    ----------
    omega_B : float
        Amplitude of magnetic field.
    omega_E : float
        Amplitude of electric field.
    u0 : np.1darray
        Initial condition for position, and for velocity.
    q : np.1darray
        Particle's charge.
    m : np.1darray
        Mass.
    nparts : int
        The number of particles.
    sig : float
        The smoothing parameter :math:`\lambda>0`.

    Attributes
    ----------
    work_counter : dict
        Counts the calls of the right-hand side, and calls of the Boris solver.

    References
    ----------
    .. [1] F. Penning. Die Glimmentladung bei niedrigem Druck zwischen koaxialen Zylindern in einem axialen Magnetfeld.
        Physica. Vol. 3 (1936).
    .. [2] Mathias Winkel, Robert Speck and Daniel Ruprecht. A high-order Boris integrator.
        Journal of Computational Physics (2015).
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, omega_B, omega_E, u0, nparts, sig):
        # invoke super init, passing nparts, dtype_u and dtype_f
        super().__init__((12, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nparts', localVars=locals(), readOnly=True)
        self._makeAttributeAndRegister('omega_B', 'omega_E', 'u0', 'sig', localVars=locals())
        self.work_counters['Boris_solver'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()
        self.coarse_zeroth_order=False

    @staticmethod
    @jit(nopython=True, nogil=True)
    def fast_interactions(N, pos, sig, q):
        r"""
        Computes the fast interactions.

        Parameters
        ----------
        N : int
            Number of particles.
        pos : np.2darray
            Position.
        sig : float
            The smoothing parameter :math:`\lambda > 0`.
        q : np.1darray
            Particle's charge.

        Returns
        -------
        Efield : np.2darray
            The internal E field for each particle.
        """
        Efield = np.zeros((3, N))
        contrib = np.zeros(3)

        for i in range(N):
            contrib[:] = 0

            for j in range(N):
                dist2 = (
                    (pos[0, i] - pos[0, j]) ** 2 + (pos[1, i] - pos[1, j]) ** 2 + (pos[2, i] - pos[2, j]) ** 2 + sig**2
                )
                contrib += q[j] * (pos[:, i] - pos[:, j]) / dist2**1.5

            Efield[:, i] += contrib[:]

        return Efield

    def get_interactions(self, part):
        r"""
        Routine to compute the particle-particle interaction, assuming :math:`q = 1` for all particles.

        Parameters
        ----------
        part : dtype_u
            The particles.

        Returns
        -------
        Efield : np.ndarray
            The internal E field for each particle.
        """

        N = self.nparts

        Efield = self.fast_interactions(N, part.pos, self.sig, part.q)

        return Efield

    def eval_f(self, u, t):
        """
        Routine to compute the E and B fields (named f for consistency with the original PEPC version).

        Parameters
        ----------
        part : dtype_u
            The particles.
        t : float
            Current time of the particles (not used here).

        Returns
        -------
        f : dtype_f
            Fields for the particles (internal and external), i.e., the right-hand side of the problem.
        """

        
        self.work_counters['rhs']()
        c=2.0
        Y=u[:6]
        U=u[6:]
        
        Emat = c*np.diag([-1, 1/2, 1/2])
        f = self.dtype_f(self.init)
        f[0]=Y[3]
        f[1]=0.0
        f[2]=0.0
        f[3]=-c*Y[0]
        f[4]=0.0
        f[5]=0.0
        
        f[6]=U[3]
        f[7]=(c/2)*Y[2]
        f[8]=(-c/2)*Y[1]
        f[9]=-c*U[0]
        f[10]=(-c/2)*Y[5]
        f[11]=(c/2)*Y[4]
        
        return f

    def R_matrix(self, t, epsilon, s):
        theta=(t-s)/epsilon
        R=np.eye(3)*np.cos(theta)
        R[0,0]=1.0
        R[1,2]=np.sin(theta)
        R[2,1]=-np.sin(theta)
        return R

    # TODO : Warning, this should be moved to u_exact(t=0) !
    def u_init(self):
        """
        Routine to compute the starting values for the particles.

        Returns
        -------
        u : dtype_u
            Particle set filled with initial data.
        """

        u0 = self.u0
        

        u = self.dtype_u(self.init)
        # breakpoint()
        u[:6]=u0
        u[6:]=np.zeros(6)
        return u
    def fuct(self, x, dt, rhs):
        Y=x[:6]
        U=x[6:]
        c=2.0
        f = self.dtype_f(self.init)
        f[0]=Y[3]
        f[1]=0.0
        f[2]=0.0
        f[3]=-c*Y[0]
        f[4]=0.0
        f[5]=0.0
        
        f[6]=U[3]
        f[7]=(c/2)*Y[2]
        f[8]=(-c/2)*Y[1]
        f[9]=-c*U[0]
        f[10]=(-c/2)*Y[5]
        f[11]=(c/2)*Y[4]
        return x-dt*f-rhs

    def solve_system(self, rhs, dt, u0, t):
        from scipy.optimize import newton
        u_array=np.zeros(12)
        # np.copyto()


        u=newton(self.fuct, u0, args=(dt, rhs))
        me=self.dtype_u(self.init)
        np.copyto(me, u)
        
        return me
        

    def u_exact(self, t):
        r"""
        Routine to compute the exact trajectory at time :math:`t` (only for single-particle setup).

        Parameters
        ----------
        t : float
            Current time of the exact trajectory.

        Returns
        -------
        u : dtype_u
            Particle type containing the exact position and velocity.
        """

        # some abbreviations
        wE = self.omega_E
        wB = self.omega_B
        N = self.nparts
        u0 = self.u0

        if N != 1:
            raise ProblemError('u_exact is only valid for a single particle')

        u = self.dtype_u(((3, 1), self.init[1], self.init[2]))

        wbar = np.sqrt(2) * wE

        # position and velocity in z direction is easy to compute
        u.pos[2, 0] = u0[0][2] * np.cos(wbar * t) + u0[1][2] / wbar * np.sin(wbar * t)
        u.vel[2, 0] = -u0[0][2] * wbar * np.sin(wbar * t) + u0[1][2] * np.cos(wbar * t)

        # define temp. variables to compute complex position
        Op = 1 / 2 * (wB + np.sqrt(wB**2 - 4 * wE**2))
        Om = 1 / 2 * (wB - np.sqrt(wB**2 - 4 * wE**2))
        Rm = (Op * u0[0][0] + u0[1][1]) / (Op - Om)
        Rp = u0[0][0] - Rm
        Im = (Op * u0[0][1] - u0[1][0]) / (Op - Om)
        Ip = u0[0][1] - Im

        # compute position in complex notation
        w = (Rp + Ip * 1j) * np.exp(-Op * t * 1j) + (Rm + Im * 1j) * np.exp(-Om * t * 1j)
        # compute velocity as time derivative of the position
        dw = -1j * Op * (Rp + Ip * 1j) * np.exp(-Op * t * 1j) - 1j * Om * (Rm + Im * 1j) * np.exp(-Om * t * 1j)

        # get the appropriate real and imaginary parts
        u.pos[0, 0] = w.real
        u.vel[0, 0] = dw.real
        u.pos[1, 0] = w.imag
        u.vel[1, 0] = dw.imag

        return u

    def build_f(self, f, part, t):
        """
        Helper function to assemble the correct right-hand side out of B and E field.

        Parameters
        ----------
        f : dtype_f
            The field values.
        part : dtype_u
            The current particles data.
        t : float
            The current time.

        Returns
        -------
        rhs : acceleration
            Correct right-hand side of type acceleration.
        """

        if not isinstance(part, particles):
            raise ProblemError('something is wrong during build_f, got %s' % type(part))

        N = self.nparts

        rhs = acceleration(self.init)
        for n in range(N):
            rhs[:, n] = part.q[n] / part.m[n] * (f.elec[:, n] + np.cross(part.vel[:, n], f.magn[:, n]))

        return rhs

    # noinspection PyTypeChecker
    def boris_solver(self, c, dt, old_fields, new_fields, old_parts):
        r"""
        The actual Boris solver for static (!) B fields, extended by the c-term.

        Parameters
        ----------
        c : dtype_u
            The c term gathering the known values from the previous iteration.
        dt : float
            The (probably scaled) time step size.
        old_fields : dtype_f
            The field values at the previous node :math:`m`.
        new_fields : dtype_f
            The field values at the current node :math:`m+1`.
        old_parts : dtype_u
            The particles at the previous node :math:`m`.

        Returns
        -------
        vel : particles
            The velocities at the :math:`(m+1)`-th node.
        """

        N = self.nparts
        vel = particles.velocity(self.init)
        self.work_counters['Boris_solver']()
        Emean = 0.5 * (old_fields.elec + new_fields.elec)
        for n in range(N):
            a = old_parts.q[n] / old_parts.m[n]

            c[:, n] += dt / 2 * a * np.cross(old_parts.vel[:, n], old_fields.magn[:, n] - new_fields.magn[:, n])

            # pre-velocity, separated by the electric forces (and the c term)
            vm = old_parts.vel[:, n] + dt / 2 * a * Emean[:, n] + c[:, n] / 2
            # rotation
            t = dt / 2 * a * new_fields.magn[:, n]
            s = 2 * t / (1 + np.linalg.norm(t, 2) ** 2)
            vp = vm + np.cross(vm + np.cross(vm, t), s)
            # post-velocity
            vel[:, n] = vp + dt / 2 * a * Emean[:, n] + c[:, n] / 2

        return vel
