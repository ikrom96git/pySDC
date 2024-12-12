import numpy as np

from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.core.errors import ProblemError
from pySDC.implementations.datatype_classes.particles import particles, fields, acceleration
class duffingequation(penningtrap):
    def __init__(self, omega=1.0, b=1.0, epsilon=0.1, u0=(2, 0)):
        super().__init__(omega_B=0.0, omega_E=omega, u0=u0, nparts=1, sig=1)
        self._makeAttributeAndRegister('omega', 'b', 'epsilon', 'u0', localVars=locals())
        
    def eval_f(self, part, t):
        f=self.dtype_f(self.init)
        Emat1=np.diag([-self.omega**2, 0, 0])
        Emat2=np.diag([-self.epsilon*self.b, 0, 0])
        f.elec=np.zeros((3, self.nparts))
        f.elec=np.dot(Emat1, part.pos)+np.dot(Emat2, part.pos**3)
        f.magn=np.zeros((3, self.nparts))
        return f
    
    def u_init(self):
        """
        #Returns
        -------
        #u : dtype_u
        #    Particle set filled with initial data.
        """

        u0 = self.u0
        N = self.nparts

        u = self.dtype_u(self.init)

        if u0[2][0] != 1 or u0[3][0] != 1:
            raise ProblemError('so far only q = m = 1 is implemented')

        # set first particle to u0
        u.pos[0, 0] = u0[0][0]
        u.pos[1, 0] = u0[0][1]
        u.pos[2, 0] = u0[0][2]
        u.vel[0, 0] = u0[1][0]
        u.vel[1, 0] = u0[1][1]
        u.vel[2, 0] = u0[1][2]

        u.q[0] = u0[2][0]
        u.m[0] = u0[3][0]
        return u
    
    def u_exact(self, t):
        raise ProblemError('Duffing equation does not have exact solution')

    def build_f(self, f, part, t):
        if not isinstance(part, particles):
            raise ProblemError('something is wrong during build_f, got %s' % type(part))

        N = self.nparts

        rhs = acceleration(self.init)
        for n in range(N):
            rhs[:, n] = part.q[n] / part.m[n] * (f.elec[:, n] + np.cross(part.vel[:, n], f.magn[:, n]))

        return rhs