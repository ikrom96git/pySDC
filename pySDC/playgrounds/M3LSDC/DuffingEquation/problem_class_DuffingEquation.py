import numpy as np

from pySDC.implementations.problem_classes.PenningTrap_3D import penningtrap
from pySDC.core.errors import ProblemError
from pySDC.implementations.datatype_classes.particles import particles, fields, acceleration
class duffingequation(penningtrap):
    def __init__(self, omega=1.0, b=1.0, epsilon=0.1, u0=(2, 0)):
        super().__init__(omega_B=0.0, omega_E=omega, u0=u0, nparts=1, sig=1)
        self.name='Duffing equation'
        self._makeAttributeAndRegister('omega', 'b', 'epsilon', 'u0', localVars=locals())
        self.zerosOrderModel=duffingequation_zeros_model(omega=omega, u0=u0)
        self.firstOrderModel=duffingequation_first_model(omega=omega, b=b, epsilon=epsilon, u0=u0)
    def eval_f(self, part, t):
        f=self.dtype_f(self.init)
        Emat1=np.diag([-self.omega**2, 0, 0])
        Emat2=np.diag([-self.epsilon*self.b, 0, 0])
        f.elec=np.zeros((3, self.nparts))
        f.elec=np.dot(Emat1, part.pos)+np.dot(Emat2, part.pos**3)
        f.magn=np.zeros((3, self.nparts))
        return f
    
    
    def u_exact(self, t):
        raise ProblemError('Duffing equation does not have exact solution')
    
class duffingequation_zeros_model(penningtrap):
    def __init__(self, omega=0.1, u0=None):
        super().__init__(omega_B=0.0, omega_E=omega**2, u0=u0, nparts=1, sig=1)
        
        self.name='Duffing equation zeros order reduced model'
        self._makeAttributeAndRegister('omega', 'u0', localVars=locals())

        
    
    def eval_f(self, part, t):
        f=self.dtype_f(self.init)
        Emat=np.diag([-self.omega**2, 0, 0])
        f.elec=np.dot(Emat, part.pos)
        f.magn=np.zeros((3, self.nparts))
        return f
    
    def u_exact(self, t):
        breakpoint()
        
        me=self.dtype_u(self.init)
        z=self.omega*t
        u0=self.u_init()
        M0=u0.pos[0]
        N0=u0.vel[0]
        me.pos=np.zeros((3, 1))
        me.vel=np.zeros((3, 1))
        me.pos[0, 0]=M0*np.cos(z)+N0*np.sin(z)
        me.vel[0, 0]=self.omega*(-M0*np.sin(z)+N0*np.cos(z))
        return me

class duffingequation_first_model(duffingequation):
    def __init__(self, omega=1.0, b=1.0, epsilon=0.1, u0=None):
        u0[0][0]=0.0
        u0[1][0]=0.0
        super().__init__(omega=omega, b=b, epsilon=epsilon, u0=u0)

    def eval_f(self, part, t):
        return super().eval_f(part, t)
    
    def build_f(self, f, part, t):
        righthandside = -(0.25 * (self.duffing_zeros.u0[0] ** 3) * self.b) * (np.cos(3 * z) + 3 * np.cos(z))

        return super().build_f(f, part, t)+np.array([righthandside, 0, 0])

    def u_exact(self, t):
        
        
        me = self.dtype_u(self.init)
        M1 = -(self.zerosOrderModel.u0[0][0] ** 3 * self.b) / (32 * self.omega**2)
        N1 = 0.0
        z = self.omega * t
        me.pos=np.zeros((3, 1))
        me.vel=np.zeros((3, 1))
        me.pos[0] = M1 * np.cos(z) + N1 * np.sin(z) + M1 * (12 * z * np.sin(z) - np.cos(3 * z))
        me.vel[0] = (
            -M1 * self.omega * np.sin(z)
            + N1 * self.omega * np.cos(z)
            + M1 * self.omega * (12 * np.sin(z) + 12 * z * np.cos(z) + 3 * np.sin(3 * z))
        )
        return me
    
    def u_asymptotic_expansion(self, t):
        
        me = self.dtype_u(self.init)
        u_zeros = self.zerosOrderModel.u_exact(t)
        u_first = self.u_exact(t)
        me.pos = u_zeros.pos + self.epsilon * u_first.pos
        me.vel = u_zeros.vel + self.epsilon * u_first.vel
        return me

        