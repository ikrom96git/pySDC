import numpy as np

from pySDC.core.problem import Problem
from pySDC.core.errors import ProblemError
from pySDC.implementations.problem_classes.HarmonicOscillator import harmonic_oscillator
from pySDC.implementations.datatype_classes.particles import particles, acceleration
from scipy.integrate import solve_ivp
class duffingequation(Problem):
    dtype_u=particles
    dtype_f=acceleration

    def __init__(self, omega=1.0, b=1.0, epsilon=0.1, u0=(1,0)):
        u0=np.asarray(u0)
        super().__init__((1, None, np.dtype("float64")))
        self._makeAttributeAndRegister('omega', 'b', 'u0', 'epsilon', localVars=locals(), readOnly= True)

    def eval_f(self, u, t):
        
        me=self.dtype_f(self.init)
        me[:]=-self.omega**2*u.pos-self.epsilon*self.b*u.pos**3

        return me
    
    def u_init(self):

        u0=self.u0
        u=self.dtype_u(self.init)
        u.pos[0]=u0[0]
        u.vel[0]=u[1]

    def u_exact(self, t):
        raise ProblemError('Exact solution of Duffing equation does not exist')
    
    def scipy_solve_ivp(self, t):
        

class duffing_zeros_model(harmonic_oscillator):
    def __init__(self, omega=1.0, u0=(1, 0)):
        super().__init__(k=omega**2, u0=u0)
    
   
class duffing_first_model(duffingequation):
    def __init__(self, omega=1, b=1, epsilon=0.1, u0=(1, 0)):
        super().__init__(omega, b, epsilon, u0)

    def eval_f(self, u, t):
        me=self.dtype_f(self.init)
        z=t*self.omega
        righthandside=-(0.25*self.u0[0]*self.b)*(np.cos(3*z)+3*np.cos(z))

        me[:]=-self.omega**2*u.pos+righthandside
        return me
    
    def u_init(self):
        
        u=self.dtype_u(self.init)
        u.pos[0]=0.0
        u.vel[0]=0.0
        return u
    
    def u_exact(self, t):
        me=self.dtype_u(self.init)

        z=self.omega*t

        me.pos=(self.u0[0]**3*self.b*np.sin(z))*(np.sin(2*z)+6*z)/(16*self.omega**2)
        me.vel=-(self.u0[0]*self.b/(32*self.omega))*(3*np.sin(3*z)+11*np.sin(z)+12*z*np.cos(z))
        return me

    def u_asymptotic_expan(self, u_zeros, u_first):
        me=self.dtype_u(self.init)
        me.pos=u_zeros.pos+self.epsilon*u_first.pos
        me.vel=u_zeros.vel+self.epsilon*u_first.vel
        return me

if __name__=='__main__':
    problem_params=dict()
    problem_params['k']=1.0
    problem_params['mu']=0.0
    problem_params['u0']=np.array([1, 0])
    zeros=duffing_zeros_model()
    u_ex=zeros.u_exact(0.1)
    breakpoint()
    print(u_ex)
