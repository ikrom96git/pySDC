import numpy as np
from numba import jit

from pySDC.core.errors import ProblemError
from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh


# noinspection PyUnusedLocal
class uniform_electric_field(Problem):
   

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, epsilon, u0):
        # invoke super init, passing nparts, dtype_u and dtype_f\
        nvars=12
        super().__init__((6, None, np.dtype('float64')))
        self._makeAttributeAndRegister('epsilon', 'u0', localVars=locals())
        self.work_counters['Boris_solver'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()
        self.coarse_zeroth_order=False
        self.coarse_first_order=False

 

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
        
        
        
        
        f = self.dtype_f(self.init)
        f[0]=u[3]
        f[1]=u[4]
        f[2]=u[5]
        f[3]=0.0
        f[4]=(1/self.epsilon)*u[5]+np.sin(t/self.epsilon)
        f[5]=-(1/self.epsilon)*u[4]+np.cos(t/self.epsilon)
        return f

    def R_matrix(self, t, epsilon, s):
        theta=(t-s)/epsilon
        R=np.eye(3)*np.cos(theta)
        R[0,0]=1.0
        R[1,2]=np.sin(theta)
        R[2,1]=-np.sin(theta)
        return R
    
    def RR_matrix(self, t, epsilon, s):
        theta=(t-s)/epsilon
        RR=np.eye(3)*np.sin(theta)
        RR[0,0]=0.0
        RR[1,2]=1.0-np.cos(theta)
        RR[2,1]=np.cos(theta)-1.0
        return RR

    def E_mat(self, y):
        c=2.0
        return c*np.array([-y[0], y[1]/2, y[2]/2])
    
    def G_expansion(self, y, t, eps, s=0.0):
        y0, u0, y1, u1=np.split(y, 4)
        R=self.R_matrix(t, eps, s)
        RR=self.RR_matrix(t, eps, s)
        E=self.E_mat(y0)
        G0=np.concatenate((y0, R@u0))
        G1=np.concatenate((y1+RR@u0, R@u1+RR@E))
        G=np.concatenate((G0+eps*G1, np.zeros(6)))
        return G


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
        np.copyto(u, u0)
        return u
    def fuct(self, x, dt, rhs, t):
        
        f = self.dtype_f(self.init)
        f[0]=x[3]
        f[1]=x[4]
        f[2]=x[5]
        f[3]=0.0
        f[4]=(1/self.epsilon)*x[5]+np.sin(t/self.epsilon)
        f[5]=-(1/self.epsilon)*x[4]+np.cos(t/self.epsilon)
        return x-dt*f-rhs

    def solve_system(self, rhs, dt, u0, t):
        from scipy.optimize import newton
        

        u=newton(self.fuct, u0, args=(dt, rhs,t), tol=1e-14)
        me=self.dtype_u(self.init)
        np.copyto(me, u)
        
        return me
        
        
        
class uniform_electric_field_6D(Problem):
   

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, epsilon, u0):
        # invoke super init, passing nparts, dtype_u and dtype_f\
        nvars=12
        super().__init__((12, None, np.dtype('float64')))
        self._makeAttributeAndRegister('epsilon', 'u0', localVars=locals())
        self.work_counters['Boris_solver'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()
        self.coarse_zeroth_order=False
        self.coarse_first_order=False

 

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
        
        
        
        
        f = self.dtype_f(self.init)
        f[0]=u[3]
        f[1]=u[4]
        f[2]=u[5]
        f[3]=0.0
        f[4]=(1/self.epsilon)*u[5]+np.sin(t/self.epsilon)
        f[5]=-(1/self.epsilon)*u[4]+np.cos(t/self.epsilon)
        f[6]=0.0
        f[7]=0.0
        f[8]=0.0
        f[9]=0.0
        f[10]=0.0
        f[11]=0.0
        return f

    def R_matrix(self, t, epsilon, s):
        theta=(t-s)/epsilon
        R=np.eye(3)*np.cos(theta)
        R[0,0]=1.0
        R[1,2]=np.sin(theta)
        R[2,1]=-np.sin(theta)
        return R
    
    def RR_matrix(self, t, epsilon, s):
        theta=(t-s)/epsilon
        RR=np.eye(3)*np.sin(theta)
        RR[0,0]=0.0
        RR[1,2]=1.0-np.cos(theta)
        RR[2,1]=np.cos(theta)-1.0
        return RR

    def E_mat(self, y):
        c=2.0
        return c*np.array([-y[0], y[1]/2, y[2]/2])
    
    def G_expansion(self, y, t, eps, s=0.0):
        y0, u0, y1, u1=np.split(y, 4)
        R=self.R_matrix(t, eps, s)
        RR=self.RR_matrix(t, eps, s)
        E=self.E_mat(y0)
        G0=np.concatenate((y0, R@u0))
        G1=np.concatenate((y1+RR@u0, R@u1+RR@E))
        G=np.concatenate((G0+eps*G1, np.zeros(6)))
        return G


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
        np.copyto(u, u0)
        return u
    def fuct(self, x, dt, rhs, t):
        
        f = self.dtype_f(self.init)
        f[0]=x[3]
        f[1]=x[4]
        f[2]=x[5]
        f[3]=0.0
        f[4]=(1/self.epsilon)*x[5]+np.sin(t/self.epsilon)
        f[5]=-(1/self.epsilon)*x[4]+np.cos(t/self.epsilon)
        f[6]=0.0
        f[7]=0.0
        f[8]=0.0
        f[9]=0.0
        f[10]=0.0
        f[11]=0.0
        return x-dt*f-rhs

    def solve_system(self, rhs, dt, u0, t):
        from scipy.optimize import newton
        

        u=newton(self.fuct, u0, args=(dt, rhs,t), tol=1e-14)
        me=self.dtype_u(self.init)
        np.copyto(me, u)
        
        return me
        
        

   