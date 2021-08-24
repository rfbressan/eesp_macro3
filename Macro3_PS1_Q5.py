from scipy.optimize import root
import numpy as np

class twop_economy():
    
    def __init__(self, A0, A1, alpha, sigma, beta, gama, r, tol=1e-4) -> None:
        self.A0, self.A1, self.alpha, self.sigma, self.beta, self.gama, self.r = A0, A1, alpha, sigma, beta, gama, r
        self.tol = tol
        self.eq = None # Equilibrium values
    
    def u(self, c):
        return (c**(1-self.sigma)-1)/(1-self.sigma)
    
    def v(self, l):
        return (l**(1-self.sigma)-1)/(1-self.sigma)
    
    def du(self, c):
        return c**(-self.sigma)
    
    def dv(self, l):
        return l**(-self.sigma)
    
    def solve_ss(self, x0) -> None:
        
        def obj_fun(x):
            F = np.empty(9)
            c0, c1, h0, h1, a1, w0, w1, pi0, pi1 = x
            # Solving the system of nonlinear equations
            F = np.array([
                        #################################################
                        # Using one equation for each endogenous variable
                        c0 - (w0*h0+pi0-a1),
                        c1 - (((1+self.r)*self.beta)**(1/self.sigma)*c0),
                        h0 - (1-(self.gama/w0)**(1/self.sigma)*c0),
                        h1 - (1-(self.gama/w1)**(1/self.sigma)*c1),
                        a1 - (c1-w1*h1-pi1)/(1+self.r),
                        w0 - (self.alpha*self.A0*(h0)**(self.alpha-1)),
                        w1 - (self.alpha*self.A1*(h1)**(self.alpha-1)),
                        pi0 - ((1-self.alpha)*self.A0*(h0)**self.alpha),
                        pi1 - ((1-self.alpha)*self.A1*(h1)**self.alpha)
                        ])
            return F
        
        self.eq = root(obj_fun, x0, method='hybr')
    
    def check_constraints(self) -> None:
        c0, c1, h0, h1, a1, w0, w1, pi0, pi1 = self.eq.x
        print("Checking Equilibrium!")
        print(f"Euler equation: {abs(self.du(c0)-(1+self.r)*self.beta*self.du(c1)) < self.tol}")
        print(f"Budget constraint t0: {abs(c0+a1-w0*h0-pi0 < self.tol)}")
        print(f"Budget constraint t1: {abs(c1-w1*h1-(1+self.r)*a1 - pi1 < self.tol)}")
        
parameters = {'A0':1, 'A1':1, 'alpha': 2/3, 'sigma': 2, 'beta': 0.98**25, 'gama': 1, 'r': 1.05**25-1}
econ = twop_economy(**parameters)
# Order for x0: c0, c1, h0, h1, a1, w0, w1, pi0, pi1
x0 = np.array([0.5, 0.9, 0.5, 0.5, 0.2, 0.9, 0.9, 0.5, 0.5])
x0a = np.array([0.5, 0.9, 0.3, 0.6, 0.1, 0.3, 0.6, 0.35, 0.25])
econ.solve_ss(x0)
print(f"Root finding successfull? {econ.eq.success}")
result = dict(zip(["c0", "c1", "h0", "h1", "a1", "w0", "w1", "pi0", "pi1"], econ.eq.x))
print("\n")
print(result)
# Checking constraints
econ.check_constraints()
# New initial guess
econ.solve_ss(x0a)
resulta = dict(zip(["c0", "c1", "h0", "h1", "a1", "w0", "w1", "pi0", "pi1"], econ.eq.x))
print("\n")
print(resulta)
# Checking constraints
econ.check_constraints()