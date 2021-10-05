# -*- coding: utf-8 -*-

from IPython import get_ipython

import matplotlib.pyplot as plt
plt.rc('figure', figsize=(12, 6))
import numpy as np
import quantecon as qe
from quantecon.markov import DiscreteDP
from numba import jit
from scipy.optimize import root
from scipy.interpolate import splev, splrep
get_ipython().run_line_magic('matplotlib', 'inline')

class Household:
    """
    This class takes the parameters that define a household asset accumulation
    problem and computes the corresponding reward and transition matrices R
    and Q required to generate an instance of DiscreteDP, and thereby solve
    for the optimal policy.

    Comments on indexing: We need to enumerate the state space S as a sequence
    S = {0, ..., n}.  To this end, (a_i, z_i) index pairs are mapped to s_i
    indices according to the rule

        s_i = a_i * z_size + z_i

    To invert this map, use

        a_i = s_i // z_size  (integer division)
        z_i = s_i % z_size

    """


    def __init__(self,
                r=0.01,                      # Interest rate
                w=1.0,                       # Wages
                β=0.96,                      # Discount factor
                a_min=1e-10,
                Π=[[0.9, 0.1], [0.1, 0.9]],  # Markov chain
                z_vals=[0.1, 1.0],           # Exogenous states
                a_max=18,
                a_size=80,
                sigmac=2,
                sigmal=2,
                gamma=0.75):

        # Store values, set up grids over a and z
        self.r, self.w, self.β = r, w, β
        self.a_min, self.a_max, self.a_size = a_min, a_max, a_size
        self.sigmac, self.sigmal, self.gamma = sigmac, sigmal, gamma

        self.Π = np.asarray(Π)
        self.z_vals = np.asarray(z_vals)
        self.z_size = len(z_vals)

        self.a_vals = np.linspace(a_min, a_max, a_size)
        self.n = a_size * self.z_size
        

        # Build the array Q
        self.Q = np.zeros((self.n, a_size, self.n))
        self.build_Q()

        # Build the array R
        self.R = np.empty((self.n, a_size))
        self.build_R()

    def set_prices(self, r, w):
        """
        Use this method to reset prices. Calling the method will trigger a
        re-build of R.
        """
        self.r, self.w = r, w
        self.build_R()

    def build_Q(self):
        populate_Q(self.Q, self.a_size, self.z_size, self.Π)

    def build_R(self):
        self.R.fill(-np.inf)
        populate_R(self.R,
                self.a_size,
                self.z_size,
                self.a_vals,
                self.z_vals,
                self.r,
                self.w,
                self.sigmac,
                self.sigmal,
                self.gamma)
    
    def get_hh_params(self):
        return self.sigmac, self.sigmal, self.gamma
    
class Firm():
    """Firm in an Aiyagari econony

    Returns:
        [type]: [description]
    """
    
    def __init__(self, r, alpha = 0.4, delta = 0.08, A=1) -> None:
        self.A = A
        self.alpha = alpha
        self.delta = delta
        self.r = r
    
    def output(self, K, N):
        return self.A*(K)**self.alpha*(N)**(1-self.alpha)
    
    def wage(self, r):
        a, d = self.alpha, self.delta
        return self.A*(1-a)*((self.A*a)/(r+d))**(a/(1-a))
    
    def capital_ratio(self, r):
        a, d = self.alpha, self.delta
        return ((self.A*a)/(r+d))**(1/(1-a))
    
    def rd(self, k):
        """Inverse demand of capital ratio

        Args:
            k (float): capital ratio

        Returns:
            float: interest rate associated with a given demand for k
        """
        a, d = self.alpha, self.delta
        return self.A*a*(k**(a-1))-d

# Do the hard work using JIT-ed functions

# @jit(nopython=True)
def populate_R(R, a_size, z_size, a_vals, z_vals, r, w, sigmac, sigmal, gamma):
    n = a_size * z_size
    for s_i in range(n):
        a_i = s_i // z_size
        z_i = s_i % z_size
        a = a_vals[a_i]
        z = z_vals[z_i]
        for new_a_i in range(a_size):
            a_new = a_vals[new_a_i]
            c = compute_c(w, r, a, z, a_new, sigmac, sigmal, gamma) # Compute c from NL equation
            l = compute_l(w, z, c, sigmac, sigmal, gamma)  # Compute l from intra-temporal
            if c > 0:
                R[s_i, new_a_i] = u(c, l, sigmac, sigmal, gamma)  # Utility

@jit(nopython=True)
def populate_Q(Q, a_size, z_size, Π):
    n = a_size * z_size
    for s_i in range(n):
        z_i = s_i % z_size
        for a_i in range(a_size):
            for next_z_i in range(z_size):
                Q[s_i, a_i, a_i*z_size + next_z_i] = Π[z_i, next_z_i]
                
@jit(nopython=True)
def u(c, l, sigmac, sigmal, gamma):
    return c**(1-sigmac)/(1-sigmac)+gamma*((1-l)**(1-sigmal)/(1-sigmal))


def compute_c(w, r, a, z, a_new, sigmac, sigmal, gamma):
    
    def obj_fun(c):
        e = c + c**(sigmac/sigmal)*w*z*(gamma/(w*z))**(1/sigmal)-w*z-(1+r)*a+a_new
        return e
    
    c0 = ((1+r)*a+w*z-a_new)/(1+w*z*(gamma/(w*z))**(1/sigmal)) # linear case
    res = root(obj_fun, c0)
    # Debug
    # print(f"{res.status}")
    c = res.x[0]
    return c    

def compute_l(w, z, c, sigmac, sigmal, gamma):
    return np.fmax(1 - c**(sigmac/sigmal)*(gamma/(w*z))**(1/sigmal), 0)    

@jit(nopython=True)
def asset_marginal(s_probs, a_size, z_size):
    a_probs = np.zeros(a_size)
    for a_i in range(a_size):
        for z_i in range(z_size):
            a_probs[a_i] += s_probs[a_i*z_size + z_i]
    return a_probs

@jit(nopython=True)
def invariant_dist(s_probs, a_size, z_size):
    probs = np.zeros((z_size, a_size))
    for a_i in range(a_size):
        for z_i in range(z_size):
            probs[z_i, a_i] = s_probs[a_i*z_size + z_i]
    return probs

def get_equilibrium(hh, fm, r_eq):
    w_eq = fm.wage(r_eq)
    hh.set_prices(r_eq, w_eq)
    hh_ddp = DiscreteDP(hh.R, hh.Q, hh.β)
    # Solve using policy function iteration
    results = hh_ddp.solve(method='policy_iteration')
    # Compute the stationary distribution
    stationary_probs = results.mc.stationary_distributions[0]
    # Extract the invariant distribution
    inv_dist = invariant_dist(stationary_probs, hh.a_size, hh.z_size)
    # Simplify names
    z_size, a_size = hh.z_size, hh.a_size
    z_vals, a_vals = hh.z_vals, hh.a_vals
    n = a_size * z_size
    sigmac, sigmal, gamma = hh.get_hh_params()
    # Get all optimal actions across the set of a indices with z fixed in each ow
    ga = np.empty((z_size, a_size))
    gc = ga.copy()
    gl = gc.copy()
    for s_i in range(n):
        a_i = s_i // z_size
        z_i = s_i % z_size
        ga[z_i, a_i] = a_vals[results.sigma[s_i]]
        gc[z_i, a_i] = compute_c(w_eq, r_eq, a_vals[a_i], z_vals[z_i], ga[z_i, a_i],
                                sigmac, sigmal, gamma)
        gl[z_i, a_i] = compute_l(w_eq, z_vals[z_i], gc[z_i, a_i], 
                                sigmac, sigmal, gamma)
    # Capital supply
    Ks = np.sum(inv_dist*ga)
    # Labor supply
    Ns = np.sum(inv_dist*gl*z_vals[:, None])
    # Capital ratio supply
    ks = Ks/Ns
    
    return {'ga': ga, 'gc': gc, 'gl': gl, 'inv_dist': inv_dist, 
            'ks': ks, 'K': Ks, 'N': Ns, 'res': results}

# Item a)
sigmau = np.sqrt(0.621*(1-0.98**2))
# Markov chain for ln(z)!
mc = qe.markov.approximation.tauchen(0.98, sigmau)
plt.plot(np.arange(1000), mc.simulate(1000))

# Item e)
# Example prices
r0 = 0.03
# Instantiate a firm and get wage for that r value
fm = Firm(r0)
w = fm.wage(r0)
# Create an instance of Household
hh = Household(a_max=30, a_size=150, r=r0, w=w, Π=mc.P, 
               z_vals=np.exp(mc._state_values))

# Create a grid of r values at which to compute demand and supply of capital
num_points = 20
r_vals = np.linspace(0.03, 0.035, num_points)

# Compute supply of capital
k_vals = np.empty(num_points)
for i, r in enumerate(r_vals):
    k_vals[i] = get_equilibrium(hh, fm, r)['ks']
    
# Plot against demand for capital by firms
fig, ax = plt.subplots()
ax.plot(k_vals, r_vals, lw=2, alpha=0.6, label='supply of capital')
ax.plot(k_vals, fm.rd(k_vals), lw=2, alpha=0.6, label='demand for capital')
ax.grid()
ax.set_xlabel('capital')
ax.set_ylabel('interest rate')
ax.legend(loc='upper right')

# Capital ratio from firms
kd = fm.capital_ratio(r_vals)
# Excess demand
ek = kd-k_vals
# Interpolate the excess demand to find the equilibrium r
sp = splrep(r_vals, ek)
fine_r = np.linspace(0.03, 0.035, 200)
fine_ek = splev(fine_r, sp)
# Index for the minimum abs(fine_ek) will give the equilibrium r
r_eq = fine_r[np.argmin(abs(fine_ek))]
# Now we can solve the whole problem for this interest rate
equilibrium = get_equilibrium(hh, fm, r_eq)

a_vals = hh.a_vals
z_vals = hh.z_vals
z_size = hh.z_size
ga = equilibrium['ga']
gc = equilibrium['gc']
gl = equilibrium['gl']
ga_flat = ga.flatten('F') # Flatten optimal policies for MC simulations
gc_flat = gc.flatten('F')
gl_flat = gl.flatten('F')
inv_dist = equilibrium['inv_dist']
K_eq = equilibrium['K']
N_eq = equilibrium['N']
res_eq = equilibrium['res']
output_eq = fm.output(K_eq, N_eq)
w_eq = fm.wage(r_eq)

fig, ax = plt.subplots(3,1, figsize=(12, 24))
ax[0].plot(a_vals, a_vals, 'k--')  # 45 degrees
for i in range(z_size):
    lb = f'$z = {z_vals[i]:.2}$'
    ax[0].plot(a_vals, ga[i, :], lw=2, alpha=0.6, label=lb)
    ax[0].set_xlabel('current assets')
    ax[0].set_ylabel('next period assets')
    ax[1].plot(a_vals, gc[i, :], lw=2, alpha=0.6, label=lb)
    ax[1].set_xlabel('current assets')
    ax[1].set_ylabel('current consumption')
    ax[2].plot(a_vals, gl[i, :], lw=2, alpha=0.6, label=lb)
    ax[2].set_xlabel('current assets')
    ax[2].set_ylabel('current labor supply')

ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')
ax[2].legend(loc='upper left')

# Item f) Simulações
burnin = 100
Nhh = 1000 # Household cross-section
length = 1_000
simul = np.empty((length-burnin, Nhh), dtype='int32')
# Fill with Monte Carlo simulations
for i in range(Nhh):
    simul[:, i] = res_eq.mc.simulate(length)[burnin:]

sim_flat = simul.flatten('F')
# a_dist = a_vals[res_eq.sigma[simul.flatten('F')]]
a_dist = ga_flat[sim_flat]
# Plot the histogram
fig, ax = plt.subplots()
ax.hist(a_dist, bins=20, density=True, cumulative=True)
ax.set_xlabel('Wealth')
# Income
labor_i = w_eq*gl_flat[sim_flat]*z_vals[sim_flat % z_size]
cap_i = r_eq*a_dist
total_i = labor_i + cap_i
plt.hist(total_i, bins=50, density=True, cumulative=True)
plt.xlabel('Income')

# Quantiles
quantiles = [0.01, 0.05, 0.10, 0.5, 0.90, 0.95, 0.99]
a_q = [np.quantile(a_dist, q) for q in quantiles]
inc_q = [np.quantile(total_i, q) for q in quantiles]
# Gini
@jit(nopython=True)
def gini(x):
    g = np.append(0, np.cumsum(np.sort(x))/np.sum(x))
    gd = np.diff(g)
    area = np.sum((1+np.arange(len(gd)))*gd/len(gd))
    return abs(0.5 - area)/0.5

a_gini = gini(a_dist)
inc_gini = gini(total_i)

print(f"Capital-Output: {K_eq/output_eq}")
print(f"Gini Wealth: {a_gini}")
print(f"Gini Income: {inc_gini}")

# Item g)
# Bisection method. We know that with a higher value of r, the discount
# must also be higher, thus a lower value of beta is expected.
# We start in the interval 0.88 to 0.96
r1 = 0.04
fm1 = Firm(r = r1)
w1 = fm1.wage(r1)
kd1 = fm1.capital_ratio(r1)

def bisection(func, a, b, tol = 1e-8):
    if func(a)*func(b)>0:
        print('Conditions for the Intermidiate Value Theorem do not hold')
    else:
        # First middle point
        c = (a+b)/2
        error = abs(func(c))
        # ~ print(error) 
        # bisection loop
        while error > tol:
            if func(c)*func(a) < 0:
                b = c
            else:
                a = c
            c = (a+b)/2
            error = abs(func(c))
            # ~ print(error)

    return c

def beta_fun(beta):
    hh = Household(r = r1, w = w1, β = beta)
    ks = get_equilibrium(hh, fm1, r1)['ks']
    # Return the excess supply of capital
    return ks - kd1

# beta_r1 = bisection(beta_fun, 0.88, 0.96, tol=1e-4)
# Bisection is WAY TOO SLOW!!! Interpolate it
beta_vals = np.linspace(0.86, 0.96, 10)
# Excess supply of capital
es = np.array([beta_fun(b) for b in beta_vals])
# Interpolate the excess supply to find beta
sp = splrep(beta_vals, es)
fine_beta = np.linspace(0.86, 0.96, 1000)
fine_es = splev(fine_beta, sp)
# Index for the minimum abs(fine_ek) will give the equilibrium r
beta1 = fine_beta[np.argmin(abs(fine_es))]
print(f"Beta for r=4%: {beta1}")

# Item h)