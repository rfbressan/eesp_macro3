# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from IPython import get_ipython

# %% [markdown]
# # Macroeconomics III: Problem Set 3
# 
# Student: Rafael F. Bressan
# 
# %% [markdown]

# 1. Aiyagari Model. Time is discrete and indexed by $t=0,1,2 \ldots$ Let $\beta \in(0,1)$ be the subjective discount factor, $c_{t} \geq 0$ be consumption at period $t$ and $l_{t}$ be labor supply at $t$. Agents are ex-ante identical and have the following preferences:
# Preferences:
# $$
# E_{0}\left[\sum_{t=0}^{\infty} \beta^{t}\left(\frac{c_{t}^{1-\sigma_{c}}}{1-\sigma_{c}}+\gamma \frac{\left(1-l_{t}\right)^{1-\sigma_{l}}}{1-\sigma_{l}}\right)\right]
# $$
# where $\sigma_{c}, \sigma_{l}>1, \gamma>0 .$ Expectations are taken over an idiosyncratic shock, $z_{t}$, on labor productivity, where
# $$
# \ln \left(z_{t+1}\right)=\rho \ln \left(z_{t}\right)+\epsilon_{t+1}, \quad \rho \in[0,1]
# $$
# Variable $\epsilon_{t+1}$ is an iid shock with zero mean and variance $\sigma_{\epsilon}^{2} .$ Markets are incomplete as in Huggett (1993) and Aiyagari (1994). There are no state contingent assets and agents trade a risk-free bond, $a_{t+1}$, which pays interest rate $r_{t}$ at period $t$. In order to avoid a Ponzi game assume that $a \geq 0$.
# 
# Technology: There is no aggregate uncertainty and the technology is represented by $Y_{t}=K_{t}^{\alpha} N_{t}^{1-\alpha} .$ Let $I_{t}$ be investment at period $t .$ Capital evolves according to:
# $$
# K_{t+1}=(1-\delta) K_{t}+I_{t}
# $$
# Let $\delta=0.08, \beta=0.96, \alpha=0.4, \gamma=0.75$ and $\sigma_{c}=\sigma_{l}=2$.
# 
# (a) Use a finite approximation for the autoregressive process
# $$
# \ln \left(z^{\prime}\right)=\rho \ln (z)+\epsilon
# $$
# where $\epsilon^{\prime}$ is normal iid with zero mean and variance $\sigma_{\epsilon}^{2} .$ Use a 7 state Markov process spanning 3 standard deviations of the log wage. Let $\rho$ be equal to $0.98$ and assume that $\sigma_{z}^{2}=\frac{\sigma_{\epsilon}^{2}}{1-\rho^{2}}=0.621 .$ Simulate this shock and report results.

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
# from numpy.lib.twodim_base import eye
from scipy.stats import norm
import quantecon as qe
from quantecon.markov import DiscreteDP
from numba import jit
from scipy.optimize import root
from scipy.interpolate import splev, splrep
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rc('figure', figsize=(12, 6))

#%%
sigmau = np.sqrt(0.621*(1-0.98**2))
# Markov chain for ln(z)!
mc = qe.markov.approximation.tauchen(0.98, sigmau)
Tmc = 1000
fig, ax = plt.subplots()
ax.plot(np.arange(Tmc), mc.simulate(Tmc))
ax.set_title('Markov Chain Simulation $\ln(z_t)$')
ax.set_xlabel('time')
ax.set_ylabel('$\ln(z_t)$')


# %% [markdown]
# (b) State the households' problem.
# Household problem is to maximize expected utility over an infinity horizon, subject to a resources constraint.
#
# $$
# \begin{align*}
# \max_{c_t, l_t, a_{t+1}}&E_{0}\left[\sum_{t=0}^{\infty} \beta^{t}\left(\frac{c_{t}^{1-\sigma_{c}}}{1-\sigma_{c}}+\gamma \frac{\left(1-l_{t}\right)^{1-\sigma_{l}}}{1-\sigma_{l}}\right)\right]\\
# s.t.:\, &c_t+a_{t+1}=w_t l_t z_t +(1+r_t)a_t\\
# &a_t\geq 0, \forall t
# \end{align*}
# $$

# %% [markdown]
# (c) State the representative firm's problem.
# 
# Firms will maximize their profits taking prices as given.
# $$
# \begin{align*}
# \max_{K_t, N_t}\, &K_{t}^{\alpha} N_{t}^{1-\alpha} - w_tN_t - (r_t+\delta)K_t\\
# &K_{t+1}=(1-\delta)K_t+I_t    
# \end{align*}
# $$
 
# %% [markdown]
# (d) Define the recursive competitive equilibrium for this economy.
#
# The stationary equilibrium is a policy function $g(k, z)$, a probability distribution $\lambda(k, z)$ and positive real numbers (K, r, w) such that:
#
# i) The prices (w, r) satisfy the firm's FOC:
# $$w=\partial F(K, N)/\partial N$$
# $$r=\partial F(K, N)/\partial K -\delta$$
# ii) The policy functions $k'=g(k, z)$ and $l=h(k, z)$ solve the household's problem
#
# iii) The probability distribution $\lambda(k, z)$ is a stationary distribution associated with the policy function and the Markov process $z_t$
# $$\lambda(k',z')=\sum_z\sum_{k:k'=g(k,z)}\lambda(k,z)\mathcal{P}(z, z')$$
# iv) The average value of capital and labor implied by the average household's decision are:
# $$
# \begin{align*}
# K=\sum_{k,z}\lambda(k,z)g(k,z)\\
# N=\sum_{k,z}\lambda(k,z)h(k,z) z
# \end{align*}
# $$
# Therefore, the recursive problem of households is:
# $$
# \begin{equation}
# V(k, z_i)=\max_{k', l}\{\left[\frac{(wlz_i+(1-\delta+r)k-k')^{1-\sigma_c}}{1-\sigma_c}+\gamma\frac{(1-l)^{1-\sigma_l}}{1-\sigma_l} \right]+\beta\sum_j P_{ij}V(k', z_j)\}
# \end{equation}
# $$
#
# from which we extract the policy functions $k'=g(k, z_i)$ and $l=h(k, z_i)$ for each $z_i \in Z$. And the FOCs for the firm are:
# $$
# \begin{align*}
# w&=(1-\alpha)K^\alpha N^{-\alpha}\\
# r&=\alpha K^{\alpha-1}N^{1-\alpha} - \delta
# \end{align*}
# $$
# Which in turn define $w$ in terms of $r$ and the capital-labor ration $k:=K/N$.
# $$
# \begin{align}
# r&=\alpha k^{\alpha-1}-\delta\\
# w&=(1-\alpha)
# \end{align}
# $$
# The FOCs from the household's problem gives us optimal policies for labor supply and consumption:
# $$
# \begin{align}
# l=1-c^{\frac{\sigma_c}{\sigma_l}}\left(\frac{\gamma}{wz}\right)^{1/\sigma_l}\\
# c=wzl+(1+r)a-a'
# \end{align}
# $$
# Call those optimal policies $\tilde{l}(a, z, a')$ and $\tilde{c}(a, z, a')$. Then, inserting the policies into the household's recursive problem yields a one-dimensional maximization in $a'$.
#


# %% [markdown]

# (e) Write down a code to solve this problem. Find the policy functions for $a^{\prime}, c$, and $l$.
# %%
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

# %%
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
    
# %%
# Plot against demand for capital by firms
fig, ax = plt.subplots()
ax.plot(k_vals, r_vals, lw=2, alpha=0.6, label='supply of capital')
ax.plot(k_vals, fm.rd(k_vals), lw=2, alpha=0.6, label='demand for capital')
ax.grid()
ax.set_xlabel('capital')
ax.set_ylabel('interest rate')
ax.legend(loc='upper right')

# %%
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

# %%
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
# %% [markdown]

# (f) Solve out for the equilibrium allocations and compute statistics for this economy. Report basic statistics about this economy, such as: investment rate, the capitalto-output ratio, cumulative distribution of income (e.g., bottom $1 \%, 5 \%, 10 \%$, $50 \%$, top $1 \%$, top $5 \%$, top $10 \%$ ), cumulative distribution of wealth (e.g., bottom $1 \%, 5 \%, 10 \%, 50 \%$, top $1 \%$, top $5 \%$, top $10 \%$ ), Gini of income and Gini of Wealth.
# %% [markdown]
# From the invariant distribution we get the distributions of economic variables in the stationary equilibrium.
# $$
# \begin{align*}
# I&=\sum_{z}\lambda(z,a)gl(z,a)wz + \sum_{z}\lambda(z,a)ga(z,a)r
# \end{align*}
# $$
# %%
# Item f) Simulations
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
# Income
labor_i = w_eq*gl_flat[sim_flat]*z_vals[sim_flat % z_size]
cap_i = r_eq*a_dist
total_i = labor_i + cap_i
# %%
# Plot the histogram
fig, ax = plt.subplots()
ax.hist(a_dist, bins=20, density=True, cumulative=True)
ax.set_xlabel('Wealth')
# %%
plt.hist(total_i, bins=50, density=True, cumulative=True)
plt.xlabel('Income')

# %%
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
# %% [markdown]

# (g) Could you find a value for $\beta$ such the equilibrium interest rate is equal to $4 \%$ per year? Report this value and explain how you found it
# %% [markdown]
# For a given r, the demand for capital is fixed. We only need to create a grid of $\beta$ values and solve the household problem in order to find the amount of capital supplied. Those values must match in equilibrium. We have two options: the bisection method the search for a $\beta$ that equates demand and supply, or interpolate over a gross grid of $\beta$ values and find the one (on a finer grid) that matches supply.

# %%
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

# %%
# beta_r1 = bisection(beta_fun, 0.88, 0.96, tol=1e-4)
# %% [markdown]

# **Bisection is WAY TOO SLOW!!! Interpolate it.**

# %%
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

# %% [markdown]

# (h) Could you find a value for $\sigma_{z}^{2}$ such the equilibrium wealth Gini in $0.8$ ? Report this value and explain how you found it,

# %% [markdown]

# 2. Hopenhayn model. On paying a fixed operating cost $\kappa>0$, an incumbent firm that hires $n$ workers produces flow output $y=z n^{\alpha}$ with $0<\alpha<1$ where $z>0$ is a firm-level productivity level. The productivity of an incumbent firm evolves according to an $\mathrm{AR}(1)$ in $\operatorname{logs}$
# $$
# \ln \left(z_{t+1}\right)=(1-\rho) \ln (\bar{z})+\rho \ln \left(z_{t}\right)+\sigma \epsilon_{t+1}, \rho \in(0,1), \sigma>0
# $$
# where $\epsilon_{t+1} \sim N(0,1)$. Firms discount flow profits according to a constant discount factor $0<\beta<1$. There is an unlimited number of potential entrants. On paying a sunk entry cost $\kappa_{e}>0$, an entrant receives an initial productivity draw $z_{0}>0$ and then starts operating the next period as an incumbent firm. For simplicity, assume that initial productivity $z_{0}$ is drawn from the stationary productivity distribution implied by the AR(1) above.
# 
# Individual firms take the price $p$ of their output as given. Industry-wide demand is given by the $D(p)=\bar{D} / p$ for some constant $\bar{D}>0 .$ Let labor be the numeraire, so that the wage is $w=1 .$ Let $\pi(z)$ and $v(z)$ denote respectively the profit function and value function of a firm with productivity $z .$ Let $v_{e}$ denote the corresponding expected value of an entering firm. Let $\mu(z)$ denote the (stationary) distribution of firms and let $m$ denote the associated measure of entering firms.
# (a) Derive an expression for the profit function.
# 
# (b) Set the parameter values $\alpha=2 / 3, \beta=0.8, \kappa=20, \kappa_{e}=40, \ln (\bar{z})=1.4$, $\sigma=0.20, \rho=0.9$ and $\bar{D}=100 .$ Discretize the AR(1) process to a Markov chain on 33 nodes. Solve the model on this grid of productivity levels. Calculate the equilibrium price $p^{*}$ and measure of entrants $m^{*}$. Let $z^{*}$ denote the cutoff level of productivity below which a firm exits. Calculate the equilibrium $z^{*}$. Plot the stationary distribution of firms and the implied distribution of employment across firms. Explain how these compare to the stationary distribution of productivity levels implies by the AR(1).
# (c) Now suppose the demand curve shifts, with $\bar{D}$ increasing to $120 .$ How does this change the equilibrium price and measure of entrants? How does this change the stationary distributions of firms and employment? Give intuition for your results.

# %% [markdown]

# 3. Ramsey model in continuous time. Consider the decentralised Ramsey model in continuous time. Households solve
# $$
# \max _{c, l} \int_{0}^{\infty} e^{-\rho t} u(c, l) d t
# $$
# subject to
# $$
# \dot{a}=w(1-l)+r a-c
# $$
# where $c$ is consumption, $a$ denotes assets and $l$ is leisure. $\rho$ is the subjective discount rate, $r$ is the interest rate and $w$ is the wage rate. Firms rent capital and labour from households to maximise
# $$
# \max A K^{\alpha} N^{1-\alpha}-w N-(r+\delta) K
# $$
# where $\delta$ is the depreciation rate and $A$ is a productivity factor.
# 
# (a) Write down the HJB associated with the households problem. Explain the steps to derive it.
# 
# **From now on, assume that $u(c, l)=\log (c)+\eta \log (l)$. You can assume that $\rho=0.04$ and $\eta=0.75$**
# 
# (b) For a given interest rate $r$ and wage rate $w$, write down a code to solve the households problem.
# 
# (c) Write down the market clearing conditions.
# Assume that $\delta=0.06, A=1$ and $\alpha=0.33$.
# 
# (d) Write down the equations that describe the steady-state of the system and solve for the steady-state level of capital and labour supply.
# 
# (e) Write down a code to solve out the whole transition. Then simulate a permanent change in the TFP factor, such that $A$ increases from $A=1$ to $A=1.2$. Plot the evolution of capital, labour and consumption.