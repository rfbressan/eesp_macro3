"""
Author: Jacob Hess 
Date: January 2021
Written in python 3.8 on Spyder IDE.
Description: Finds the stationary equilibrium in a production economy with incomplete markets and idiosyncratic income
risk as in Aiyagari (1994). Features of the algorithm are: 
    1) value function iteration to solve the household problem
    2) discrete approximation up to 7 states of a continuous AR(1) income process using the Rouwenhorst method 
    (Tauchen is also an option in the code)
    3) approximation of the stationary distribution using a monte carlo simulation
Aknowledgements: I used notes or pieces of code from the following :
    1) Gianluca Violante's notes (https://sites.google.com/a/nyu.edu/glviolante/teaching/quantmacro)
    2) Fabio Stohler (https://github.com/Fabio-Stohler)
    3) Jeppe Druedahl (https://github.com/JeppeDruedahl) and NumEconCopenhagen (https://github.com/NumEconCopenhagen)
    
Required packages: 
    -- Packages from the anaconda distribution. (to install for free: https://www.anaconda.com/products/individual)
    -- QuantEcon (to install: 'conda install quantecon')
    -- Interpolation from EconForge
       * optimized interpolation routines for python/numba
       * to install 'conda install -c conda-forge interpolation'
       * https://github.com/EconForge/interpolation.py
Note: If simulation tells you to increase grid size, increase self.a_max in function setup_parameters.
"""


import time
import numpy as np
from numba import njit, prange
import quantecon as qe
from scipy.stats import rv_discrete
from interpolation import interp
import matplotlib.pyplot as plt
import seaborn as sns




#############
# I. Model  #
############

class AiyagariVFI:
    
    """
    Class object of the model. AiyagariVFI().solve_model() runs everything
    """

    ############
    # 1. setup #
    ############

    def __init__(self, 
                rho_z = 0.98,          #autocorrelation coefficient
                sigma_u = 0.1568171,   #std. dev. of shocks at annual frequency
                Nz = 7,                #number of discrete income states
                z_bar = 0,             #constant term in continuous income process (not the mean of the process)
                a_bar = 0,             #select borrowing limit
                plott =1,              #select 1 to make plots
                plot_supply_demand = 1, # select 1 for capital market supply/demand graph
                # Household and Firm parameters
                beta = 0.96,
                gamma = 0.75,
                sigmac =2,               #crra coefficient
                sigmal = 2,
                delta = 0.08,
                alpha = 0.4
                       ):
        
        #parameters subject to changes
        self.a_bar = a_bar
        self.plott, self.plot_supply_demand = plott, plot_supply_demand
        self.rho_z, self.sigma_u, self.Nz = rho_z, sigma_u, Nz
        self.z_bar = z_bar 
        self.beta = beta  # discount factor
        self.gamma = gamma
        self.sigmac, self.sigmal = sigmac, sigmal
        self.delta = delta  # depreciation rate
        self.alpha = alpha  # cobb-douglas coeffient
        
        self.setup_parameters()
        self.setup_grid()
        self.setup_discretization()
        

    def setup_parameters(self):

        # a. model parameters
        self.rho = (1-self.beta)/self.beta #discount rate

        # b. hh solution
        self.tol_hh = 1e-6  # tolerance for consumption function iterations
        self.maxit_hh = 2000  # maximum number of iterations when finding consumption function in hh problem
       
        # asset grid 
        self.Na = 1000
        self.a_min = self.a_bar
        self.a_max = 80
        self.curv = 3 

        # c. simulation
        self.seed = 123
        self.ss_a0 = 1.0  # initial cash-on-hand (homogenous)
        self.ss_simN = 50_000  # number of households
        self.ss_simT = 2000  # number of time-periods
        self.ss_sim_burnin = 1000  # burn-in periods before calculating average savings

        # d. steady state solution
        self.ss_ret_tol = 1e-4  # tolerance for finding interest rate
        self.dp_big = 1/10      # dampening parameter to update new interest rate guess 
        self.dp_small = 1/100    # dampening parameter to prevent divergence
        self.maxit = 100    # maximum iterations steady state solution
        
        # e. complete markets solution
        self.ret_cm = 1/self.beta - 1
        self.k_cm = self.k_demand(self.ret_cm)


    def setup_grid(self):
        # a. asset grid
        self.grid_a = self.make_grid(self.a_min, self.a_max, self.Na, self.curv)  #savings grid
        self.grid_l = np.linspace(0, 0.99, 100)

        
    def setup_discretization(self):
        
        # a. discretely approximate the continuous income process 
        # self.mc = qe.markov.approximation.rouwenhorst(self.Nz, self.z_bar, self.sigma_u, self.rho_z)
        self.mc = qe.markov.approximation.tauchen(self.rho_z, self.sigma_u, self.z_bar, 3, self.Nz)

        # b. transition matrix and states
        self.pi = self.mc.P
        self.grid_z = np.exp(self.mc.state_values)
        
        # c. initial distribution of z
        z_diag = np.diag(self.pi ** 1000)
        self.ini_p_z = z_diag / np.sum(z_diag)
        
        # d. idiosyncratic shock simulation for each household
        self.shock_matrix= np.zeros((self.ss_simT, self.ss_simN))
            
        # initial z shock drawn from initial distribution
        random_z = rv_discrete(values=(np.arange(self.Nz),self.ini_p_z),seed=self.seed)
        self.z0_idx = random_z.rvs(size=self.ss_simN)   #returns shock index, not grid value 
        
        # idiosyncratic income shock index realizations for all individuals
        for n in range(self.ss_simN) :
            self.shock_matrix[:,n] = self.mc.simulate_indices(self.ss_simT, init=self.z0_idx[n])
    
    #######################
    # 2. helper functions #
    ######################
    
    def make_grid(self, min_val, max_val, num, curv):  
        """
        Makes an exponential grid of degree curv. 
        
        A higher curv will put more points closer a_min. 
        
        Equivalenty, np.linspace(min_val**(1/curv), max_val**(1/curv), num)**curv will make
        the exact same grid that this function does.
        """
        grd = np.zeros(num)
        scale=max_val-min_val
        grd[0] = min_val
        grd[num-1] = max_val
        for i in range(1,num-1):
            grd[i] = min_val + scale*((i)/(num - 1)) ** curv
        
        return grd
    
    
    ##############
    # household #
    #############
    
    def u(c, l, sigmac, sigmal, gamma):
        eps = 1e-8
        # if  sigmac == 1:
        #     return np.log(np.fmax(c, eps))
        # else:
        #     return (np.fmax(c, eps) ** (1 - sigmac) -1) / (1 - sigmac)
        caux = (np.fmax(c, eps)**(1-sigmac))/(1-sigmac)
        laux = (gamma*(1-np.fmax(l, eps))**(1-sigmal))/(1-sigmal)
        return caux + laux

    def u_prime(self, c) :
        eps = 1e-8
        return np.fmax(c, eps) ** (-self.sigmac)

    
    def u_prime_inv(self, x):    
        eps = 1e-8 
        return np.fmax(x, eps) ** (-1/self.sigmac)
    
    #########
    # firm #
    ########
    
    def  f(self,k) :
        eps = 1e-8
        return np.fmax(k, eps) ** self.alpha
    
    def f_prime(self,k):
        eps = 1e-8
        return self.alpha * np.fmax(k, eps) ** (self.alpha - 1)
    
    def f_prime_inv(self,x):
        eps = 1e-8
        return (np.fmax(x, eps) / self.alpha) ** ( 1 / (self.alpha - 1) )
    
    def ret_func(self, k):
        return  self.f_prime(k) - self.delta

    def w_func(self, ret):
        k = self.f_prime_inv(ret + self.delta)
        return self.f(k) - self.f_prime(k) * k
    
    def k_demand(self,ret):
        return (self.alpha/(ret+self.delta))**(1/(1-self.alpha))
    
    #############################
    # 3. stationary equilibrium #
    #############################
    
    def graph_supply_demand(self,ret_vec):
        
        """
        Returns capital market supply and demand to plot. Note that for some extreme values of ret_vec 
        the monte carlo simulation might tell you to increase the grid size. You can ignore this here.
        
        *Input
            - ret_vec : vector of interest rates
        *Output
            - k_demand : capital demand as a function of the interest rate
            - k_supply : capital supply as a function of the interest rate
        """
        
        #1. initialize
        k_demand = np.empty(ret_vec.size)
        k_supply = np.empty(ret_vec.size)
        
        for idx, ret_graph in enumerate(ret_vec):
            
            # 2. capital demand
            k_demand[idx] = self.k_demand(ret_graph)
            
            # 3. capital supply
            w_graph = self.w_func(ret_graph)
            
            # i. Household problem
            VF_graph, pol_sav_graph, pol_cons_graph, pol_lab_graph, it_hh_graph = solve_hh(ret_graph, self.Nz, self.Na, self.tol_hh, self.maxit_hh, 
                      self.grid_a, w_graph, self.grid_z, self.grid_l, self.sigmac, self.sigmal, self.gamma, self.beta, self.pi)
            
            # ii. Simulation
            a0 = self.ss_a0 * np.ones(self.ss_simN)
            z0 = self.grid_z[self.z0_idx]
            sim_ret_graph = ret_graph * np.ones(self.ss_simT)
            sim_w_graph = w_graph * np.ones(self.ss_simT)
            
            sim_k_graph, a, z, c, m = simulate_MonteCarlo(
                a0,
                z0,
                sim_ret_graph,
                sim_w_graph,
                self.ss_simN,
                self.ss_simT,
                self.grid_z,
                self.grid_a,
                pol_cons_graph,
                pol_sav_graph,
                self.pi,
                self.shock_matrix,
            )
            
            k_supply[idx] = np.mean(sim_k_graph[self.ss_sim_burnin:])
            
        plt.plot(k_demand,ret_vec)
        plt.plot(k_supply,ret_vec)
        plt.plot(k_supply,np.ones(ret_vec.size)*self.rho,'--')
        plt.title('Capital Market')
        plt.legend(['Demand','Supply','Supply in CM'])
        plt.xlabel('Capital')
        plt.ylabel('Interest Rate')
        plt.savefig('capital_supply_demand_aiyagari.pdf')
        plt.show()

        return k_demand, k_supply
            

    def ge_algorithm(self, ret_ss_guess, a0, z0, t1):
        
        """
        General equilibrium solution algorithm.
        """
        
        #given ret_ss_guess as the guess for the interest rate (step 1)
        
        # a. obtain prices from firm FOCs (step 2)
        self.ret_ss = ret_ss_guess
        self.w_ss = self.w_func(self.ret_ss)

        # b. solve the HH problem (step 3)
        
        print('\nSolving household problem...')
        
        self.VF, self.pol_sav, self.pol_cons, self.pol_lab, self.it_hh = solve_hh(self.ret_ss, self.Nz, self.Na, self.tol_hh, self.maxit_hh, self.grid_a, self.w_ss, self.grid_l, self.grid_z, self.sigmac, self.sigmal, self.gamma, self.beta, self.pi)
        
        
        if self.it_hh > self.maxit_hh:
            raise Exception('No value function convergence')
        else : 
            print(f"Value function convergence in {self.it_hh} iterations.")

            
        t2 = time.time()
        print(f'Household problem time elapsed: {t2-t1:.2f} seconds')

        # c. simulate (step 4)
        
        print('\nSimulating...')
        
        # prices
        self.ss_sim_ret = self.ret_ss * np.ones(self.ss_simT)
        self.ss_sim_w = self.w_ss * np.ones(self.ss_simT)
        
        self.ss_sim_k, self.ss_sim_a, self.ss_sim_z, self.ss_sim_c, self.ss_sim_m = simulate_MonteCarlo(
            a0,
            z0,
            self.ss_sim_ret,
            self.ss_sim_w,
            self.ss_simN,
            self.ss_simT,
            self.grid_z,
            self.grid_a,
            self.pol_cons,
            self.pol_sav,
            self.pi,
            self.shock_matrix
        )
        
        t3 = time.time()
        print(f'Simulation time elapsed: {t3-t2:.2f} seconds')

        # d. calculate difference
        self.k_ss = np.mean(self.ss_sim_k[self.ss_sim_burnin :])
        ret_ss_new = self.ret_func(self.k_ss)
        diff = ret_ss_guess - ret_ss_new
        
        return diff

    #####################
    # 4. Main Function #
    ####################

    def solve_model(self):
    
            """
            Finds the stationary equilibrium
            """    
            
            t0 = time.time()    #start the clock
    
            # a. initial values for agents
            a0 = self.ss_a0 * np.ones(self.ss_simN)
            z0 = self.grid_z[self.z0_idx]
    
            # b. initial interest rate guess (step 1)
            ret_guess = 0.03       
            
            # We need (1+r)beta < 1 for convergence.
            assert (1 + ret_guess) * self.beta < 1, "Stability condition violated."
            
            # c. iteration to find equilibrium interest rate ret_ss
            
            for it in range(self.maxit) :
                t1=time.time()
                
                print("\n-----------------------------------------")
                print("Iteration #"+str(it+1))
                
                diff_old=np.inf
                diff = self.ge_algorithm(ret_guess, a0, z0, t1)
                
                if abs(diff) < self.ss_ret_tol :
                    print("\n-----------------------------------------")
                    print('\nConvergence!')
                    self.ret_ss = ret_guess
                    break
                else :
                    #adaptive dampening 
                    if np.abs(diff) > np.abs(diff_old):
                        ret_guess = ret_guess - self.dp_small*diff  #to prevent divergence force a conservative new guess
                    else:
                        ret_guess = ret_guess - self.dp_big*diff
                    
                    print(f"\nNew interest rate guess = {ret_guess:.5f} \t diff = {diff:8.5f}")
                    diff_old=diff
            
            if it > self.maxit :
                print("No convergence")
                
            #calculate precautionary savings rate
            self.precaution_save = self.ret_cm - self.ret_ss
            
            t4 = time.time()
            print('Total iteration time elapsed: '+str(time.strftime("%M:%S",time.gmtime(t4-t0))))
            
            # d. plot
        
            if self.plott:
                
                print('\nPlotting...')
            
                ##### Policy Functions #####
                for iz in range(self.Nz):
                    plt.plot(self.grid_a, self.VF[iz,:])
                plt.title('Value Function')
                plt.xlabel('Assets')
                plt.savefig('value_function_vfi_aiyagari.pdf')
                plt.show()
                
                for iz in range(self.Nz):
                    plt.plot(self.grid_a, self.pol_sav[iz,:])
                plt.title("Savings Policy Function")
                plt.xlabel('Assets')
                plt.savefig('savings_policyfunction_vfi_aiyagari.pdf')
                plt.show()
                
                for iz in range(self.Nz):
                    plt.plot(self.grid_a, self.pol_cons[iz,:])
                plt.title("Consumption Policy Function")
                plt.xlabel('Assets')
                plt.savefig('consumption_policyfunction_vfi_aiyagari.pdf')
                plt.show()
                
                
                ##### Asset Distribution ####
                sns.histplot(self.ss_sim_a,  bins=100, stat='density')
                plt.xlabel('Assets')
                plt.title('Wealth Distribution')
                plt.savefig('wealth_distrib_vfi_aiyagari.pdf')
                plt.show()
                
            if self.plot_supply_demand:
                print('\nPlotting supply and demand...')
                
                self.ret_vec = np.linspace(-0.01,self.rho-0.001,8)
                self.k_demand, self.k_supply = self.graph_supply_demand(self.ret_vec)
         
            t5 = time.time()
            print(f'Plot time elapsed: {t5-t4:.2f} seconds')
            
            print("\n-----------------------------------------")
            print("Stationary Equilibrium Solution")
            print("-----------------------------------------")
            print(f"Steady State Interest Rate = {ret_guess:.5f}")
            print(f"Steady State Capital = {self.k_ss:.2f}")
            print(f"Precautionary Savings Rate = {self.precaution_save:.5f}")
            print(f"Capital stock in incomplete markets is {((self.k_ss - self.k_cm)/self.k_cm)*100:.2f} percent higher than with complete markets")
            print('\nTotal run time: '+str(time.strftime("%M:%S",time.gmtime(t5-t0))))
            
#########################
# II. Jitted Functions  #
########################

################################
# 1. Helper Functions  #
###############################

@njit
def u(c, l, sigmac, sigmal, gamma):
    eps = 1e-8
    # if  sigmac == 1:
    #     return np.log(np.fmax(c, eps))
    # else:
    #     return (np.fmax(c, eps) ** (1 - sigmac) -1) / (1 - sigmac)
    caux = (np.fmax(c, eps)**(1-sigmac))/(1-sigmac)
    laux = (gamma*(1-np.fmax(l, eps))**(1-sigmal))/(1-sigmal)
    return caux + laux

###############################################
# 2. Household and Value Function Iteration  #
##############################################


# @njit(parallel=True)
def solve_hh(ret, Nz, Na, tol, maxit, grid_a, w, grid_l, grid_z, sigmac, sigmal, gamma, beta, pi):
   
    """
    Solves the household problem.
    
    *Output
        * VF is value function
        * pol_sav is the a' (savings) policy function
        * pol_cons is the consumption policy function
        * it_hh is the iteration number 
    """

    # a. Initialize counters, initial guess. storage matriecs
    dist = np.inf
    
    VF_old    = np.zeros((Nz,Na))  #initial guess
    VF = np.copy(VF_old)
    pol_sav = np.copy(VF_old)
    pol_cons = np.copy(VF_old)
    pol_lab = np.copy(VF_old)
    indk = np.zeros((Nz, Na), dtype='int32')
    indl = np.copy(indk)
    Tvl = np.zeros((grid_l.size, Na))
    
    # b. Iterate
    for it_hh in range(maxit) :
        for iz in range(Nz):
            for ia in prange(Na):
                for il in range(grid_l.size):
                    l = grid_l[il]
                    # Consumption for every future assets
                    c = (1+ret)*grid_a[ia] + w*l*grid_z[iz] - grid_a
                    util = u(c, l, sigmac, sigmal, gamma)
                    util[c < 0] = -10e9
                    Tvl[il,:] = util + beta*(np.dot(pi[iz,:], VF_old))
                # Maximum over columns and rows
                maxc=np.max(Tvl, axis=1)
                maxr=np.max(Tvl, axis=0)
                indl[iz, ia] = np.argmax(maxc)
                VF[iz,ia] = Tvl[np.argmax(maxc), np.argmax(maxr)]
                indk[iz,ia] = np.argmax(maxr)
                pol_sav[iz,ia] = grid_a[indk[iz, ia]]
            
            # obtain consumption and labor policy function
            pol_cons[iz,:] = (1+ret)*grid_a + w*grid_z[iz]*grid_l[indl[iz,:]] - pol_sav[iz,:]
            pol_lab[iz,:] = grid_l[indl[iz,:]]
            
        dist = np.linalg.norm(VF-VF_old)
            
        if dist < tol :
            break
            
        VF_old = np.copy(VF)

    
    return VF, pol_sav, pol_cons, pol_lab, it_hh



####################
# 3. Simulation   #
##################

@njit(parallel=True)
def simulate_MonteCarlo( 
    a0,
    z0,
    sim_ret,
    sim_w,
    simN,
    simT,
    grid_z,
    grid_a,
    pol_cons,
    pol_sav,
    pi,
    shock_matrix
        ):
    
    """
    Monte Carlo simulation for T periods for N households. Also checks 
    the grid size by ensuring that no more than 1% of households are at
    the maximum value of the grid.
    
    *Output
        - sim_k : aggregate capital (total savings in previous period)
        - sim_sav: current savings (a') profile
        - sim_z: income profile 
        - sim_c: consumption profile
        - sim_m: cash-on-hand profile ((1+r)a + w*z)
    """
    
    
    
    # 1. initialization
    sim_sav = np.zeros(simN)
    sim_c = np.zeros(simN)
    sim_m = np.zeros(simN)
    sim_z = np.zeros(simN, np.float64)
    sim_z_idx = np.zeros(simN, np.int32)
    sim_k = np.zeros(simT)
    edge = 0
    
    # 2. savings policy function interpolant
    polsav_interp = lambda a, z: interp(grid_a, pol_sav[z, :], a)
    
    # 3. simulate markov chain
    for t in range(simT):   #time
    
        #calculate cross-sectional moments
        if t <= 0:
            sim_k[t] = np.mean(a0)
        else:
            sim_k[t] = np.mean(sim_sav)
        
        for i in prange(simN):  #individual

            # a. states 
            if t == 0:
                a_lag = a0[i]
            else:
                a_lag = sim_sav[i]
                
            # b. shock realization 
            sim_z_idx[i] = shock_matrix[t,i]
            sim_z[i] = grid_z[sim_z_idx[i]]
                
            # c. income
            y = sim_w[t]*sim_z[i]
            
            # d. cash-on-hand
            sim_m[i] = (1 + sim_ret[t]) * a_lag + y
            
            # e. consumption path
            sim_c[i] = sim_m[i] - polsav_interp(a_lag,sim_z_idx[i])
            
            if sim_c[i] == pol_cons[sim_z_idx[i],-1]:
                edge = edge + 1
            
            # f. savings path
            sim_sav[i] = sim_m[i] - sim_c[i]
            
    # 4. grid size evaluation
    frac_outside = edge/grid_a.size
    if frac_outside > 0.01 :
        print('\nIncrease the number of asset grid points (a_max)!')

    return sim_k, sim_sav, sim_z, sim_c, sim_m



#run everything

ge_vfi = AiyagariVFI()
ge_vfi.solve_model()