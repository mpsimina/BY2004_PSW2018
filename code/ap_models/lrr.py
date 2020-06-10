import numpy as np 
from scipy import optimize
import math
from integration import gaussh
from cheby import cheby
import itertools

class BY2004:
#! make example unpacking a list of arguments before passing them
    def __init__(self,                        \
                        delta = .998,     gamma     = 10,    psi       = 1.5,    mu_c      = .0015,    \
                        rho   = .979,     phi_x     = .044,  sigma_bar = 0.0078, nu        = .987,     \
                        phi_s = .0000023, mu_d      = .0015, phi       = 3,      phi_dc    = 0,        \
                        phi_d = 4.5,      years     = 10,    ns        = 1,      mtype     = 'noSV'    ):

        self.delta, self.gamma, self.psi,       self.mu_c,     \
        self.rho,   self.phi_x, self.sigma_bar, self.nu,       \
        self.phi_s, self.mu_d,  self.phi,       self.phi_dc,   \
        self.phi_d, self.years, self.ns,        self.months,   \
        self.mtype, self.theta   = \
            delta,      gamma,      psi,            mu_c,      \
            rho,        phi_x,      sigma_bar,      nu,        \
            phi_s,      mu_d,       phi,            phi_dc,    \
            phi_d,      years,      ns,             12*years,  \
            mtype,      (1 - gamma)/( 1 - 1/psi)

        self.shock_x, self.shock_c, self.shock_d, self.shock_sig2 = [],[],[],[]
        self.x,     self.sig2,   self.cg,        self.dg       =\
            np.zeros((self.months+1, ns)), np.zeros((self.months+1, ns)),\
            np.zeros((self.months+1, ns)), np.zeros((self.months+1, ns))
                # Initial values of time series (t = 0)
        self.x[0,:], self.cg[0,:], self.dg[0,:], self.sig2[0,:] = 0, sigma_bar**(2), mu_c, mu_d
        self.xmin,   self.xmax,    self.sig2min,  self.sig2max   = np.zeros(2),np.zeros(2),np.zeros(2),np.zeros(2)

        self.avg_wc, self.k0,   self.k1,   self.A0,   self.A1,   self.A2   = [],[],[],[],[],[]
        self.avg_pd, self.k0_m, self.k1_m, self.A0_m, self.A1_m, self.A2_m = [],[],[],[],[],[]
        self.A0_rf,  self.A1_rf,self.A2_rf                                 = [],[],[]

        self.nint,   self.deg                                              = [],[]
        self.cheby_nodes_11,    self.cheby_nodes_ab                        = [],[]
        self.Tx,     self.Tsig2                                            = [],[]
        self.xt, self.sig2t                                                = [],[]
        self.xt1, self.sig2t1,  self.cgt1, self.dgt1                       = [],[],[],[]                      

        #self.grid, self.idx_grid = [],[]
        self.size_state_vars     = []

        self.xt1max, self.sig2t1max, self.xt1min, self.sig2t1min = [],[],[],[]
        self.G, self.coefs                                       = [],[]

    def generate_shocks(self, seed = 2020):

        # A function can have any number of default arguments, 
        # however if an argument is set to be a default, 
        # all the arguments to its right must always be default.
        # def demo(name, age = "30"): 
        # For example, def demo(name = “Mohan”, age): would throw an error
        # (SyntaxError: non-default argument follows default argument) 
        # because name is a default argument so all the arguments that are following it,
        # must always be default.

        # The order or arguments types is important!
        # first positional, then key
        # first arbitrary positional *args, then **kwargs
        # if calling def demo(name = '0', age = '1'): like demo(name = 'myname', 45) we get an error!

        # Unpacking parameters:
        delta, gamma, psi,       mu_c,     \
        rho,   phi_x, sigma_bar, nu,       \
        phi_s, mu_d,  phi,       phi_dc,   \
        phi_d, years, ns,        months,   \
        mtype, theta  = self.delta, self.gamma, self.psi,       self.mu_c,     \
                        self.rho,   self.phi_x, self.sigma_bar, self.nu,       \
                        self.phi_s, self.mu_d,  self.phi,       self.phi_dc,   \
                        self.phi_d, self.years, self.ns,        self.months,   \
                        self.mtype, self.theta   

        np.random.seed(seed)
        self.shock_x, self.shock_c, self.shock_d, self.shock_sig2 = \
        np.random.normal(0, 1, (months+1, ns)), np.random.normal(0, 1, (months+1, ns)), \
            np.random.normal(0, 1, (months+1, ns)), np.random.normal(0, 1, (months+1, ns))
        
    def generate_time_series(self, seed = 2020):
                # Unpacking parameters:
        delta, gamma, psi,       mu_c,     \
        rho,   phi_x, sigma_bar, nu,       \
        phi_s, mu_d,  phi,       phi_dc,   \
        phi_d, years, ns,        months,   \
        mtype, theta  = self.delta, self.gamma, self.psi,       self.mu_c,     \
                        self.rho,   self.phi_x, self.sigma_bar, self.nu,       \
                        self.phi_s, self.mu_d,  self.phi,       self.phi_dc,   \
                        self.phi_d, self.years, self.ns,        self.months,   \
                        self.mtype, self.theta 
        
        self.generate_shocks(seed)
        if mtype == 'noSV': 
            for i in range(ns):
                for t in range(1, months+1):    
                    self.x[t, i]   = rho*self.x[t-1, i] + phi_x*sigma_bar*self.shock_x[t, i]
                    self.cg[t, i]  = mu_c + self.x[t, i] + sigma_bar*self.shock_c[t, i]
                    self.dg[t, i]  = mu_d +  phi*self.x[t, i] + phi_d*sigma_bar*self.shock_d[t, i] + phi_dc*self.sig2[t-1, i]**(0.5)*self.shock_c[t, i]
                self.xmin[i], self.xmax[i]= np.min(self.x[:,i]), np.max(self.x[:,i])

        elif mtype == 'SV':
            for i in range(ns):
                for t in range(1, months+1):    
                    self.x[t, i]   = rho*self.x[t-1, i] + phi_x*self.sig2[t-1, i]**(0.5)*self.shock_x[t, i]
                    self.sig2[t, i] = max( sigma_bar**(2)*(1 - nu) + nu*self.sig2[t-1, i] + phi_s*self.shock_sig2[t-1, i], np.spacing(1))
                    self.cg[t, i]  = mu_c + self.x[t, i] + self.sig2[t, i]**(0.5)*self.shock_c[t, i]
                    self.dg[t, i]  = mu_d +  phi*self.x[t, i] + phi_d*self.sig2[t-1, i]**(0.5)*self.shock_d[t, i] + phi_dc*self.sig2[t-1, i]**(0.5)*self.shock_c[t, i] # phi_dc = 0 !
                self.xmin[i],   self.xmax[i]  = np.min(self.x[:,i]), np.max(self.x[:,i])
                self.sig2min[i], self.sig2max[i]= np.min(self.sig2[:,i]), np.max(self.sig2[:,i])
    
    def ll_avg_wc(self, z):

                    # Unpacking parameters:
        delta, gamma, psi,       mu_c,     \
        rho,   phi_x, sigma_bar, nu,       \
        phi_s, mu_d,  phi,       phi_dc,   \
        phi_d, years, ns,        months,   \
        mtype, theta  = self.delta, self.gamma, self.psi,       self.mu_c,     \
                        self.rho,   self.phi_x, self.sigma_bar, self.nu,       \
                        self.phi_s, self.mu_d,  self.phi,       self.phi_dc,   \
                        self.phi_d, self.years, self.ns,        self.months,   \
                        self.mtype, self.theta     

        if self.mtype == 'noSV':
            # mean of x = 0
            k1 = (np.exp(z) / (np.exp(z) - 1))
            k0 = (np.log( (k1 - 1)**(1 - k1) * k1**k1))
            A1 =  (1 - 1/psi)/(k1 - rho)
            A0 =  (1/(k1 - 1)) * (np.log(delta) + (1 - 1/psi)*mu_c + k0 + 0.5*theta*(1-1/psi)**2*sigma_bar**2 + 0.5*theta*A1**2*sigma_bar*phi**2)
            diff = z - A0
            return diff

        elif self.mtype == 'SV':

            k1 = (np.exp(z) / (np.exp(z) - 1))
            k0 = (np.log( (k1 - 1)**(1 - k1) * k1**k1))
            A1 =  (1 - 1/psi)/(k1 - rho)
            A2 =  0.5* theta*((1 - 1/psi)**2 + (A1 * phi_x)**2)/(k1 - nu)
            A0 =  (1/(k1 - 1)) * ( np.log(delta) + (1 - 1/psi)*mu_c + k0 + A2 * sigma_bar**2 * (1 - nu) + theta / 2*(A2 * phi_s)**(2 ) )
            diff = z - A0 - A2* sigma_bar**2
            return diff

    def ll_wc(self, wc0 = 5, start_wc = 2, end_wc = 8000, max_iter = 200):

                    # Unpacking parameters:
        delta, gamma, psi,       mu_c,     \
        rho,   phi_x, sigma_bar, nu,       \
        phi_s, mu_d,  phi,       phi_dc,   \
        phi_d, years, ns,        months,   \
        mtype, theta  = self.delta, self.gamma, self.psi,       self.mu_c,     \
                        self.rho,   self.phi_x, self.sigma_bar, self.nu,       \
                        self.phi_s, self.mu_d,  self.phi,       self.phi_dc,   \
                        self.phi_d, self.years, self.ns,        self.months,   \
                        self.mtype, self.theta  

    # Find the mean wealth-consumption ratio

    #1. Bisection*
    ## z_opt = optimize.bisect(mean_WC_LL, start_wc , end_wc, maxiter = max_iter ,args= argslistWC)

    #2. Ridder
    ## z_opt = optimize.ridder(mean_WC_LL, start_wc , end_wc, maxiter = max_iter ,args= argslistWC)

    #3. Brentq
    ## z_opt = optimize.brentq(mean_WC_LL, start_wc , end_wc, maxiter = max_iter ,args= argslistWC)

    #4. Minimize
    ##res = optimize.minimize(mean_WC_LL, start_wc, args= argslistWC, method='SLSQP',options={'maxiter': 200})
    #res = optimize.minimize(mean_WC_LL, wc0, args= argslistWC, method='BFGS',options={'maxiter': 1000})
    #z_opt = res.x

    #5. fsolve*
        self.avg_wc = optimize.fsolve(self.ll_avg_wc, wc0, xtol = 1.0000e-06)

    #6. fmin BAD
    ## z_opt = optimize.fmin(mean_WC_LL, wc0, args= argslistWC, xtol = 1.0000e-06, ftol = 1.0000e-06)
        if self.mtype == 'noSV':
            self.k1 = (np.exp(self.avg_wc)/(np.exp(self.avg_wc) - 1))
            self.k0 = (np.log( (self.k1 - 1)**(1 - self.k1)*self.k1**self.k1))
            self.A1 =  (1 - 1/self.psi)/(self.k1 - self.rho)
            self.A0 =  (1/(self.k1 - 1))*(np.log(self.delta) + \
                        (1 - 1/self.psi)*self.mu_c + self.k0 +    \
                        0.5*self.theta*(1-1/self.psi)**2*self.sigma_bar**2 +    \
                        0.5*self.theta*self.A1**2*self.sigma_bar*self.phi**2)

        elif self.mtype == 'SV':
        # find parameters required to solve the model
            self.k1 = (np.exp(self.avg_wc)/(np.exp(self.avg_wc) - 1))
            self.k0 = (np.log((self.k1 - 1)**(1 - self.k1)*self.k1**self.k1))
            self.A1 = (1 - 1/self.psi)/(self.k1 - self.rho)
            self.A2 =  0.5*self.theta*((1 - 1/self.psi)**2 + (self.A1 * self.phi_x)**2)/(self.k1 - self.nu)
            self.A0 = (1/(self.k1 - 1))*(np.log(self.delta) + (1 - 1/self.psi)*self.mu_c + \
                       self.k0 + self.A2*self.sigma_bar**2 * (1 - self.nu) + self.theta/2*(self.A2*self.phi_s)**2)

    def ll_avg_pd(self, z):
            # Unpacking parameters:
        delta, gamma, psi,       mu_c,     \
        rho,   phi_x, sigma_bar, nu,       \
        phi_s, mu_d,  phi,       phi_dc,   \
        phi_d, years, ns,        months,   \
        mtype, theta, k0,        k1,       \
        A0,    A1,    A2 = self.delta, self.gamma, self.psi,       self.mu_c,     \
                            self.rho,   self.phi_x, self.sigma_bar, self.nu,       \
                            self.phi_s, self.mu_d,  self.phi,       self.phi_dc,   \
                            self.phi_d, self.years, self.ns,        self.months,   \
                            self.mtype, self.theta, self.k0,        self.k1,       \
                            self.A0,    self.A1,    self.A2

        if self.mtype == 'noSV':

            k1_m = (np.exp(z)/(np.exp(z) + 1))
            k0_m = (-np.log( (1 - k1_m)**(1 - k1_m)*k1_m**(k1_m)))

            A1_m = (phi - 1/psi)/(1 - rho*k1_m)
            A0_m = (theta*np.log(delta) - gamma*mu_c + mu_d + 0.5*(theta - 1 - theta/psi + phi_dc)**2*sigma_bar**2 + \
                    (theta - 1)*(k0 + A0*(1 - k1)) + k0_m + 0.5*((theta - 1)*A1 + k1_m*A1_m)**2*sigma_bar**2*phi_x**2 + \
                    0.5*phi_d**2*sigma_bar**2)/(1 - k1_m)
            diff = z - A0_m

        elif self.mtype == 'SV':

            k1_m = (np.exp(z)/(np.exp(z) + 1))
            k0_m = (-np.log( (1 - k1_m)**(1 - k1_m) * k1_m**(k1_m) ) )

            A1_m = (phi - 1/psi) / (1 - rho * k1_m)
            A2_m = (0.5 * (phi_dc - gamma)**(2) + 0.5 * phi_d**(2) + (theta - 1) * A2 * (nu - k1)  + 0.5*( (theta - 1) *A1 + k1_m * A1_m )**(2) *phi_x**(2) )  / (1 - k1_m *nu)
            A0_m = (theta *np.log(delta) - gamma *mu_c + mu_d + (theta - 1) * ( k0 + A0 * (1 - k1) ) + k0_m + ( (theta - 1) * A2 + k1_m * A2_m) * sigma_bar**(2) * (1 - nu)  \
                    + 0.5*( (theta - 1) * A2 + k1_m *A2_m)**(2) *phi_s**(2) ) / (1 - k1_m)            
            
            diff = z - A0_m - A2_m * sigma_bar**(2)

        return diff

    def ll_pd(self, pd0 = 6.5 , start_pd = 2 , end_pd = 8000 , max_iter = 500):
        # Unpacking parameters:
        delta, gamma, psi,       mu_c,     \
        rho,   phi_x, sigma_bar, nu,       \
        phi_s, mu_d,  phi,       phi_dc,   \
        phi_d, years, ns,        months,   \
        mtype, theta, k0,        k1,       \
        A0,    A1,    A2 = self.delta, self.gamma, self.psi,       self.mu_c,     \
                            self.rho,   self.phi_x, self.sigma_bar, self.nu,       \
                            self.phi_s, self.mu_d,  self.phi,       self.phi_dc,   \
                            self.phi_d, self.years, self.ns,        self.months,   \
                            self.mtype, self.theta, self.k0,        self.k1,       \
                            self.A0,    self.A1,    self.A2

    # find the mean price-dividend ratio
    #z_opt = optimize.bisect(mean_PD_LL, start_search , end_search, maxiter = max_iterations, args= argslistPD)
        self.avg_pd = optimize.fsolve(self.ll_avg_pd, pd0, xtol = 1.0000e-06)
    # find parameters required to solve the model
        if self.mtype == 'noSV':

            self.k1_m = (np.exp(self.avg_pd)/(np.exp(self.avg_pd) + 1))
            self.k0_m = (-np.log((1 - self.k1_m)**(1 - self.k1_m)*self.k1_m**(self.k1_m)))
            
            self.A1_m = (self.phi - 1/self.psi) / (1 - self.rho*self.k1_m)
            self.A0_m = (self.theta*np.log(self.delta) - self.gamma*self.mu_c + self.mu_d +\
                        0.5*(self.theta - 1 - self.theta/self.psi + self.phi_dc)**2*self.sigma_bar**2 + \
                        (self.theta - 1)*(self.k0 + self.A0*(1 - self.k1)) + self.k0_m + \
                        0.5*((self.theta - 1)*self.A1 + self.k1_m*self.A1_m)**2*self.sigma_bar**2*self.phi_x**2 + \
                        0.5*self.phi_d**2*self.sigma_bar**2)/(1 - self.k1_m)

        elif self.mtype == 'SV':
            
            self.k1_m = (np.exp(self.avg_pd)/(np.exp(self.avg_pd) + 1))
            self.k0_m = (-np.log((1 - self.k1_m)**(1 - self.k1_m)*self.k1_m**(self.k1_m)))
            
            self.A1_m = (self.phi - 1/self.psi) / (1 - self.rho*self.k1_m)
            self.A2_m = (0.5 * (self.phi_dc - self.gamma)**(2) + 0.5 * self.phi_d**(2) +\
                        (self.theta - 1)*self.A2*(self.nu - self.k1)  + 0.5*((self.theta - 1)*self.A1 +\
                        self.k1_m*self.A1_m)**(2)*self.phi_x**(2))/(1 - self.k1_m*self.nu)
            self.A0_m = (self.theta *np.log(self.delta) - self.gamma*self.mu_c + \
                         self.mu_d + (self.theta - 1) * (self.k0 + self.A0*(1 - self.k1)) +\
                        self.k0_m + ((self.theta - 1)*self.A2 + \
                        self.k1_m*self.A2_m)*self.sigma_bar**(2)*(1 - self.nu)+\
                        0.5*((self.theta - 1)*self.A2 + self.k1_m*self.A2_m)**(2) *self.phi_s**(2))/(1 - self.k1_m)


    def ll_rf(self):
        
        if self.mtype == 'noSV':
            self.A1_rf = self.theta - 1 - self.theta/self.psi + (self.theta - 1)*self.A1*(self.rho - self.k1)
            self.A0_rf = self.theta*np.log(self.delta) + (self.theta - 1 - self.theta/self.psi)*self.mu_c + \
                        (self.theta - 1)*(self.A0 + self.k0 - self.k1*self.A0) + \
                        0.5*(self.theta - 1 - self.theta/self.psi)**2*self.sigma_bar**2 + \
                        0.5*((self.theta - 1)*self.A1*self.phi_x*self.sigma_bar)**2

        elif self.mtype == 'SV':
            self.A1_rf = self.theta - 1 - self.theta/self.psi + (self.theta - 1)*self.A1*(self.rho - self.k1)
            self.A2_rf = 0.5*(self.theta - 1 - (self.theta/self.psi))**2 + \
                        0.5*((self.theta - 1)*self.A1*self.phi_x)**2 + (self.theta - 1)*self.A2*(self.nu - self.k1)
            self.A0_rf = self.theta*np.log(self.delta) + (self.theta - 1 - (self.theta/self.psi))*self.mu_c + \
                         (self.theta - 1)*(self.A0 + self.A2*self.sigma_bar**(2) *(1 - self.nu) + \
                        self.k0 - self.k1*self.A0) + 0.5*((self.theta - 1)*self.A2*self.phi_s)**(2)


    def kernelstate_wc(self,coeff,i = 999,j = 999, k = 999, l = 999,m = 999,n = 999):
        
        if self.mtype == 'noSV':
            # Normalize state for next period to fit [-1,1] for the next period wc, pd approximations
            zxt1    = cheby.normalize_node(self.xt1[i,k,m,n], self.xt1min,self.xt1max)
            Txt1    = cheby.basisfunctions(zxt1, self.deg)
            #wct1    = cheby.fapprox1(self.coefs,Txt1) 
            wct1    = cheby.fapprox1(coeff,Txt1) 

            zxt     = cheby.normalize_node(self.xt[i].item(), self.min_states.item(),self.max_states.item())
            Txt     = cheby.basisfunctions(zxt, self.deg)
            #wct     = cheby.fapprox1(self.coefs,Txt) 
            wct     = cheby.fapprox1(coeff,Txt) 

            ks = 1 - self.delta**self.theta*\
                        (np.exp(wct1)*(np.exp(wct)-1)**(-1))**self.theta*\
                            np.exp(self.cgt1[i,k,m,n]*(self.theta - self.theta/self.psi))
            return ks

        elif self.mtype == 'SV':
            # Normalize states for next period to fit [-1,1] for the next period wc, pd approximations
            zxt1    = cheby.normalize_node(self.xt1[i,j,k,l,m,n], self.xt1min,self.xt1max)
            zsig2t1 = cheby.normalize_node(self.sig2t1[i,j,k,l,m,n], self.sig2t1min,self.sig2t1max)
            Txt1    = cheby.basisfunctions(zxt1, self.deg)
            Tsig2t1 = cheby.basisfunctions(zsig2t1, self.deg)
            #wct1    = cheby.fapprox2(self.coefs,Txt1,Tsig2t1) 
            wct1    = cheby.fapprox2(coeff,Txt1,Tsig2t1)

            #xt      = self.xt[i]
            #sig2t   = self.sig2t[i]
            
            zxt     = cheby.normalize_node(self.xt[i].item(), self.min_states[0].item(),self.max_states[0].item())
            zsig2t  = cheby.normalize_node(self.sig2t[i].item(), self.min_states[1].item(),self.max_states[1].item())
            Txt     = cheby.basisfunctions(zxt, self.deg)
            Tsig2t  = cheby.basisfunctions(zsig2t, self.deg)
            #wct     = cheby.fapprox2(self.coefs,Txt,Tsig2t) 
            wct     = cheby.fapprox2(coeff,Txt,Tsig2t) 

            ks = 1 - self.delta**self.theta*\
                        (np.exp(wct1)*(np.exp(wct)-1)**(-1))**self.theta*\
                            np.exp(self.cgt1[i,j,k,l,m,n]*(self.theta - self.theta/self.psi))
            return ks

    def poly_approx(self, nint = 5, deg = 6):

        self.nint = nint
        self.deg  = deg

        # Quadrature vector for one state var: nodes and weights
        self.epsi, self.wi =  gaussh.gh_nodes(nint)
        #transform the nodes to fit our distribution:
        self.epsi = 2**0.5*self.epsi

        # Uncomment if you do not want to use this class method outside this function first
        # and already generate samples
        # Generate samples from the state variables distributions
        #>>> self.generate_time_series()

        if self.mtype == 'noSV':

            self.size_state_vars = 1
            self.min_states = np.array(self.xmin[0]) # returns min for all the samples creates : ns >>> 1 elem : BY2004 model 1; 2x1 array : BY2004 model 2
            self.max_states = np.array(self.xmax[0]) # returns max for all the samples creates : ns

            self.cheby_nodes_11  = cheby.nodes_11(self.deg + 1, self.size_state_vars)
            self.cheby_nodes_ab  = cheby.nodes_ab(self.cheby_nodes_11, self.min_states, self.max_states)
            self.xt              = self.cheby_nodes_ab[0]

            self.xt1    = np.zeros((self.deg + 1,self.nint + 1,self.nint + 1,self.nint + 1))
            self.cgt1   = np.zeros((self.deg + 1,self.nint + 1,self.nint + 1,self.nint + 1))
            self.dgt1   = np.zeros((self.deg + 1,self.nint + 1,self.nint + 1,self.nint + 1))


            for i in range(self.deg + 1):
                for k in range(self.nint):
                    for m in range(self.nint):
                        for n in range(self.nint):
                            self.xt1[i,k,m,n]    = self.rho*self.xt[i] + self.phi_x*self.sigma_bar*self.epsi[k] # shock x
                            self.cgt1[i,k,m,n]   = self.mu_c + self.xt[i] + self.sigma_bar**0.5*self.epsi[m]
                            self.dgt1[i,k,m,n]   = self.mu_d + self.phi*self.xt[i] + self.phi_d*self.sigma_bar**0.5*self.epsi[n]+\
                                                    self.phi_dc*self.sigma_bar**0.5*self.epsi[m]

            self.xt1max = np.amax(self.xt1)
            self.xt1min = np.amin(self.xt1)

            self.coefs = np.ones(self.deg + 1)
            coeff      = self.coefs
            '''
            self.G = np.zeros(self.deg + 1)
            for i in range(self.deg + 1):
                self.G[i] = 0
                for k in range(self.nint):
                    for m in range(self.nint):
                        for n in range(self.nint):
                                    self.G[i] = self.G[i] + self.kernelstate_wc(self.coefs, i,[],k,[],m,n)*\
                                                    self.wi[k]*self.wi[m]*self.wi[n]
            #print(self.G)
            '''
        elif self.mtype == 'SV':

            self.size_state_vars = 2
            
            self.min_states = np.array([self.xmin[0],self.sig2min[0]], ndmin = 2) # returns min for all the samples creates : ns >>> 1 elem : BY2004 model 1; 2x1 array : BY2004 model 2
            self.min_states = self.min_states.transpose()
            self.max_states = np.array([self.xmax[0],self.sig2max[0]], ndmin = 2) # returns max for all the samples creates : ns
            self.max_states = self.max_states.transpose()

            self.cheby_nodes_11  = cheby.nodes_11(self.deg + 1, self.size_state_vars)
            self.cheby_nodes_ab  = cheby.nodes_ab(self.cheby_nodes_11, self.min_states, self.max_states)
            self.xt              = self.cheby_nodes_ab[0]
            self.sig2t           = self.cheby_nodes_ab[1]
            self.xt1    = np.zeros((self.deg + 1,self.deg + 1,self.nint + 1,self.nint + 1,self.nint + 1,self.nint + 1))
            self.sig2t1 = np.zeros((self.deg + 1,self.deg + 1,self.nint + 1,self.nint + 1,self.nint + 1,self.nint + 1))
            self.cgt1   = np.zeros((self.deg + 1,self.deg + 1,self.nint + 1,self.nint + 1,self.nint + 1,self.nint + 1))
            self.dgt1   = np.zeros((self.deg + 1,self.deg + 1,self.nint + 1,self.nint + 1,self.nint + 1,self.nint + 1))
            
            for i in range(self.deg + 1): # state x 7
                for j in range(self.deg + 1): # state sig 7
                    for k in range(self.nint): # eps x    5
                        for l in range(self.nint): # eps sig 5
                            for m in range(self.nint): # eps c  5
                                for n in range(self.nint): # eps d 5
                                    self.xt1[i,j,k,l,m,n] = self.rho*self.xt[i] + self.phi_x*self.sig2t[j]*self.epsi[k] # shock x
                                    self.sig2t1[i,j,k,l,m,n] = self.sigma_bar**2*(1-self.nu) + self.nu*self.sig2t[j] + \
                                                            self.phi_s*self.epsi[l] # shock sigma
                                    self.cgt1[i,j,k,l,m,n]  = self.mu_c + self.xt[i] + self.sig2t[j]**0.5*self.epsi[m] # shock c
                                    self.dgt1[i,j,k,l,m,n]  = self.mu_d + self.phi*self.xt[i] + self.phi_d*self.sig2t[j]**0.5*self.epsi[n] + \
                                                            self.phi_dc*self.sig2t[j]**0.5*self.epsi[m] # epsi[n] shock d
            
            self.xt1max = np.amax(self.xt1)
            self.sig2t1max = np.amax(self.sig2t1)
            self.xt1min = np.amin(self.xt1)
            self.sig2t1min = np.amin(self.sig2t1)

            self.coefs = np.ones((self.deg + 1, self.deg + 1))
            coeff      = self.coefs
            '''
            self.G = np.zeros((self.deg + 1, self.deg + 1))
            for i in range(self.deg + 1): # state x 7
                for j in range(self.deg): # state sig 7
                    self.G[i,j] = 0
                    for k in range(self.nint): # eps x    5
                        for l in range(self.nint): # eps sig 5
                            for m in range(self.nint): # eps c  5
                                for n in range(self.nint): # eps d 5
                                    
                                #    self.G[i,j] = self.G + self.kernelstate(self.xt[i],self.sig2t[i],\
                                #                                            self.epsi[k],self.epsi[l],self.epsi[m],self.epsi[n])*\
                                #                                            self.wi[k]*self.wi[l]*self.wi[m]*self.wi[n]
                                    self.G[i,j] = self.G[i,j] + self.kernelstate_wc(self.coefs, i,j,k,l,m,n)*\
                                                    self.wi[k]*self.wi[l]*self.wi[m]*self.wi[n]
            #print(self.G)
            '''
    def obj_euler(self, coeff):
        ''' to be developed...'''

        if self.mtype == 'noSV':
            self.G = np.zeros(self.deg + 1)
            for i in range(self.deg + 1):
                self.G[i] = 0
                for k in range(self.nint):
                    for m in range(self.nint):
                        for n in range(self.nint):
                            self.G[i] = self.G[i] + self.kernelstate_wc(coeff, i,[],k,[],m,n)*\
                                        self.wi[k]*self.wi[m]*self.wi[n]
                #print(self.G)

        elif self.mtype == 'SV':

            self.G = np.zeros((self.deg + 1, self.deg + 1))
            for i in range(self.deg + 1): # state x 7
                for j in range(self.deg): # state sig 7
                    self.G[i,j] = 0
                    for k in range(self.nint): # eps x    5
                        for l in range(self.nint): # eps sig 5
                            for m in range(self.nint): # eps c  5
                                for n in range(self.nint): # eps d 5
                                    
                                #    self.G[i,j] = self.G + self.kernelstate(self.xt[i],self.sig2t[i],\
                                #                                            self.epsi[k],self.epsi[l],self.epsi[m],self.epsi[n])*\
                                #                                            self.wi[k]*self.wi[l]*self.wi[m]*self.wi[n]
                                    self.G[i,j] = self.G[i,j] + self.kernelstate_wc(coeff, i,j,k,l,m,n)*\
                                                    self.wi[k]*self.wi[l]*self.wi[m]*self.wi[n]
            #print(self.G)
        self.G = self.G.flatten()
        return self.G

    def wc_coeff(self):
        ''' to be developed...'''
        coefs_0 = np.zeros((self.deg + 1, self.deg + 1))
        coefs_wc = optimize.fsolve(self.obj_euler, coefs_0, xtol = 1.0000e-06)
        print(coefs_wc)