import numpy as np
from scipy import optimize

def mean_WC_LL(z, *argslistWC):
    
    x = z
  
    delta, gamma, psi, mu_c, rho, phi_x, sigma_bar, nu, phi_s, theta = argslistWC

    k1 = ( np.exp(x) / (np.exp(x) - 1) )
    k0 = ( np.log( (k1 - 1)**(1 - k1) * k1**k1) )

    A1 =  (1 - 1/psi)/(k1 - rho)
    A2 =  0.5* theta*( (1 - 1/psi)**2 + (A1 * phi_x)**2) / (k1 - nu)
    A0 =  (1/(k1 - 1)) * ( np.log(delta) + (1 - 1/psi)*mu_c + k0 + A2 * sigma_bar**2 * (1 - nu) + theta / 2*(A2 * phi_s)**(2 )  )
        
    diff = x - A0 - A2* sigma_bar**2

    return diff

def WC_LL(mean_wc_search, start_search, end_search, param, max_iterations):

    argslistWC = param.delta, param.gamma, param.psi, param.mu_c, param.rho, \
        param.phi_x, param.sigma_bar, param.nu, param.phi_s, param.theta

    # Find the mean price-dividend ratio

    #1. Bisection*
    ## z_opt = optimize.bisect(mean_WC_LL, start_search , end_search, maxiter = max_iterations ,args= argslistWC)

    #2. Ridder
    ## z_opt = optimize.ridder(mean_WC_LL, start_search , end_search, maxiter = max_iterations ,args= argslistWC)

    #3. Brentq
    ## z_opt = optimize.brentq(mean_WC_LL, start_search , end_search, maxiter = max_iterations ,args= argslistWC)

    #4. Minimize
    ##res = optimize.minimize(mean_WC_LL, start_search, args= argslistWC, method='SLSQP',options={'maxiter': 200})
    #res = optimize.minimize(mean_WC_LL, mean_wc_search, args= argslistWC, method='BFGS',options={'maxiter': 1000})
    #z_opt = res.x

    #5. fsolve*
    z_opt = optimize.fsolve(mean_WC_LL, mean_wc_search, args = argslistWC, xtol = 1.0000e-06)

    #6. fmin BAD
    ## z_opt = optimize.fmin(mean_WC_LL, mean_wc_search, args= argslistWC, xtol = 1.0000e-06, ftol = 1.0000e-06)
    
    # find parameters required to solve the model
    k1 = ( np.exp(z_opt) / (np.exp(z_opt) - 1) )
    k0 = ( np.log( (k1 - 1)**(1 - k1) * k1**k1) )

    A1 =  (1 - 1/param.psi)/(k1 - param.rho)
    A2 =  0.5* param.theta*( (1 - 1/param.psi)**2 + (A1 * param.phi_x)**2) / (k1 - param.nu)
    A0 =  (1/(k1 - 1)) * ( np.log(param.delta) + (1 - 1/param.psi)*param.mu_c + \
        k0 + A2 * param.sigma_bar**2 * (1 - param.nu) + param.theta / 2*(A2 * param.phi_s)**(2 )  )

    return k1, k0, A1, A2, A0

def mean_PD_LL(z, *argslistPD):
    x = z

    delta, gamma, psi, mu_c, rho, phi_x, sigma_bar, nu, phi_s, mu_d, phi, phi_dc, phi_d, theta, k0,  k1, A0, A1, A2 = argslistPD 

    k1_m = (np.exp(x) / (np.exp(x) + 1))
    k0_m = (-np.log( (1 - k1_m)**(1 - k1_m) * k1_m**(k1_m) ) )

    A1_m = (phi - 1/psi) / (1 - rho * k1_m)
    A2_m = (0.5 * (phi_dc - gamma)**(2) + 0.5 * phi_d**(2) + (theta - 1) * A2 * (nu - k1)  + 0.5*( (theta - 1) *A1 + k1_m * A1_m )**(2) *phi_x**(2) )  / (1 - k1_m *nu)
    A0_m = (theta *np.log(delta) - gamma *mu_c + mu_d + (theta - 1) * ( k0 + A0 * (1 - k1) ) + k0_m + ( (theta - 1) * A2 + k1_m * A2_m) * sigma_bar**(2) * (1 - nu)  \
            + 0.5*( (theta - 1) * A2 + k1_m *A2_m)**(2) *phi_s**(2) ) / (1 - k1_m)

    diff = x - A0_m - A2_m * sigma_bar**(2)
        
    return diff

def PD_LL(mean_pd_search, start_search, end_search , param, paramResults , max_iterations):


    k0 = paramResults.k0
    k1 = paramResults.k1
    A0 = paramResults.A0
    A1 = paramResults.A1
    A2 = paramResults.A2

    #theta = (1 - param.gamma)/( 1 - 1/param.psi)

    argslistPD = param.delta, param.gamma, param.psi, param.mu_c, param.rho, param.phi_x, param.sigma_bar, param.nu, param.phi_s, param.mu_d, param.phi, param.phi_dc, param.phi_d, param.theta, k0,  k1, A0, A1, A2

    # find the mean price-dividend ratio
    #z_opt = optimize.bisect(mean_PD_LL, start_search , end_search, maxiter = max_iterations, args= argslistPD)
    z_opt = optimize.fsolve(mean_PD_LL, mean_pd_search, args = argslistPD, xtol = 1.0000e-06)
    # find parameters required to solve the model
    k1_m = (np.exp(z_opt) / (np.exp(z_opt) + 1))
    k0_m = (-np.log( (1 - k1_m)**(1 - k1_m) * k1_m**(k1_m) ) )

    A1_m = (param.phi - 1/param.psi) / (1 - param.rho * k1_m)
    A2_m = (0.5 * (param.phi_dc - param.gamma)**(2) + 0.5 * param.phi_d**(2) + (param.theta - 1) * A2 * (param.nu - k1)  + 0.5*( (param.theta - 1) *A1 + k1_m * A1_m )**(2) *param.phi_x**(2) )  / (1 - k1_m *param.nu)
    A0_m = (param.theta *np.log(param.delta) - param.gamma *param.mu_c + param.mu_d + (param.theta - 1) * ( k0 + A0 * (1 - k1) ) +\
            k0_m + ( (param.theta - 1) * A2 + k1_m * A2_m) * param.sigma_bar**(2) * (1 - param.nu)\
            + 0.5*( (param.theta - 1) * A2 + k1_m *A2_m)**(2) *param.phi_s**(2) ) / (1 - k1_m)

    return k0_m, k1_m, A0_m, A1_m, A2_m

def rf_coeff(param, paramResults):

    k0 = paramResults.k0
    k1 = paramResults.k1
    A0 = paramResults.A0
    A1 = paramResults.A1
    A2 = paramResults.A2


    A1_f = param.theta -1 -(param.theta/param.psi) + (param.theta-1) * A1 *(param.rho - k1)
    A2_f = 0.5*(param.theta -1 - (param.theta/param.psi) )**2 + 0.5*( (param.theta -1) *A1 *param.phi_x)**2 + (param.theta -1) * A2*(param.nu -k1)
    A0_f = param.theta * np.log(param.delta) + (param.theta -1 - (param.theta/param.psi) ) *param.mu_c + \
           (param.theta -1) * (A0 + A2 * param.sigma_bar**(2) *(1 - param.nu) + k0 - k1 *A0) + 0.5*( (param.theta -1) * A2 *param.phi_s)**(2)

    return A0_f, A1_f, A2_f