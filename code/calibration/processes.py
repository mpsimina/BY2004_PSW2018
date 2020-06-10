import numpy as np

def simulate_model(id_model, years, ns, param):
#ns = number of samples
    delta, gamma, psi, mu_c, \
        rho, phi_x, sigma_bar, \
            nu, phi_s, mu_d, phi,\
                phi_dc, phi_d, theta = param.delta,     param.gamma, param.psi,     \
                                       param.mu_c,      param.rho,   param.phi_x,   \
                                       param.sigma_bar, param.nu,    param.phi_s,   \
                                       param.mu_d,      param.phi,   param.phi_dc,  \
                                       param.phi_d,     param.theta

    
    months = years*12

    # Column vectors of time series
    x, sig, cg, dg = np.zeros((months+1, ns)), np.zeros((months+1, ns)), \
                   np.zeros((months+1, ns)), np.zeros((months+1, ns))


    # Initial values of time series (t = 0)
    x[0,:], cg[0,:], dg[0,:], sig[0,:] = 0, sigma_bar**(2), mu_c, mu_d


    #Compute the shocks
    np.random.seed(6062020)
    shock_x, shock_c, shock_d, shock_sig = \
        np.random.normal(0, 1, (months+1, ns)), np.random.normal(0, 1, (months+1, ns)), \
            np.random.normal(0, 1, (months+1, ns)), np.random.normal(0, 1, (months+1, ns))
    
    if id_model == 'noSV': 
        ''' BY2004 constant volatility '''
        for i in range(ns):
            for t in range(1, months+1):    
                x[t, i]   = rho*x[t-1, i] + phi_x*sigma_bar*shock_x[t, i]
                cg[t, i]  = mu_c + x[t, i] + sigma_bar*shock_c[t, i]
                dg[t, i]  = mu_d +  phi*x[t, i] + phi_d*sigma_bar*shock_d[t, i] + phi_dc*sig[t-1, i]**(0.5)*shock_c[t, i]
    elif id_model == 'SV':
        ''' BY2004 stochastic volatility '''
        for i in range(ns):
            for t in range(1, months+1):    
                x[t, i]   = rho*x[t-1, i] + phi_x*sig[t-1, i]**(0.5)*shock_x[t, i]
                sig[t, i] = max( sigma_bar**(2)*(1 - nu) + nu*sig[t-1, i] + phi_s*shock_sig[t-1, i], np.spacing(1))
                cg[t, i]  = mu_c + x[t, i] + sig[t, i]**(0.5)*shock_c[t, i]
                dg[t, i]  = mu_d +  phi*x[t, i] + phi_d*sig[t-1, i]**(0.5)*shock_d[t, i] + phi_dc*sig[t-1, i]**(0.5)*shock_c[t, i] # phi_dc = 0 !

    return x, sig, cg, dg
