def by2004():

    params = dict()
    # Preferences
    params['delta']  = .998
    params['gamma'] = 10
    params['psi'] = 1.5
    # Consumption
    params['mu_c'] = .0015
    params['rho'] = .979
    params['phi_x'] = .044
    params['sigma_bar'] = 0.0078
    params['nu'] = .987
    params['phi_s'] = .0000023
    # Dividends
    params['mu_d'] = .0015
    params['phi'] = 3
    params['phi_dc'] = 0
    params['phi_d'] = 4.5
    params['theta'] = (1 - params['gamma'])/( 1 - 1/params['psi'])
    
    return params

