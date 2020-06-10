import numpy as np
np.set_printoptions(precision=10) # this is too global.
from types import SimpleNamespace 
import inspect
from ap_models import lrr
from cheby import cheby
from integration import gaussh


#-----------------------------------------------------------------------------------------
# 1. Choose the object to be created
#-----------------------------------------------------------------------------------------
#   - BY2004 without stochastic volatility: lrr.BY2004(mtype = 'noSV', ns = 1, years = 50)
#   - BY2004 with stochastic volatility:    lrr.BY2004(mtype = 'SV',   ns = 1, years = 50)
#   $ ns -> number of samples to be obtained in simulations
#   $ years -> length of time series (in years)
#-----------------------------------------------------------------------------------------

##by2004_noSV = lrr.BY2004(mtype = 'noSV', ns = 2, years = 100000)
by2004_SV = lrr.BY2004(mtype = 'SV', ns = 2, years = 100000)

# 1.1 Display methods and attributes in the BY2004 class (import inspect!)
#-----------------------------------------------------------------------------------------
methods = inspect.getmembers(by2004_SV, predicate=inspect.ismethod)
methods_name = [method[0] for method in methods]

attributes = inspect.getmembers(by2004_SV, lambda a:not(inspect.isroutine(a)))
attributes_name = [attribute[0] for attribute in attributes if not(attribute[0].startswith('__') and attribute[0].endswith('__'))]

#-----------------------------------------------------------------------------------------
# 2. Simulate time series
#-----------------------------------------------------------------------------------------
# The function can take seed = number as input, otherwise seed = 7 is the default value

#by2004_noSV.generate_time_series()
by2004_SV.generate_time_series()


#-----------------------------------------------------------------------------------------
# 3. Generate the log-linear approximation coefficients for wc, pd and price of rf asset
#-----------------------------------------------------------------------------------------
'''
by2004_noSV.ll_wc()
by2004_noSV.ll_pd()
by2004_noSV.ll_rf()
'''
# 
by2004_SV.ll_wc()
by2004_SV.ll_pd()
by2004_SV.ll_rf()

#-----------------------------------------------------------------------------------------
# (Chebyshev) Polynomial approximation - collocation
#-----------------------------------------------------------------------------------------
# 1. Generate grids for state variables and shocks
#    Results can be retrieved by calling:
#    by2004_SV.xt,   by2004_SV.xt1,  by2004_SV.sig2t, by2004_SV.sig2t1
#    by2004_SV.cgt1, by2004_SV.dgt1
#    
#    The method can take the ranges of variables over which be approximate as user input
#   .poly_next_period(self, mins , maxs, nint = 5, deg = 6):
#-----------------------------------------------------------------------------------------

by2004_SV.poly_next_period()

#-----------------------------------------------------------------------------------------
# RR: Or to replicate results from: https://github.com/robertrebnor/Numerical_Methods
# This setup was used in debugging... 

xmin = -0.0101476067634704
xmax = 0.0122825098694818
smin = 1e-14
smax = 0.00032395
min_states = np.array([xmin, smin], ndmin = 2)
max_states = np.array([xmax, smax], ndmin = 2)
by2004_SV.poly_next_period(min_states,max_states)

# Check output for next period

by2004_SV.xt1[:,1,1,1,1,1]
by2004_SV.sig2t1[:,1,1,1,1,1]
by2004_SV.cgt1[:,1,1,1,1,1]
by2004_SV.dgt1[:,1,1,1,1,1]

# Starting value of approximating coefficients for root finding algorithm: the constant A0 from loglinearization

coeff = np.zeros((7,7))
coeff[0,0] = by2004_SV.A0
coeff_use = coeff.flatten()

# Call the objective function (reshaped Euler equation) with above-mentioned starting value 

by2004_SV.obj_euler(coeff_use)
# Check wc ratio values
by2004_SV.wct[:,1,1,1,1,1]
# Check Euler equation values
by2004_SV.F

#-----------------------------------------------------------------------------------------
# 2. Root-finding step: find coefficients of approximation such that 
#    the reshaped Euler equation for the consumption claim is 0 in each state 
#    For noSV case: for each  x_i
#    For SV   case: for each (x_i, sig2_j) 
#-----------------------------------------------------------------------------------------

#by2004_SV.wc_coeff()

# Display optimal coefficients
optim_coef = by2004_SV.coefs

# Display Euler equation values 
by2004_SV.obj_euler(optim_coef)
by2004_SV.F


########################################################################################################
# Use as input to check for multiple roots...
########################################################################################################
# Coef Python replication - gitrep
#coeff = np.array([ 6.21382034e+00, -7.47524686e-02,  3.13155496e-04,  1.82238274e-06,\
# -8.89589071e-09, -1.85846337e-10, -1.10743401e-10,  1.62127633e-01,\
#       -1.06865956e-03, -1.32614477e-05,  6.16625842e-09,  1.72065003e-09,\
#       -2.32516807e-10, -1.60207660e-10,  3.08332245e-04,  1.89830433e-05,\
#        1.15883766e-07, -3.22466684e-09, -5.32698887e-11, -1.45419365e-10,\
#       -7.00962664e-11, -4.88267682e-06, -2.09497436e-07,  2.23908542e-09,\
#        2.57540783e-11, -8.38951048e-11, -6.21157823e-11, -1.58637479e-11,\
#        5.42911404e-08,  1.84363721e-10, -9.05346846e-11, -1.50631251e-11,\
#       -2.89847661e-11, -1.74380983e-11, -2.04806979e-12, -2.75988111e-10,\
#        4.36094490e-11, -4.98000307e-12, -7.69623742e-12, -7.41488435e-12,\
#       -2.69714733e-12, -1.57790819e-13, -3.77338030e-12, -1.18814349e-12,\
#       -8.49927685e-13, -1.07059177e-12, -1.15481900e-12, -2.16013745e-13,\
#       -6.87146529e-15])
########################################################################################################
