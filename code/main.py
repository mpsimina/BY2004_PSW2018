import numpy as np
np.set_printoptions(precision=10) # this is too global.
from types import SimpleNamespace 
import inspect

from ap_models import lrr
from cheby import cheby
from integration import gaussh

by2004_noSV = lrr.BY2004(mtype = 'noSV', ns = 2, years = 50)
by2004_SV = lrr.BY2004(mtype = 'SV', ns = 2, years = 50)

# See methods (functions) in the BY2004 class
methods = inspect.getmembers(by2004_SV, predicate=inspect.ismethod)
methods_name = [method[0] for method in methods]

attributes = inspect.getmembers(by2004_SV, lambda a:not(inspect.isroutine(a)))
attributes_name = [attribute[0] for attribute in attributes if not(attribute[0].startswith('__') and attribute[0].endswith('__'))]

# Simulating time series

by2004_noSV.generate_time_series()
by2004_SV.generate_time_series()


# Loglinear approximation
'''
by2004_noSV.ll_wc()
by2004_noSV.ll_pd()
by2004_noSV.ll_rf()

by2004_SV.ll_wc()
by2004_SV.ll_pd()
by2004_SV.ll_rf()
'''
# (Chebyshev) Polynomial approximation - collocation

# produces grids for variables at t and t+1, as well as grids with evaluated polynomials

#by2004_noSV.poly_approx()

by2004_SV.poly_approx()


#coeff = np.random.rand(7,7)*4
#by2004_SV.obj_euler(coeff)
#by2004_SV.wc_coeff()
by2004_SV.G
