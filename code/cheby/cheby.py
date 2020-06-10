import numpy as np 
import math

def nodes_11(n, size_state_vars):
    ''' Nodes on [-1, 1] '''
    x0 = - np.tile(np.cos(((np.linspace(1,n,n)*2-1)/(2*n))*np.pi),(size_state_vars,1))
    return x0 # size_state_vars x n array

def nodes_ab(x0, xmin, xmax):
    ''' Nodes on [xmin, xmax] '''
    interval = xmax - xmin
    x1       = (x0 + 1)*interval/2 + xmin 
    return x1# size_state_vars x n array

def normalize_node(x, xmin, xmax):
    z = (x- xmin)*2/(xmax-xmin)-1
    return z

def basisfunctions(z, degree):
        ''' 
        Returns the basis functions for
        z = normalized node in [-1,1]
        '''
        if isinstance(z, float):
            Tz = np.zeros(degree + 1)
            Tz[0] = 1
            Tz[1] = z
            for i in range(2, degree + 1):
                Tz[i] = 2*np.multiply(z,Tz[i-1]) - Tz[i-2]
            return Tz
        
        elif z.shape[0] == 1:
            T1 = np.zeros((degree + 1, degree + 1))
            T1[0,:] = 1
            T1[1,:] = z[0]
            for i in range(2, degree + 1):
                T1[i,:] = 2*np.multiply(z[0],T1[i-1,:]) - T1[i-2,:]
            return T1
        elif z.shape[0] == 2:
            T1 = np.zeros((degree + 1, degree + 1))
            T1[0,:] = 1
            T1[1,:] = z[0]
            T2 = np.zeros((degree + 1, degree + 1))
            T2[0,:] = 1
            T2[1,:] = z[1]
            for i in range(2, degree + 1):
                T1[i,:] = 2*np.multiply(z[0],T1[i-1,:]) - T1[i-2,:]
                T2[i,:] = 2*np.multiply(z[1],T2[i-1,:]) - T2[i-2,:]
            return T1, T2


def fapprox1(coefs,T):
    fval = (coefs*T).sum()
    return fval

def fapprox2(coefs,T1,T2):
    fval = np.tensordot(coefs, np.outer(T1, T2))
    return fval
