import numpy as np

def gh_nodes(n):
    '''This function determines the abscisas (x) and weights (w) for the
       Gauss-Hermite quadrature of order n>1, on the interval [-INF, +INF].
       This function is valid for any degree n>=2, as the companion matrix
       (of the n'th degree Hermite polynomial) is constructed as a
       symmetrical matrix, guaranteeing that all the eigenvalues (roots)
       will be real.
       Geert Van Damme
       geert@vandamme-iliano.be '''


    ## Alternatively:
    ## [xi, wi] = np.polynomial.hermite.hermgauss(nGH)

    # Building the companion matrix CM
    # CM is such that det(xI-CM)=L_n(x), with L_n the Hermite polynomial
    # under consideration. Moreover, CM will be constructed in such a way
    # that it is symmetrical.

    i = np.linspace(1,n-1,n-1)
    a = np.sqrt(i/2)
    CM = np.diag(a, k = 1) + np.diag(a, k = -1)
    
    # Determining the abscissas (x) and weights (w)
    # - since det(xI-CM)=L_n(x), the abscissas are the roots of the
    # characteristic polynomial, i.d. the eigenvalues of CM;
    # - the weights can be derived from the corresponding eigenvectors.

    [L, V] = np.linalg.eig(CM) # returns V', where V is the output of [V, L] = eig(CM) in Matlab
    indx = np.argsort(L)
    L.sort()
    V = V[:,indx.tolist()]
    w = np.sqrt(np.pi) * np.power(V[0,:],2)

    return L, w





