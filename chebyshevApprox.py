import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def eval_chebyshev(x, a):
    # Uses Clenshaw algorithm
    bk1 = 0
    bk2 = 0
    for ak in a[:0:-1]:
        bk = ak + 2*x*bk1 - 1*bk2
        bk2 = bk1
        bk1 = bk
    return a[0] + x*bk1 - bk2

def chebshevApprox(f, a, b, N):

    """
    Continous Chebyshev Approximation

    Inputs:
    f    - function to be approximated
    a,b         - bounds for the interval the approximation is being constructed for
    N           - approximation order
    
    Returns:
    cheb_coeff - vector polynomial coefficients for chebysehv approximation
                    - (coefficients in order of increasing polynomial term)

    """

    cheb_coeff = np.zeros(N+1)

    # Chebyshev weighting function
    w = lambda x: 1 / np.sqrt(1 - x**2)

    for i in range(N+1): # Look at Lab 10 for context (Evaluating continous L2 apprrox)

        # Chebyshev i-th basis function
        phi_j = lambda x: eval_chebyshev(x,[0] * i + [1])
        # Function to evaluate phi_j^2(x)*w(x)
        phi_j_sq = lambda x: phi_j(x)**2 * w(x)
        # Evaluating normalization factor for coefficient
        norm_fac, _ = quad(phi_j_sq, a, b)

        # Function to evaluate phi_j(x)*f(x)*w(x)
        func_j = lambda x: phi_j(x) * f(x) * w(x)
        # Evaluating numerator factor for coefficient
        num_fac, _ = quad(func_j, a, b)

        # Calculating i-th coefficient
        cheb_coeff[i] = num_fac / norm_fac

    return cheb_coeff

# # Example
# f = lambda x: x**5-x**2-2
# a, b = -1, 1
# N = 3

# # Calculating chebchevy approx coefficients
# cheb_coeff = chebshevApprox(f,a,b,N)

# xeval = np.linspace(a,b,1000)
# feval = f(xeval)
# chebeval = eval_chebyshev(xeval, cheb_coeff)

# plt.figure()
# plt.plot(xeval,feval,label=r'$f(x)$')
# plt.plot(xeval,chebeval,label='Chebyshev')
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(xeval,np.abs(feval-chebeval))
# plt.xlabel('$x$')
# plt.ylabel('Abs Error')
# plt.show()
