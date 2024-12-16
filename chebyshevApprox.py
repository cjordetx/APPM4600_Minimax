import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def eval_chebyshev(x, a, interval=None):
    """
    Evaluates the Chebyshev polynomial of the first kind at x using Clenshaw's algorithm.
    If an interval [a, b] is provided, x is transformed to the corresponding t in [-1, 1].
    
    Parameters:
    x - The point(s) at which to evaluate the polynomial.
    a - Coefficients of the Chebyshev polynomial.
    interval - Optional tuple (a, b) representing the interval to which x belongs.
    
    Returns:
    Value of the Chebyshev polynomial at x.
    """
    if interval is not None:
        x = 2 * (x - interval[0]) / (interval[1] - interval[0]) - 1
    
    bk1 = 0
    bk2 = 0
    for ak in a[:0:-1]:
        bk = ak + 2 * x * bk1 - bk2
        bk2 = bk1
        bk1 = bk
    return a[0] + x * bk1 - bk2

def chebyshevApprox(f, a, b, N):
    """
    Continuous Chebyshev Approximation over an arbitrary interval [a, b].
    
    Parameters:
    f - Function to be approximated.
    a, b - Bounds of the interval over which the approximation is constructed.
    N - Order of the approximation.
    
    Returns:
    cheb_coeff - Vector of polynomial coefficients for the Chebyshev approximation
                 (coefficients are in order of increasing polynomial term).
    """
    
    cheb_coeff = np.zeros(N + 1)
    
    # Chebyshev weighting function
    w = lambda x: 1 / np.sqrt(1 - x**2)
    
    for i in range(N + 1):
        # Chebyshev i-th basis function on [a, b]
        phi_j = lambda x: eval_chebyshev(x, [0] * i + [1], interval=(a, b))
        
        # Function to evaluate phi_j^2(x)*w(x) over [a, b]
        phi_j_sq = lambda x: phi_j(x)**2 * w(2 * (x - a) / (b - a) - 1)
        
        # Evaluating normalization factor for coefficient
        norm_fac, _ = quad(phi_j_sq, a, b)
        
        # Function to evaluate phi_j(x)*f(x)*w(x) over [a, b]
        func_j = lambda x: phi_j(x) * f(x) * w(2 * (x - a) / (b - a) - 1)
        
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
# cheb_coeff = chebyshevApprox(f,a,b,N)

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
