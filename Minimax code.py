import numpy as np
import numpy.linalg as la
import numpy.polynomial.polynomial
from numpy.linalg import inv
from numpy.linalg import norm
from numpy.polynomial import Polynomial
from numpy.polynomial import Chebyshev
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy import optimize

def driver():

    N = 7
    a = -1
    b = 1
    E = 0
    h = (a - b) / 2
    Neval = 100
    tol = 10 ** (-8)
    xeval = np.linspace(a,b,N+1)

    f = lambda x: np.sin(x)
    fp = lambda x: np.cos(x)

    
    ''' Create interpolation nodes'''
    xint = np.linspace(a,b,N+2)
#    print('xint =',xint)
    '''Create interpolation data'''
    yint = f(xint)
#    print('yint =',yint)
    
    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint,N)
    #print('V = ',V)

    'find coeffients and error'
    coeffs, error1 = coefficients(yint, V)
    #print('coeffs, error = ', coeffs, error1)

    dcoeffs = np.zeros(len(coeffs))
    for i in range(len(coeffs)):
        dcoeffs[i] = coeffs[i] * i

    dcoeffs = dcoeffs[1:]
    errorfunc = lambda x: f(x) - numpy.polynomial.polynomial.polyval(x, coeffs)
    derrorfunc = lambda x: fp(x) - numpy.polynomial.polynomial.polyval(x, dcoeffs)


    #roots = optimize.root_scalar(derrorfunc, bracket = [xeval[], xeval[]], x0 = 0)
    #print('roots = ', roots)
    'evaluate minimax polynomial using coeffs'
    yeval = eval_monomial(xeval, coeffs, N, N)
    #print('yeval:', yeval)



# exact function
    yex = f(xeval)
    err = (yex - yeval)
    maxerr =  max(np.abs(yex-yeval))
    #print('err = ', err)
    print('maxerr = ', maxerr)
    plt.plot(xeval, yex, label = 'exact')
    plt.plot(xeval, yeval, label = 'approximation')
    plt.legend()
    plt.show()
    plt.plot(xeval, errorfunc(xeval), label = 'error')
    plt.legend()
    plt.show()



def Vandermonde(xint, N):
    V = np.zeros((N + 2, N + 2))

    ''' fill the first column'''
    for j in range(N + 2):
        V[j][0] = 1.0
        V[j][N + 1] = (-1) ** (j + 1)

    for i in range(1, N + 1):
        for j in range(N + 2):
            V[j][i] = xint[j] ** i

    return V


def coefficients(fvalues, V):
    result = inv(V) @ fvalues
    error = result[-1]
    coeffs = result[:-1]
    return coeffs, error


def  eval_monomial(xeval,coeffs,N,Neval):

    yeval = coeffs[0]*np.ones(Neval+1)
    
#    print('yeval = ', yeval)
    
    for j in range(1,N+1):
      for i in range(Neval+1):
#        print('yeval[i] = ', yeval[i])
#        print('a[j] = ', a[j])
#        print('i = ', i)
#        print('xeval[i] = ', xeval[i])
        yeval[i] = yeval[i] + coeffs[j]*xeval[i]**j

    return yeval


def error_eval(xeval, f, fp, coeffs, N, Neval):
    approx = coeffs[0] * np.ones(Neval + 1)
    dapprox = np.zeros(Neval + 1)
    error = np.zeros(Neval + 1)
    derror = np.zeros(Neval + 1)

    for j in range(1, N + 1):
        for i in range(Neval + 1):
            approx[i] = approx[i] + coeffs[j] * xeval[i] ** j
            dapprox[i] = dapprox[i] + j * coeffs[j] * xeval[i] ** (j - 1)

    for i in range(Neval + 1):
        error[i] = approx[i] - f(xeval[i])
        derror[i] = dapprox[i] - fp(xeval[i])

    return derror

#roots = fsolve(error_eval, x0=xeval, args=(f, fp, coeffs, N, N))
#print('roots=', roots)


def newton(f, fp, p0, tol, Nmax):
    """
    Newton iteration.

    Inputs:
      f,fp - function and derivative
      p0   - initial guess for root
      tol  - iteration stops when p_n,p_{n+1} are within tol
      Nmax - max number of iterations
    Returns:
      p     - an array of the iterates
      pstar - the last iterate
      info  - success message
            - 0 if we met tol
            - 1 if we hit Nmax iterations (fail)

    """
    p = np.zeros(Nmax + 1);
    p[0] = p0
    for it in range(Nmax):
        p1 = p0 - f(p0) / fp(p0)
        p[it + 1] = p1
        if (abs(p1 - p0) < tol):
            pstar = p1
            info = 0
            return [p, pstar, info, it]
        p0 = p1
    pstar = p1
    info = 1
    return [p, pstar, info, it]


def newtonroots(x0, f, fp, coeffs, N, Nmax, tol):
    x = np.zeros(Nmax+1)
    x[0] = x0
    for it in range(Nmax):
        approx = coeffs[0]
        dapprox = 0
        for j in range(1, N + 1):
            approx += coeffs[j] * x0 ** (j)
            dapprox += j * coeffs[j] * x0 ** (j - 1)
        g = approx - f(x0)
        gp = dapprox - fp(x0)
        x1 = x0 - (g/gp)
        x[it + 1] = x1
        if (abs(x1 - x0) < tol):
            xstar = x1
            return [x, xstar, it]
        x0 = x1
    xstar = x1
    return [x, xstar, it]


def bisection(f,a,b,tol):
    # Inputs:
    # f,a,b - function and endpoints of initial interval
    # tol - bisection stops when interval length < tol
    # Returns:
    # astar - approximation of root
    # ier - error message
    # - ier = 1 => Failed
    # - ier = 0 == success
    # first verify there is a root we can find in the interval
    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
        ier = 1
        astar = a
        return [astar, ier]
    #verify end points are not a root
    if (fa == 0):
        astar = a
        ier =0
        return [astar, ier]

    if fb == 0:
        astar = b
        ier = 0
        return [astar, ier]

    count = 0
    d = 0.5*(a+b)

    while (abs(d - a) > tol):
        fd = f(d)
        if (fd == 0):
            astar = d
            ier = 0
            return [astar, ier]
        if (fa * fd < 0):
            b = d
        else:
            a = d
            fa = fd
        d = 0.5 * (a + b)
        count = count + 1
    #    print('abs(d-a) = ', abs(d-a))
    astar = d
    ier = 0
    return [astar, ier]


driver()