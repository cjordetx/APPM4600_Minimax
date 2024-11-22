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
import numdifftools as nd

def driver():

    N = 15 # degree number of polynomial
    a = -1 # left endpoint
    b = 1 # right endpoint
    E = 1 #error set value for while loop
    h = (a - b) / 2 #bisection of the interval
    Nmax = 100 #max number of iterations
    tol = 10 ** (-8) #tolerance
    xeval = np.linspace(a,b,N+1) #evaluation points

    f = lambda x: np.sin(x) #function f(x)
    fp = lambda x: np.cos(x) #f'(x)

    ''' Create interpolation nodes'''
    xint = np.array([np.cos(2*i - 1)*np.pi/2 for i in range(N+2)]) #n + 2 chebyshev nodes
    #print('xint', xint)

    '''Create interpolation data'''
    yint = f(xint) # function at chebyshev nodes

    ''' Create the Polynomial matrix'''
    P = Polynomial(xint, N)
    #creates the matrix with 1's in the first column, each row is an interpolation node
    # each column is what power that node is set to and the last column is (-1)^i where i is the row number

    'find coeffients and error'
    coeffs, E = coefficients(yint, P) # calculates P @ f(xint) to get the coeficients and max error term

    dcoeffs = np.zeros(len(coeffs)) # this calculates the coefficients for the derivative of the approximation
    for i in range(len(coeffs)):
        dcoeffs[i] = coeffs[i] * i

    dcoeffs = dcoeffs[1:] #deletes the first entry since the derivative is a polynomial of degree n-1
    errorfunc = lambda x: f(x) - numpy.polynomial.polynomial.polyval(x, coeffs) #error function f(x) - q(x)
    derrorfunc = lambda x: fp(x) - numpy.polynomial.polynomial.polyval(x, dcoeffs) #derivative of error function f'(x) - q'(x)

    roots, its = Newton(derrorfunc, xint, tol, Nmax)
    # calculates the roots and iterations from newton for the derivative of the error function at the nodes
    roots.sort() # sorts these roots
    print('Newton:', roots)

    q = eval_monomial(xint, coeffs, N, N+1) # evaluates q(xeval)


    yint1 = f(roots) #evaluates f(roots) or f at the output of the first iteration of newton
    P2 = Polynomial(roots, N) #starts over again with xint=roots

    coeffs1, E1 = coefficients(yint1, P2) #calculates the new coefficients and max error term using the polynomial matrix
    # with the new roots and the function evaluated at the new roots
    q1 = eval_monomial(roots, coeffs1, N, N+1)
    dcoeffs1 = np.zeros(len(coeffs1)) #calculates the new coefficients for the derivative of the approximation: q'(x)
    for i in range(len(coeffs1)):
        dcoeffs1[i] = coeffs1[i] * i

    dcoeffs1 = dcoeffs1[1:] #deletes first term
    errorfunc1 = lambda x: f(x) - numpy.polynomial.polynomial.polyval(x, coeffs1) #error function of the function minus the new "better" approximation
    derrorfunc1 = lambda x: fp(x) - numpy.polynomial.polynomial.polyval(x, dcoeffs1) #derivative of error function above

    roots2, its2 = Newton(derrorfunc1, roots, tol, Nmax) #calculates a second iteration of newton using the roots from the first one
    roots2.sort() #sorts these
    print('roots:', roots2)

    yint2 = f(roots2)  # evaluates f(roots) or f at the output of the first iteration of newton
    P3 = Polynomial(roots2, N)  # starts over again with xint=roots

    coeffs2, E2 = coefficients(yint2, P3)
    # calculates the new coefficients and max error term using the polynomial matrix
    # with the new roots and the function evaluated at the new roots

    dcoeffs2 = np.zeros(len(coeffs2))  # calculates the new coefficients for the derivative of the approximation: q'(x)
    for i in range(len(coeffs2)):
        dcoeffs2[i] = coeffs2[i] * i

    dcoeffs2 = dcoeffs2[1:]  # deletes first term
    errorfunc2 = lambda x: f(x) - numpy.polynomial.polynomial.polyval(x,coeffs2)
    # error function of the function minus the new "better" approximation
    derrorfunc2 = lambda x: fp(x) - numpy.polynomial.polynomial.polyval(x,dcoeffs2)
    # derivative of error function above

    yex = f(xint)
    err = (yex - q)
    #err1 = (yex - q1)
    #err2 = (yex - q2)
    plt.plot(xint, yex, label='exact')
    plt.plot(xint, q, label='approximation')
    #plt.plot(roots, q1, label='approximation')
    plt.plot(roots, q2, label='approx2')
    plt.legend()
    plt.show()
    plt.plot(xint, errorfunc(xint), label='error')
    plt.plot(xint, errorfunc2(xint), label='error2')
    plt.legend()
    plt.show()



def Polynomial(xint, N):
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


def eval_monomial(xeval, coeffs, N, Neval):
    yeval = coeffs[0] * np.ones(Neval + 1)

    #    print('yeval = ', yeval)

    for j in range(1, N + 1):
        for i in range(Neval + 1):
            #        print('yeval[i] = ', yeval[i])
            #        print('a[j] = ', a[j])
            #        print('i = ', i)
            #        print('xeval[i] = ', xeval[i])
            yeval[i] = yeval[i] + coeffs[j] * xeval[i] ** j

    return yeval


def Newton(f, x0, tol, Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
        J = nd.Jacobian(f)(x0)
        Jinv = inv(J)
        F = f(x0)

        x1 = x0 - Jinv.dot(F)

        if (norm(x1 - x0) < tol):
            xstar = x1
            return xstar, its

        x0 = x1

    xstar = x1
    return xstar, its





driver()
