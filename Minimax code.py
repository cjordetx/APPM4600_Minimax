import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt

def driver(): 

    f = lambda x: np.sin(x)
    fp = lambda x: np.cos(x)
    
    N = 10
    a = -3
    b = 1
    x0 = 0.5

    Nmax = 100
    tol = 1.e-14
    Neval = 100
    xeval = np.linspace(a,b,Neval+1)

    
    ''' Create interpolation nodes'''
    xint = np.linspace(a,b,N+2)
#    print('xint =',xint)
    '''Create interpolation data'''
    yint = f(xint)
#    print('yint =',yint)
    
    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint,N)
    print('V = ',V)

    'find coeffients and error'
    coeffs, error = coefficients(yint, V)
    print('coeffs, error = ', coeffs, error)

    'evaluate minimax polynomial using coeffs'
    yeval = eval_monomial(xeval, coeffs, N, Neval)
    print('yeval:', yeval)

    "finds the roots of |q'(x) - f'(x)| by newtons method"
    for i in range(len(xint)):
        [x, xstar, its] = newtonroots(x0, f, fp, coeffs, N, Nmax, tol)
    print('iterates:', x, 'final iterate:', xstar, '# of iterates:', its)



# exact function
    yex = f(xeval)
    err = norm(yex - yeval)
    maxerr =  max(np.abs(yex-yeval))
    print('err = ', err)
    print('maxerr = ', maxerr)
    plt.plot(xeval, yex, label = 'exact')
    plt.plot(xeval, yeval, label = 'approximation')
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


driver()
