import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import numpy.polynomial.chebyshev as ch
from nonlinearMinimaxApprox import nonlinearMinimax
from remez_poly import remez
from chebyshevApprox import chebyshevApprox, eval_chebyshev

# Approximation order
N = 9

# Functions to approximate

# f(x) = e^x
f = lambda x: np.exp(x)
fp = f
fpp = f
remezf = lambda x: np.exp(x)

# f(x) = x^(8/3)
#f = lambda x: np.cbrt(x ** 8)
#fp = lambda x: 8 * np.cbrt(x ** 5) / 3
#fpp = lambda x: 40 * np.cbrt(x ** 2) / 9
#remezf = lambda x: mp.cbrt(x ** 8)

# f(x) = x^2cos(x)
# f = lambda x: x ** 2 * np.cos(x)
# fp = lambda x: 2 * x * np.cos(x) - x ** 2 * np.sin(x)
# fpp = lambda x: 2 * np.cos(x) - x ** 2 * np.cos(x) - 4 * x * np.sin(x)
# remezf = lambda x: np.power(x, 2) * np.cos(x)

# Interval for approximation
a, b = -1, 1

# Initial Guess for coefficients

# Monomial
# int_coeff = np.ones(N+1)

# Chebychev Approximation
# print(chebshevApprox(f,a,b,N))
chebExpansion_coeff = chebyshevApprox(f,a,b,N)
cheb_coeff = ch.cheb2poly(chebExpansion_coeff)
int_coeff = np.zeros(N+1)
int_coeff[0:len(cheb_coeff)] = cheb_coeff

XP, info = nonlinearMinimax(f, fp, fpp, a, b, int_coeff, N)
if info == 1:
    print('Max Number of iterations')
rho_n = XP[0] # Maximum error
qstar_coeff = XP[1:N+2] # Minimax polynomial coefficients
xstar = XP[N+2:] # Points where maximum error occurs
print("Max Error", rho_n)
print("\n")
print("Max Error pts", xstar)

# Testing Implementation
xeval = np.linspace(a, b, 1000)
feval = f(xeval)

# Nonlinear minimax approximation
qstar_Neval = np.polyval(qstar_coeff[::-1], xeval)

# Remez implementation
#rem_coeff, _ = remez(remezf, N, a, b)
#rem_Neval = np.polyval(rem_coeff[::-1], xeval)

# Chebyshev Approx
cheb_Neval = eval_chebyshev(xeval,chebExpansion_coeff,(a,b))

# Plotting Chebyshev vs Minimax
plt.figure()
plt.plot(xeval, feval, label=r'$f(x)$') # f(x)
plt.plot(xeval, cheb_Neval, label=r"$T_{7}(x)$")
plt.plot(xeval, qstar_Neval, label=r'$q_{9}^\star(x)$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Chebyshev vs Minimax Approximation')
plt.legend()
plt.show()

plt.figure()
plt.plot(xeval, feval-cheb_Neval, label=r'$E_{Chebyshev}(x)$')
plt.plot(xeval, feval-qstar_Neval, label=r'$E_{Minimax}(x)$') # qN(x)
plt.xlabel(r'$x$')
plt.ylabel(r'$E(x)$')
plt.title('Minimax and Chebyshev Error')
plt.legend()
plt.show()


plt.figure()
plt.plot(xeval, feval, label=r'$f(x)$') # f(x)
#plt.plot(xeval, rem_Neval, label=r'$q_{}^\star(x)$ Remez'.format(N))
plt.plot(xeval, qstar_Neval, label=r'$q_{N}^\star(x)$')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$f(x)$ vs Minimax Approximation')
plt.legend()
plt.show()

plt.figure()
#plt.plot(xeval, feval-rem_Neval, label=r'$E_{}(x)$ Remez'.format(N)) # q1(x)
plt.plot(xeval, cheb_Neval-qstar_Neval) # qN(x)
plt.xlabel(r'$x$')
plt.ylabel(r'$E_{Difference}(x)$')
plt.title('Minimax Error')
plt.show()