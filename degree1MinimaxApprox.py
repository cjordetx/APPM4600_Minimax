import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from nonlinearMinimaxApprox import nonlinearMinimax
from remez_poly import remez

# Function and interval
f = np.exp
a, b = -1, 1

# Degree 1 Minimax approximation
a0 = np.exp(-1)/2 + np.sinh(1) - np.sinh(1)*np.log(np.sinh(1))/2
a1 = np.sinh(1)
q1 = lambda x: a0 + a1*x

# Evaluation function and approximation
xeval = np.linspace(a,b,1000)
feval = f(xeval)
q1eval = q1(xeval)

# Plotting Approximation and function
plt.figure()
plt.plot(xeval, feval, label=r'$e^x$') # f(x)
plt.plot(xeval, q1eval, label=r'$q_1^\star(x)$') # q1(x)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
# plt.savefig("C:/Users/crazy/OneDrive - UCB-O365/Classes/2024 Fall/APPM 4600/Project/deg1ApproxPlot.png",dpi=300)
plt.close()


# Plotting Error
Eeval = feval-q1eval
plt.figure()
plt.plot(xeval, Eeval) # f(x)
plt.xlabel(r'$x$')
plt.ylabel(r'$E_1(x)$')
# plt.savefig("C:/Users/crazy/OneDrive - UCB-O365/Classes/2024 Fall/APPM 4600/Project/deg1ApproxError.png",dpi=300)
# plt.show()
plt.close()

# Testing nonlinear minimax approx code
N = 2
# Coefficients for 1 order monomial
int_coeff = np.ones(N+1)
# int_coeff = np.array([1.00009000287256, 0.9973092607333836, 0.49883509185510316, 0.17734527451461188, 0.04415554008758067])

fp = f
fpp = f

qstar_coeff, info = nonlinearMinimax(f, fp, fpp, a, b, int_coeff, N)
if info == 1:
    print('Max Number of iterations')

# print(np.array([a0, a1]))
print(qstar_coeff[1:N+2])
f = mp.exp
rem_coeff, _ = remez(f,N,a,b)
print(rem_coeff)

qNeval = np.polyval(qstar_coeff[N+1:0:-1], xeval)
remNeval = np.polyval(rem_coeff[::-1], xeval)

plt.figure()
plt.plot(xeval, feval, label=r'$e^x$') # f(x)
plt.plot(xeval, qNeval, label=r'$q_{}^\star(x)$'.format(N)) 
plt.plot(xeval, remNeval, label=r'$q_{}^\star(x)$ Remez'.format(N)) 
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'$f(x)$ vs Minimax Approximation')
plt.legend()
plt.show()

plt.figure()
plt.plot(xeval, feval-qNeval, label=r'$E_{}(x)$'.format(N)) # q1(x)
plt.plot(xeval, feval-remNeval, label=r'$E_{}^(x)$ Remez'.format(N)) # q1(x)
plt.xlabel(r'$x$')
plt.ylabel(r'$E_1(x)$')
plt.title('Minimax Error')
plt.legend()
plt.show()