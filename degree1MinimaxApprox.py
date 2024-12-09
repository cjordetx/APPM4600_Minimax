import numpy as np
import matplotlib.pyplot as plt
from nonlinearMinimaxApprox import nonlinearMinimax

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

fp = f
fpp = f

qstar_coeff, info = nonlinearMinimax(f, fp, fpp, a, b, int_coeff, N)

print(np.array([a0, a1]))
# print(info)
print(qstar_coeff[1:N+2])

qNeval = np.polyval(qstar_coeff[N+1:0:-1], xeval)
q2Correct = 0.9952 + 1.1361*xeval + 0.5479*xeval**2

plt.figure()
plt.plot(xeval, feval, label=r'$e^x$') # f(x)
plt.plot(xeval, qNeval, label=r'$q_{}^\star(x)$'.format(N)) # q1(x)
plt.plot(xeval, q2Correct, label=r'$q_2^\star(x)$ correct') # q1(x)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
plt.show()