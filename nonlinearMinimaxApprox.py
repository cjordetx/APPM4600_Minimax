import numpy as np

def newtonND(F,J_F,X0, tol, Nmax):
  """
  Newton N-Dimensional iteration.
  
  Inputs:
    F,J_F - Vector containing system of non-linear equations and Jacobian
    X0   - initial guess for root
    tol  - iteration stops when X_n,X_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    X     - an array of the iterates
    Xstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
    it    - number of iterations
     
  """
  X = []
  X.append(X0)
  for it in range(Nmax):
      Y0 = np.linalg.solve(J_F(X0), -F(X0))
      X1 = X0 + Y0
      X.append(X1)
      if (np.linalg.norm(Y0,2) < tol):
          Xstar = X1
          info = 0
          return [np.array(X),Xstar,info,it+1]
      X0 = X1
  Xstar = X1
  info = 1
  return [np.array(X),Xstar,info,it+1]

def nonlinearMinimax(f, fp, fpp, a, b, int_coeff, N):

    """
    Nonlinear Minimax Approximation Solver

    Inputs:
    f,fp,fpp    - function to be approximated, and its first and second derivatives
    a,b         - bounds for the interval the approximation is being constructed for
    int_coeff   - vector of polynomial coefficients for initial guess at minimax approximation
                    - (coefficients in order of increasing polynomial term)
    N           - approximation order
    Returns:
    qstar_coeff - vector of polynomial coefficients for final minimax approximation
                    - (coefficients in order of increasing polynomial term)
    info        - success message
                    - 0 if we met tol
                    - 1 if we hit Nmax iterations (fail)
        
    """

    # Contructis fucntion for root-solve
    def Feval(X):
        p = X[N+1:0:-1]
        qstar = lambda x: np.polyval(p,x)
        qstarp = lambda x: np.polyval(np.arange(N,0,-1)*p[:-1],x)

        F = np.zeros(2*N+2)
        rho = X[0]
        sign = (-1)**np.arange(1,N+1)
        Xstar = X[N+2:]
        F[0] = f(a) - qstar(a) - rho
        F[1:N+1] = f(Xstar) - qstar(Xstar) - sign*rho
        F[N+1] = f(b) - qstar(b) - (-1)**(N+1)*rho
        F[N+2:] = fp(Xstar) - qstarp(Xstar)
        return F

    # Construct jacobian of function
    def J_F(X):
        p = X[N+1:0:-1]
        Xstar = X[N+2:]
        qstarpp = lambda x: np.polyval(np.arange(N,1,-1)*np.arange(N-1,0,-1)*p[:-2],x)

        J_F = np.zeros((2*N+2,2*N+2))
        J_F[:N+2,0] = -(-1)**np.arange(0,N+2)
        J_F[:N+2,1:N+2] = -np.vander(np.concatenate(([a],Xstar,[b])),N+1, increasing=True)
        J_F[0,N+2:] = np.zeros(N)
        J_F[1:N+1,N+2:] = np.diag(fp(Xstar))
        J_F[N+1,N+2:] = np.zeros(N)
        J_F[N+2:,0:2] = np.zeros((N,2))
        J_F[N+2:,2:N+2] = -np.vander(Xstar,N,increasing=True)@np.diag(np.arange(1,N+1))
        J_F[N+2:,N+2:] = np.diag(fpp(Xstar)-qstarpp(Xstar))
        return J_F


    # Using Newton's method to solve for minimax approx polynomial coefficients

    # Initial guess for maximum error
    xtest = np.linspace(a,b,1000)
    qstar = lambda x: np.polyval(int_coeff[::-1],x)
    rho_0 = np.max(np.abs(f(xtest)-qstar(xtest)))

    # Initial guess of points where maximum error occurs
    Xstar_0 = np.linspace(a,b,N+2)[1:-1] 

    X0 = np.concatenate(([rho_0],int_coeff,Xstar_0))

    (_,qstar_coeff,info,it) = newtonND(Feval,J_F,X0, 1e-8, 20)

    return (qstar_coeff, info)
      


# N = 10
# print(np.arange(N,1,-1))
# print(np.arange(N-1,0,-1))

# x = np.arange(1,10)
# print(x)
# print(np.concat(([0],x,[10])))
      
   