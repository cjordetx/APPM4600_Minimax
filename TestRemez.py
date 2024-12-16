import numpy as np
import matplotlib.pyplot as plt
from Remez import remez


def driver():
    #gotta make a function
    f = lambda x: np.sin(x)
    a = 0
    b = np.pi/2
    xplot = np.linspace(a,b,1000)
    
    deg = 7
    p,maxerr,CheckState = remez(f,a,b,deg)
    g=lambda x: np.polyval(p[::-1],x)
    plt.figure()
    plt.plot(xplot,g(xplot))
    print("Max Error for minimax of deg ", deg, "is ",maxerr)
    print("Did it work?", CheckState)
    plt.show()
    return








driver()
