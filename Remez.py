import numpy as np
import scipy.linalg as lalg
import mpmath as mp
from scipy import integrate
import matplotlib.pyplot as plt
from chebyshevApprox import chebshevApprox
from chebyshevApprox import eval_chebyshev
from scipy.signal import find_peaks


#Remez algorithm code. Jackson Braun, Last Update 12/09/2024


#builds the chebysev nodes that we will go off of. This old code
def chebyshev_points(a,b,Nint):
    index = np.arange(1, Nint+1)
    range_ = abs(b- a)
    return (a+b)/2 + .5* (b-a) * np.cos((2*index - 1)/(2*Nint)*np.pi)

#bisection Code. Will be used for finding roots, may also look into secant tho I think this may make the most sense
def bisection(f,a,b,n,tol):
    fa = f(a)
    fb = f(b)
    print("a: ",a)
    print("b: ", b)
    print("f(a): ", fa)
    print("f(b): ", fb)
    err = 0
    if fa*fb < 0:
        i =0
        while(i < n ):
            xn = (b+a)/2
            fx = f(xn)
            if fx*fa < 0:
                b = xn
                fb = fx
            elif fx*fa > 0:
                a = xn
                fa = fx
            else:
                x = xn
                err = 1
                return [x,err]
            i = i+1
            if (abs(a - b) < tol):
                x = xn
                err = 1
                return [x,err]
        x =xn
        err = 1

    elif fa*fb > 0:
        err = 0
        x = a
        i = 0
    elif fa ==0:
        x = a
        err = 1
        i=0
    else:
        x = b
        err  = 1
        i=0
    return [x, err]
            
def FindPeaks(E,low,high):
    dx =  1e-5 #kinda arbitrary, code im referencing uses the mpmath toolbox to get like a kinda nicer one that uses floating point stuff to get the small timestep with like relatve error on the scale region. I should probably lean towards that
    df = lambda x: (np.abs(E(x+dx))-np.abs(E(x-dx)))/(2*dx)
    return bisection(df,low,high,50,10**(-10))

def RemezMatrix(x,deg):
    A = np.zeros((deg+2,deg+2))
    for i in range(deg+2):
        for j in range(deg+1):
            A[i][j] = x[i]**j
        A[i][j+1] = (-1)**i

    return A

def remez(f,a,b,deg):
    #think I will probably simplify
    #Start with chebyshev, 0---> n+1
    #make the matrix, I think I can use a python function for this (powers) if that is too shitty then I can for loop it (would be seperate function)
    #Solve the matrix, maybe just numpy.solve but whoknows
    #Build the polynomial from that
    #Root find to get the maxima and adjust the points
    #Check tollarence 
    #Loop through to the 3rd step


    xint = sorted(chebyshev_points(a,b,deg+2))
    
    xplot = np.linspace(a,b,1000)
    numits =0
    CheckState = False

    while CheckState == False and numits < 10:
        #Do some BS
        #set up the matrix
        bvec = f(xint)
        A = RemezMatrix(xint,deg)

        print("Shape A", np.shape(A))
        print("shape bvec", np.shape(bvec))
        print(A)
        #Solve the Matrix
        p = lalg.solve(A,bvec)
        pUseME = p[:-1]
        print("Poly coefs: ", pUseME)
        
        #build the polynomial g(x)
        g=lambda x: np.polyval(pUseME[::-1],x)

        plt.plot(xplot,f(xplot))
        plt.plot(xplot,g(xplot))
        plt.show()
        E = lambda x: f(x) - g(x)



        #rootfind the max of the system, Couple of options here, may just use the find_peaks() function like in the remezfit code that is online
        #this does bisection a couple of times. Is quite silly Almost done tho
        roots  = np.zeros(deg+1)

        for i in range(deg+1):
            [roots[i],trash] = bisection(E,xint[i],xint[i+1],50,10**(-6))

        

        print("Roots ", roots)
        #I am now finding roots well I thinK!


        #check for peak between edge and first found root. As well as at the end. Need a out from bisection to tell me if its good or not for the if
        #Now I have my roots, need to feed into peak finding function type thing. This is getting weird
        xpeaks = np.zeros(deg+2)
        #This checks the fact that the max will either between the boundary point and the first intersection or at the boundary and forces that to be true
        xpeaks[0],err = FindPeaks(E,a,roots[0]-.001)
        if err == 0:
            xpeaks[0] = a
        xpeaks[-1],err = FindPeaks(E,roots[-1]+.001,b)
        if err == 0:
            xpeaks[-1] = b
        plt.plot(xplot,abs(E(xplot)))
        plt.title("Error abs")
        plt.show()
        #Uses the equiosiclation and that it is required to be true due to the matrix that we are using. 
        for j in range(deg):
            [xpeaks[j+1],trash] = FindPeaks(E,roots[j]+.001,roots[j+1]-.001)
            
            print("PeakFind:",trash)

        #hypothetically should be ok to check now. I think I should do a sanity check on how many of these peaks I have. I also need to add stuff to my bisection
        #Need to check difference between min peak error and max peak error 
        #Check amount changed and either break out of loop or  do it again
        print("peaks ", xpeaks)
        minpeak = min(abs(E(xpeaks)))
        maxpeak = max(abs(E(xpeaks)))
        xint=xpeaks
        plt.figure()
        plt.plot(xplot,E(xplot))
        plt.legend(["Error plot of degree: ", deg])
        print("Max Error: ", maxpeak)
        print("Min Err: ", minpeak)
        if maxpeak<=1.05*minpeak:
            maxerr = maxpeak
            CheckState = True
        numits = numits+1
    plt.show()

    maxerr = maxpeak


    #p is the polynomial coeffecents, maxerr is the value of the absolute max err found through remez
    return pUseME,maxerr,CheckState
