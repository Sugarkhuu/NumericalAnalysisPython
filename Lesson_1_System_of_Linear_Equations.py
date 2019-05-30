import numpy as np
import scipy.linalg

# Warning! Matrices need to defined everytime you run
# Define "a" and "b" in a*x = b
a = np.array([[ 3, -1, 4], \
[-2, 0, 5], \
[ 7, 2, -2]], dtype=float)
b = np.array([ 6.0, 3.0, 7.0])

# AX=B - X=inv(A)*B
def gaussElimin(a,b):
    # Gauss Elimination Method
    n = len(b)
    # Elimination Phase
    for k in range(0,n-1):
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                a[i,k] = 0.0
                b[i] = b[i] - lam*b[k]
    # Back Substitution Phase
    for k in range(n-1,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    return a, b

# Example 1: Gauss Elimination and Inverse matrix
[asol,bsol] = gaussElimin(a,b)
bsol1=np.dot(np.linalg.inv(a),b)


def LUdecomp(a):
    # LU decomposition of Doolittle
    n = len(a)
    for k in range(0,n-1):
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                a[i,k] = lam
    L=a.copy()
    U=a.copy()
    for j in range(n):
        L[j,j+1:n]=0
        L[j,j]=1
        U[j,0:j]=0
    return L, U


def mySolve(L,U,b):
    #solve a*x = b as L*U*x = b -> Ly=b and Ux=y
    n = len(b)
    # Solution of Ly = b
    for k in range(n):
        b[k] = (b[k] - np.dot(L[k,0:k],b[0:k]))/L[k,k]
    # Solution of Ux = y
    for k in range(n-1,-1,-1):
        b[k] = (b[k] - np.dot(U[k,k+1:n],b[k+1:n]))/U[k,k]
    return b

# Example 2: LU with pivoting in scipy and our function
[P, Ls, Us] = scipy.linalg.lu(a) # LU with scipy PLU=A
[L,U] = LUdecomp(a)              # LU defined
bsol = mySolve(L,U,b)            # solving a*x=b


import math
def choleski(a):
    # cholesky decomposition (LU)
    n = len(a)
    for k in range(n):
        try:
            a[k,k] = math.sqrt(a[k,k] \
             - np.dot(a[k,0:k],a[k,0:k]))
        except NegativeInSqrt:
            print('Matrix is not positive definite')
        for i in range(k+1,n):
            a[i,k] = (a[i,k] - np.dot(a[i,0:k],a[k,0:k]))/a[k,k]
    for k in range(1,n): 
        a[0:k,k] = 0.0
    return a

# Example 3: LU decomposition with Cholesky decomposition
a = np.array([[ 1.44, -0.36, 5.52, 0.0], \
[-0.36, 10.33, -7.78, 0.0], \
[ 5.52, -7.78, 28.40, 9.0], \
[ 0.0, 0.0, 9.0, 61.0]])
b = np.array([0.04, -2.15, 0.0, 0.88])

L= choleski(a)                    # with our function
Lnp = np.linalg.cholesky(a)       # LU with np cholesky
bsol1=np.dot(np.linalg.inv(a),b)  # solving a*x=b with inverse matrix
bsol = mySolve(L,L.transpose(),b) # solving a*x=b with cholesky LU