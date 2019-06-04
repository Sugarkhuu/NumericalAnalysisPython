
def gauss_seidel(m, x0=None, eps=1e-5, max_iteration=100):
  """
  Parameters
  ----------
  m  : list of list of floats : coefficient matrix
  x0 : list of floats : initial guess
  eps: float : error tolerance
  max_iteration: int
  
  Returns
  -------  
  list of floats
      solution to the system of linear equation
  
  Raises
  ------
  ValueError
      Solution does not converge
  """
# author: Worasait Suwannik http://bit.ly/wannik
# date: May 2015
  
  n  = len(m)
  x0 = [0] * n if x0 == None else x0
  x1 = x0[:]
#  print("x1, x2, x3, iter")
  for iter in range(max_iteration):
    for i in range(n):
      s = sum(-m[i][j] * x1[j] for j in range(n) if i != j) 
      x1[i] = (m[i][n] + s) / m[i][i]
    print(x1, iter)
    if all(abs(x1[i]-x0[i]) < eps for i in range(n)):
      return x1
    x0 = x1[:]    
  raise ValueError('Solution does not converge')


## Example
  
import numpy as np

a = np.array([[ 7, -1, 4, 10], \
[ 1, 5, -2, 4], \
[-2, 0, 8, 6]], dtype=float)
    
a1 = np.array([[ 7, -1, 4, 10], \
[-2, 0, 8, 6], \
[ 1, 5, -2, 4]], dtype=float)
    
a0 = np.array([[ 7, -1, 4], \
[-2, 0, 8], \
[ 1, 5, -2]], dtype=float)
b0 = np.array([ 10.0,6.0, 4.0])
  