import numpy as np

#definition of the function of the constraints
def g_c(x): #x can be a column vector or a matrix
    f = np.zeros((2, x.shape[1]))
    f[0,:] = x[0,:]
    # f[1,:] = x[10,:]
    f[1,:] = x[0,:]**3
    return f

#returning the beta_c values
def beta_c():
    beta_c = np.array([[20], [150]])
    return beta_c