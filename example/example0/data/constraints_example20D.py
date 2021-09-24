import numpy as np

#definition of the function of the constraints
def g_c(x): #x can be a column vector or a matrix
    f = np.zeros((2, x.shape[1]))
    f[0,:] = x[0,:]
    f[1,:] = x[1,:]
    return f

#returning the beta_c values
def beta_c():
    beta_c = np.array([[10],[2]])
    return beta_c

def D_x_g_c(x): #x can be a column vector or a matrix
    D = np.zeros((x.shape[0],2))
    D[0,0] = 1
    D[1,1] = 1

    return D
