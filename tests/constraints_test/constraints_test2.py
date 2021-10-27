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

def beta_c_aux(beta_c, x_min, alpha):
    beta_c_aux = np.zeros(beta_c.shape)
    beta_c_aux[0] = (beta_c[0] -x_min[0])/alpha[0]
    beta_c_aux[1] = (beta_c[1] -3*beta_c[0]*x_min[0]**2+x_min[0]**3)/(alpha[0]**3)
    return beta_c_aux

def D_x_g_c(x): #x can be a column vector or a matrix
    D = np.zeros((x.shape[0],2))
    D[0,0] = 1
    D[1,1] = 1

    return D