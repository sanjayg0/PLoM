import numpy as np
import os
from ctypes import *
from sys import platform
if platform == "linux" or platform == "linux2":
    c_lib = CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),"lib/linux/PLoM_C_library.so"))
elif platform == "darwin":
    c_lib = CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),"lib/macOS/PLoM_C_library.so"))
elif platform == "win32":
    c_lib = CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),"lib/win/PLoM_C_library.so"))

c_lib.rho.restype = c_double
c_lib.rho.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                        np.ctypeslib.ndpointer(dtype=np.float64),c_int,c_int,c_double,c_double]

c_lib.gradient_rho.restype = np.ctypeslib.ndpointer(dtype=np.float64)
c_lib.gradient_rho.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                               np.ctypeslib.ndpointer(dtype=np.float64),
                               np.ctypeslib.ndpointer(dtype=np.float64),
                               c_int,c_int,c_double,c_double]

def rhoctypes(y, eta, nu, N, s_v, hat_s_v):
    return c_lib.rho(np.array(y,np.float64),np.array(eta,np.float64),nu,N,s_v,hat_s_v)

def gradient_rhoctypes(gradient, y, eta, nu, N, s_v, hat_s_v):
    return c_lib.gradient_rho(np.array(gradient,np.float64),\
                              np.array(y,np.float64),\
                              np.array(eta,np.float64),\
                              nu, N, s_v, hat_s_v)

def splat_f(args):
    return f(*args)

def kernel(x, y, epsilon):
    """
    >>> kernel(np.array([1,0]), np.array([1,0]), 0.5)
    1.0
    """
    dist = np.linalg.norm(x-y)**2
    k = np.exp(-dist/(4*epsilon))
    return k

def f(i,j,eta,epsilon):
    if j != i:
        K = kernel((eta[:,i]),((eta[:,j])), epsilon)
    else:
        K = 1
    return K

def splat_h(args):
    return h(*args)

def h(l, y, x_mean, eta, s_v, hat_s_v, mu, phi, psi, lambda_i, D_x_g_c, nu, N):
    yl = np.resize(y[:,l],(len(y[:,l]),1))
    rho_ = rhoctypes(yl, np.resize(np.transpose(eta),(nu*N,1)),\
             nu, N, s_v, hat_s_v)
    rho_ = 1e250*rho_
    # compute the D_x_g_c if D_x_g_c is not 0 (KZ)
    if D_x_g_c:
        grad_g_c = D_x_g_c(x_mean+np.resize(phi.dot(np.diag(mu)).dot(yl), (x_mean.shape)))
    else:
        # not constraints and no D_x_g_c
        grad_g_c = np.zeros((x_mean.shape[0],1))
    if rho_ < 1e-250:
        closest = np.inf
        for i in range(0,N):
            if closest > np.linalg.norm((hat_s_v/s_v)*np.resize(eta[:,i],yl.shape)-yl):
                closest = np.linalg.norm((hat_s_v/s_v)*np.resize(eta[:,i],yl.shape)-yl)
                vector = (hat_s_v/s_v)*np.resize(eta[:,i],yl.shape)-yl
        L = (  np.resize(vector/(hat_s_v**2),(nu))\
            -np.resize(np.diag(mu).dot(np.transpose(phi)).\
            dot(grad_g_c).dot(psi).dot(lambda_i), (nu)))

    else:
        array_pointer = cast(gradient_rhoctypes(np.zeros((nu,1)),yl,\
            np.resize(np.transpose(eta),(nu*N,1)), nu, N, s_v, hat_s_v), POINTER(c_double*nu))
        gradient_rho = np.frombuffer(array_pointer.contents)
        L = np.resize(1e250*gradient_rho/rho_,(nu))\
            -np.resize(np.diag(mu).dot(np.transpose(phi)).\
            dot(grad_g_c).dot(psi).dot(lambda_i), (nu))
    return L