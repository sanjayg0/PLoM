# -*- coding: utf-8 -*-
#JGA
from matplotlib import pyplot as plt
import numpy as np
from scipy import integrate
from math import sqrt, exp, pi, log
import time
from ctypes import *
import os
from general import Logfile, DBServer
from multiprocessing import Pool
import defs

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

def scaling(x):
    n = x.shape[0]
    alpha = np.zeros(n)
    x_min = np.zeros((n,1))
    for i in range(0,n):
        x_max_k = max(x[i,:])
        x_min_k= min(x[i,:])
        x_min[i] = x_min_k
        if x_max_k - x_min_k != 0:
            alpha[i] = x_max_k - x_min_k
        else:
            alpha[i] = 1
    x_scaled = np.diag(1/alpha).dot(x-x_min)
    return x_scaled, alpha, x_min

def gradient_rhoctypes(gradient, y, eta, nu, N, s_v, hat_s_v):
    return c_lib.gradient_rho(np.array(gradient,np.float64),\
                              np.array(y,np.float64),\
                              np.array(eta,np.float64),\
                              nu, N, s_v, hat_s_v)

def kernel(x, y, epsilon):
    """
    >>> kernel(np.array([1,0]), np.array([1,0]), 0.5)
    1.0
    """
    dist = np.linalg.norm(x-y)**2
    k = np.exp(-dist/(4*epsilon))
    return k

def K(eta, epsilon):
    """
    >>> K((np.array([[1,1],[1,1]])), 3)
    (array([[1., 1.],
           [1., 1.]]), array([[2., 0.],
           [0., 2.]]))
    """
    # use parallel process for KDE
    # variable number of works
    import multiprocessing
    num_cpu = multiprocessing.cpu_count()
    pool = Pool(num_cpu)
    N = eta.shape[1]
    # use functions in defs.py
    results = pool.map(defs.splat_f, ((i, j, eta, epsilon) for i in range(N) for j in range(N)))
    K = np.array(results).reshape(N, N)
    b = np.diag(np.sum(K,1))
    return K, b

def g(K, b):
    """
    >>> g((np.array([[1,0.5],[0.5,1]])), np.array([[1.5, 0.], [0., 1.5]]))
    (array([[ 0.57735027, -0.57735027],
           [ 0.57735027,  0.57735027]]), array([1.        , 0.33333333]))
    """
    invb = np.diag(1/np.diag(b))
    inv_sqrt_b = np.sqrt(invb)
    xi = np.linalg.eigh(inv_sqrt_b.dot(K).dot(inv_sqrt_b))
    xi[1][:,:] = np.transpose(xi[1][:,:])
    xi[1][:,:] = xi[1][[np.argsort(xi[0], kind = 'mergesort', axis = 0)[::-1]], :]
    eigenvalues = np.sort(xi[0], kind = 'mergesort', axis = 0)[::-1]
    g = inv_sqrt_b.dot(np.transpose(xi[1][:,:]))
    norm = np.diagonal(np.transpose(g).dot(b).dot(g))
    sqrt_norm = np.sqrt(1/norm)
    g = np.multiply(g, sqrt_norm)
    return g, eigenvalues

def m(eigenvalues, beta=0.1):
    """
    >>> m(np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025]))
    11
    """
    i = 2
    m = 0
    print('beta = {}'.format(beta))
    while i < len(eigenvalues) and m == 0:
        if eigenvalues[i] <= eigenvalues[1]*beta:
            return i+1
        i = i+1
    if m == 0:
        return max(round(len(eigenvalues)/10), 3)
    return m

def mean(x):
    """
    >>> mean(np.array([[1,1],[0,1],[2,4]]))
    array([[1. ],
           [0.5],
           [3. ]])
    """
    dim = x.shape[0]
    x_mean = np.zeros((dim,1))
    for i in range(0,dim):
        x_mean[i] = np.mean(x[i,:])
    return x_mean

def covariance(x):
    """
    >>> covariance(np.array([[1,1],[0,1],[2,4]]))
    array([[0. , 0. , 0. ],
           [0. , 0.5, 1. ],
           [0. , 1. , 2. ]])
    """
    dim = x.shape[0]
    N = x.shape[1]
    C = np.zeros((dim,dim))
    x_mean = mean(x)
    for i in range(0,N):
        C = C + (np.resize(x[:,i], x_mean.shape) - x_mean).dot(np.transpose((np.resize(x[:,i], x_mean.shape) - x_mean)))
    return C/(N-1)

def PCA(x, tol):
    """
    >>> PCA(np.array([[1,1],[0,1],[2,4]]), 0.1)
    (array([[-0.70710678,  0.70710678]]), array([1.58113883]), array([[-1.13483031e-17],
           [ 4.47213595e-01],
           [ 8.94427191e-01]]))
    """
    x_mean = mean(x)
    (phi,mu,v) = np.linalg.svd(x-x_mean)
    mu = mu/sqrt(len(x[0])-1)
    #plt.figure()
    #plt.plot(np.arange(len(mu)), mu)
    #plt.xlabel('# eigenvalue of X covariance')
    #plt.show()
    error = 1
    i = 0
    errors = [1]
    while error > tol and i < len(mu):
        error = error -  (mu[i]**2)/sum((mu**2))
        i = i+1
        nu = i
        errors.append(error)
    while i < len(mu):
        error = error -  (mu[i]**2)/sum((mu**2))
        i = i+1
        errors.append(error)
    #plt.figure()
    #plt.semilogy(np.arange(len(mu)+1), errors)
    #plt.xlabel('# eigenvalue of Covariance matrix of X')
    #plt.ylabel('Error of the PCA associated with the eigenvalue')
    #plt.show()
    mu = mu[0:nu]
    phi = phi[:,0:nu]
    mu_sqrt_inv = (np.diag(1/(mu))) #no need to do the sqrt because we use the singularvalues
    eta = mu_sqrt_inv.dot(np.transpose(phi)).dot((x-x_mean))
    return eta, mu, phi #mu is the diagonal matrix with the singularvalues up to a tolerance

def parameters_kde(eta):
    """
    >>> parameters_kde(np.array([[1,1],[0,1],[2,4]]))
    (0.8773066621237415, 0.13452737030512696, 0.7785858648409519)
    """
    nu = eta.shape[0]
    N = eta.shape[1]
    s_v = (4/(N*(2+nu)))**(1/(nu+4))#(4/(N*(2+nu)))**(1/(nu+4))
    hat_s_v = s_v/sqrt(s_v**2+((N-1)/N))
    c_v = 1/(sqrt(2*pi)*hat_s_v)**nu
    return s_v, c_v, hat_s_v

def kde(y, eta, s_v = None, c_v = None, hat_s_v = None):
    """
    >>> kde(np.array([[1, 2, 3]]), np.array([[1,1],[0,1],[2,4]]))
    0.01940049487135241
    """
    nu = eta.shape[0]
    N = eta.shape[1]
    if s_v == None or c_v == None or hat_s_v == None:
        s_v, c_v, hat_s_v = parameters_kde(eta)
    return c_v*rhoctypes(np.resize(y,(y.shape[0]*y.shape[1],1)), np.resize(np.transpose(eta),(nu*N,1)),\
        nu, N, s_v, hat_s_v)

def PCA2(C_h_hat_eta, beta, tol): #taking only independent constraints
    """
    >>> PCA2(np.array([[1. , 1. , 1. ], [1. , 4.5, 1.5 ], [1. , 1.5 , 2. ]]), np.array([10, 1, 2]), 0.1)
    (array([-4.53648062,  5.2236145 ]), array([[-0.28104828,  0.42570005],
           [-0.85525695, -0.51768266],
           [-0.43537043,  0.74214832]]))
    """
    (lambda_c, psi) = np.linalg.eig(C_h_hat_eta) #eigenvalue decomposition as the dimensions are not so big
    psi = np.transpose(psi)
    psi = psi[np.argsort(lambda_c, kind = 'mergesort', axis = 0)[::-1], :]
    psi = np.transpose(psi)
    lambda_c = np.sort(lambda_c, kind = 'mergesort', axis = 0)[::-1]
    i = 1
    nu_c = 1
    while i < len(lambda_c) and not(lambda_c[i-1] > tol*lambda_c[0] and lambda_c[i] <= tol*lambda_c[0]):
        i = i+1
        nu_c = i
    lambda_c = lambda_c[0:nu_c]
    psi = psi[:, 0:nu_c]
    b_c = np.transpose(psi).dot(beta)
    return b_c, psi

def h_c(eta, g_c, phi, mu, psi, x_mean):
    return np.transpose(psi).dot(g_c(x_mean +phi.dot(np.diag(mu)).dot(eta) ))

def gradient_gamma(b_c, eta_lambda, g_c, phi, mu, psi, x_mean):
    return (b_c) - mean(h_c(eta_lambda, g_c, phi, mu, psi, x_mean)) #the mean is the empirical expectation

def hessian_gamma(eta_lambda, psi, g_c, phi, mu, x_mean):
    return covariance(h_c(eta_lambda, g_c, phi, mu, psi, x_mean))

def solve_inverse(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        return Logfile().write_msg(msg='PLoM: solve_inverse non-square matrix.',msg_type='ERROR',msg_level=0)
    else:
        inverse = np.zeros(matrix.shape)
        for j in range(0,matrix.shape[1]):
            unit = np.zeros(matrix.shape[1])
            unit[j] = 1
            solve = np.linalg.solve(matrix, unit)
            inverse[:,j] = solve
        return inverse


def generator(z_init, y_init, a, n_mc, x_mean, eta, s_v, hat_s_v, mu, phi, g, psi = 0, lambda_i = 0, g_c = 0, D_x_g_c = 0, seed_num=None):
    if seed_num:
        np.random.seed(seed_num)
    delta_t = 2*pi*hat_s_v/20
    #print('delta t: ', delta_t)
    f_0 = 1.5
    l_0 = 10#200
    M_0 = 10#20
    beta = f_0*delta_t/4
    nu = z_init.shape[0]
    N = a.shape[0]
    eta_lambda = np.zeros((nu,(n_mc+1)*N))
    nu_lambda = np.zeros((nu,(n_mc+1)*N))
    n = x_mean.shape[0]
    x_ = np.zeros((n,n_mc))
    x_2 = np.zeros((n,n_mc))
    z_l = z_init
    y_l = y_init
    eta_lambda[:,0:N] = z_init.dot(np.transpose(g))
    nu_lambda[:,0:N] = y_init.dot(np.transpose(g))
    for i in range (0,l_0):
        z_l_half = z_l + delta_t*0.5*y_l
        w_l_1 = np.random.normal(scale = sqrt(delta_t), size = (nu,N)).dot(a) #wiener process
        L_l_half = L(z_l_half.dot(np.transpose(g)), g_c, x_mean, eta, s_v, hat_s_v, mu, phi, psi, lambda_i, D_x_g_c).dot(a)
        y_l_1 = (1-beta)*y_l/(1+beta) + delta_t*(L_l_half)/(1+beta) + sqrt(f_0)*w_l_1/(1+beta)
        z_l = z_l_half + delta_t*0.5*y_l_1
        y_l = y_l_1
    for l in range(M_0, M_0*(n_mc+1)):
        z_l_half = z_l + delta_t*0.5*y_l
        w_l_1 = np.random.normal(scale = sqrt(delta_t), size = (nu,N)).dot(a) #wiener process
        L_l_half = L(z_l_half.dot(np.transpose(g)), g_c, x_mean, eta, s_v, hat_s_v, mu, phi, psi, lambda_i, D_x_g_c).dot(a)
        y_l_1 = (1-beta)*y_l/(1+beta) + delta_t*(L_l_half)/(1+beta) + sqrt(f_0)*w_l_1/(1+beta)
        z_l = z_l_half + delta_t*0.5*y_l_1
        y_l = y_l_1
        if l%M_0 == M_0-1:
            eta_lambda[:,int(l/M_0)*N:(int(l/M_0)+1)*N] = z_l.dot(np.transpose(g))
            nu_lambda[:,int(l/M_0)*N:(int(l/M_0)+1)*N] = y_l.dot(np.transpose(g))
            x_[:,int(l/M_0)-1:int(l/M_0)] = mean(x_mean + phi.dot(np.diag(mu)).dot(eta_lambda[:,:(int(l/M_0)+1)*N]))
            x_2[:,int(l/M_0)-1:int(l/M_0)] = mean((x_mean + phi.dot(np.diag(mu)).dot(eta_lambda[:,:(int(l/M_0)+1)*N]))**2)
    return eta_lambda[:,N:], nu_lambda[:,N:], x_, x_2

def ac(sig):
    sig = sig - np.mean(sig)
    sft = np.fft.rfft( np.concatenate((sig,0*sig)) )
    return np.fft.irfft(np.conj(sft)*sft)

def L(y, g_c, x_mean, eta, s_v, hat_s_v, mu, phi, psi, lambda_i, D_x_g_c): #gradient of the potential
    nu = eta.shape[0]
    N = eta.shape[1]
    L = np.zeros((nu,N))

    if False:
        for l in range(0,N):
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
                #KZ L[:,l] = (  np.resize(vector/(hat_s_v**2),(nu))\
                #    -np.resize(np.diag(mu).dot(np.transpose(phi)).\
                #            dot(D_x_g_c(x_mean+np.resize(phi.dot(np.diag(mu)).dot(yl), (x_mean.shape)))).\
                #            dot(psi).dot(lambda_i), (nu)))
                L[:,l] = (  np.resize(vector/(hat_s_v**2),(nu))\
                    -np.resize(np.diag(mu).dot(np.transpose(phi)).\
                            dot(grad_g_c).dot(psi).dot(lambda_i), (nu)))

            else:
                array_pointer = cast(gradient_rhoctypes(np.zeros((nu,1)),yl,\
                    np.resize(np.transpose(eta),(nu*N,1)), nu, N, s_v, hat_s_v), POINTER(c_double*nu))
                gradient_rho = np.frombuffer(array_pointer.contents)
                #KZ L[:,l] = np.resize(1e250*gradient_rho/rho_,(nu))\
                #        -np.resize(np.diag(mu).dot(np.transpose(phi)).\
                #                dot(D_x_g_c(x_mean+np.resize(phi.dot(np.diag(mu)).dot(yl), (x_mean.shape)))).\
                #                    dot(psi).dot(lambda_i), (nu))
                L[:,l] = np.resize(1e250*gradient_rho/rho_,(nu))\
                        -np.resize(np.diag(mu).dot(np.transpose(phi)).\
                                dot(grad_g_c).dot(psi).dot(lambda_i), (nu))
    if True:
        import multiprocessing
        num_cpu = multiprocessing.cpu_count()
        pool = Pool(num_cpu)
        results = pool.map(defs.splat_h, ((l, y, x_mean, eta, s_v, hat_s_v, mu, phi, psi, lambda_i, D_x_g_c, nu, N) for l in range(N)))
        L = np.array(results).reshape(N,nu).transpose()
        
    return L


def err(gradient, b_c):
    return np.linalg.norm(gradient)/np.linalg.norm(b_c)

def gamma(lambda_i, eta, s_v, hat_s_v, g_c, phi, mu, psi, x_mean, b_c):
    return np.transpose(lambda_i).dot(b_c)\
        + log(inv_c_0(lambda_i, eta, s_v, hat_s_v, g_c, phi, mu, psi, x_mean))

def func(x,y, eta, s_v, hat_s_v, g_c, phi, mu, psi, x_mean, lambda_i):
    nu = eta.shape[0]
    N = eta.shape[1]
    return rhoctypes(np.array([x,y]), np.resize(np.transpose(eta),(nu*N,1)),\
        nu, N, s_v, hat_s_v)*\
                exp(-np.transpose(lambda_i).dot(h_c(np.array([[x],[y]]), g_c, phi, mu, psi, x_mean)))

def gaussian_bell(x,y):
    return exp(-(x**2+y**2)/2)/(2*pi)

def inv_c_0(lambda_i, eta, s_v, hat_s_v, g_c, phi, mu, psi, x_mean):
    c,error = integrate.dblquad(func,\
                            -3, 3, -3, 3, args=(eta, s_v, hat_s_v, g_c, phi, mu, psi, x_mean, lambda_i))
    return c #integral mathematica

def expo(y):
    meann = np.array([[0],[0]])
    sigma = np.array([[1, 0],[0, 1]])
    f = exp(-0.5*np.transpose(y-meann).dot(y-meann))
    return f

def gradient_expo(y):
    meann = np.array([[0],[0]])
    sigma = np.array([[1, 0],[0, 1]])
    f = np.zeros((2,1))
    f = -(y-meann)*exp(-0.5*np.transpose(y-meann).dot(y-meann))
    return f

if __name__ == "__main__":
    import doctest
    doctest.testmod()
