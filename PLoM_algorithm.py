#!/usr/bin/env python
# coding: utf-8

# # Probabilistic learning on Manifolds

# Initial data set and initial constraints:

# In[1]:


import numpy as np
import random
from math import pi, sqrt
import PLoM_library as plom
import matplotlib.pyplot as plt
#export DISPLAY=localhost:0.0
from ctypes import *

#read values of the constraints
beta_c = np.array([[10], [150]]) #np.array([[0.49120901], [0.35532349]])
m_c = beta_c.shape[0]

#read x points in columns
#x = ...
N = 300
x = np.zeros((20,N))
np.random.seed(6)
r = np.random.weibull(3, N)+10*np.ones((1,N))
angle = np.random.uniform(0, 2*pi, N)
x[0] = r*np.sin(angle)+ 10*np.ones((1,N))
x[1] = r*np.cos(angle)
x[2] = r*r*np.cos(angle*angle)
x[3] = np.random.rayleigh(0.5, N)
x[4] = -2*x[3] + np.random.uniform(-0.1, 0.1, N)
for i in range(5,20):
    x[i] = ((-1)**2)*i*x[3]+ np.random.uniform(-0.1, 0.1, N)

# read x from a '.dat' file
#x = []
#with open('./data for batteries/plmdata_big.dat', 'r') as f:
    #d = f.readlines()
    #for i in d:
        #k = i.rstrip().split(" ")
        #x.append([float(i) for i in k])

x = (np.array(x, dtype='float'))

# plot of the two first components
plt.plot(x[0], x[1], 'bo')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('Slice of the Initial data set (x)')
plt.show()


#histogram of the first nine components
plt.xlabel('x1')
plt.subplot(3,3,1)
plt.hist(x[0], bins = 100, color = 'blue')
plt.title('Histogram x_1')
plt.subplot(3,3,2)
plt.hist(x[1], bins = 100, color = 'blue')
plt.title('Histogram x_2')
plt.subplot(3,3,3)
plt.hist(x[2], bins = 100, color = 'blue')
plt.title('Histogram x_3')
plt.subplot(3,3,4)
plt.hist(x[3], bins = 100, color = 'blue')
plt.title('Histogram x_4')
plt.subplot(3,3,5)
plt.hist(x[4], bins = 100, color = 'blue')
plt.title('Histogram x_5')
#plt.subplot(3,3,6)
#plt.hist(x[5], bins = 100, color = 'blue')
#plt.title('Histogram x_6')
#plt.subplot(3,3,7)
#plt.hist(x[6], bins = 100, color = 'blue')
#plt.title('Histogram x_7')
#plt.subplot(3,3,8)
#plt.hist(x[7], bins = 100, color = 'blue')
#plt.title('Histogram x_8')
#plt.subplot(3,3,9)
#plt.hist(x[8], bins = 100, color = 'blue')
#plt.title('Histogram x_9')
plt.show()

#definition of the function of the constraints
def g_c(x): #x can be a column vector or a matrix
    f = np.zeros((2, x.shape[1]))
    f[0,:] = x[0,:]
    # f[1,:] = x[10,:]
    f[1,:] = x[0,:]**2
    return f

# evaluation of g_c with the initial data
print('g_c: ', plom.mean(g_c(x)))

#scaling of x
x, alpha, x_min = plom.scaling(x)

# evaluation of g_c with the scaled initial data
print('g_c scaled: ', plom.mean(g_c(x)))

#scale the value of betas (it deppends on if it is a mean value or if it is a second moment)
print('alpha5', (beta_c[0] -x_min[0])/alpha[0])
beta_c = [[float((beta_c[0] -x_min[0])/alpha[0])],[float((beta_c[1] -2*beta_c[0]*x_min[0]+x_min[0]**2)/(alpha[0]**2))]]

#function to unscale x
def unscale(x):
    return np.diag(alpha).dot(x)+x_min


# definition of basic constants
x_mean = plom.mean(x)
N = x.shape[1] #initial number of points
n = x.shape[0] #initial dimension

print('N: ', N)
print('n: ', n)

# 1. PCA
tol = 1e-9#tolerance
(eta, mu, phi) = plom.PCA(x, tol)
nu = len(eta)
print('nu: ', nu)

plt.plot(eta[0], eta[1], 'ro')
#plt.plot(x[0], x[1], 'bo')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.plot(x[0], x[1], 'bo')
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.title('Slice of the Initial data set (x)')
plt.show()

plom.covariance(eta)

# 2. KDE

(s_v, c_v, hat_s_v) = plom.parameters_kde(eta)

plom.kde(np.array([1,0]), eta, s_v, c_v, hat_s_v)


# 3. Reformulation to $\mathbf{H}^c$. Reduction of constraints

C_h_hat_eta = plom.covariance(g_c(x_mean+phi.dot(np.diag(mu)).dot(eta)))
tol = 1e-5

b_c, psi = plom.PCA2(C_h_hat_eta, beta_c, tol)
nu_c = len(b_c)

print('Psi: ', psi)

# 5. Reformulation introducing random vector $\mathbf{H}_\lambda$#calculation of lambda_0
inverse = np.zeros((nu_c,nu_c))
hessian = plom.hessian_gamma(eta, psi, g_c, phi, mu, x_mean)
for j in range(0,nu_c):
    unit = np.zeros(nu_c)
    unit[j] = 1
    solve = np.linalg.solve(hessian, unit)
    inverse[:,j] = solve

lambda_i = -(inverse)\
            .dot(plom.gradient_gamma(b_c, eta, g_c, phi, mu, psi, x_mean))

print('Gradient of gamma: (should be zero)', plom.gradient_gamma(b_c, eta, g_c, phi, mu, psi, x_mean))
print('Hessian: ', hessian)
print('inv Hessian: ', inverse)
print('Lambda: ', lambda_i)


# commented section to calculate m (where the function stps decreasing)
###############################################################################
# m_list = []
# for epsilon in range(1,30,1):#method to calculate epsilon
#     print(epsilon)
#     K, b = plom.K(eta,epsilon)
#
#     g, eigenvalues = plom.g(K,b) #diffusion maps
#     g = g
#     eigenvalues = eigenvalues
#     m = plom.m(eigenvalues)
#     #print('m: ', m)
#     m_list.append(m)
#     #print(eigenvalues[2])
#     a = g[:,0:m].dot(np.linalg.inv(np.transpose(g[:,0:m]).dot(g[:,0:m])))
#
# plt.plot(np.arange(1,30,1), m_list)
# plt.xlabel('m')
# plt.show()
# plt.semilogy(np.arange(1,30,1), m_list)
# plt.xlabel('m')
# plt.show()
###############################################################################

#calculate diffusion maps basis
epsilon = 16
K, b = plom.K(eta,epsilon)

g, eigenvalues = plom.g(K,b) #diffusion maps
g = g
eigenvalues = eigenvalues
m = plom.m(eigenvalues)
print('m: ', m)

print(eigenvalues[1])
a = g[:,0:m].dot(np.linalg.inv(np.transpose(g[:,0:m]).dot(g[:,0:m])))

plt.semilogy(np.arange(len(eigenvalues)), eigenvalues)
plt.xlabel('# eigenvalue of transition matrix K')
plt.show()



#ISDE and Newton method
eta_init = eta
nu_init = np.random.normal(size = (nu,N))


z_init = eta_init.dot(a)
y_init = nu_init.dot(a)
gradient = plom.gradient_gamma(b_c, eta_init, g_c, phi, mu, psi, x_mean)

n_mc = 10#adjust the number of new matrices
i = 0

errors = [plom.err(gradient,b_c)]

tol = 1e-6

while (i < 50 and errors[i] > tol):#tol:
    eta_lambda, nu_lambda, x_, x_2 = plom.generator(z_init, y_init, a,\
                                    n_mc, x_mean, eta, s_v, hat_s_v, mu, phi, g[:,0:m],  psi,\
                                    lambda_i, g_c) #solve the ISDE in n_mc iterations


    gradient = plom.gradient_gamma(b_c, eta_lambda, g_c, phi, mu, psi, x_mean)
    #solve the system not inverse
    inverse = np.zeros((nu_c,nu_c))
    hessian = plom.hessian_gamma(eta_lambda, psi, g_c, phi, mu, x_mean)
    for j in range(0,nu_c):
        unit = np.zeros(nu_c)
        unit[j] = 1
        solve = np.linalg.solve(hessian, unit)
        inverse[:,j] = solve
    lambda_i = lambda_i - 0.3*(inverse).dot(gradient)
    z_init = eta_lambda[:,-N:].dot(a)
    y_init = nu_lambda[:,-N:].dot(a)
    i = i+1
    print("Iteration #", i)
    print("Error: ", plom.err(gradient,b_c))
    print('inv Hessian Lambda: ', inverse)
    print('Gradient lambda: ', gradient)
    print("Lambda:", lambda_i)
    errors.append(plom.err(gradient,b_c))
    x_c = x_mean + phi.dot(np.diag(mu)).dot(eta_lambda)
    print('g_c(x_lambda): ', plom.mean(g_c(x_c)))


    # if i == 30:
    #     plt.plot(np.arange(i+1), errors[:i+1])
    #     plt.xlabel('# iteration')
    #     plt.ylabel('error')
    #     plt.show()
    #
    #     plt.semilogy(np.arange(i+1), errors[:i+1])
    #     plt.xlabel('# iteration')
    #     plt.ylabel('error')
    #     plt.show()

plt.plot(x_c[0,:], x_c[1,:], 'rx')
plt.plot(x[0], x[1], 'bo')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

plt.subplot(2,2,1)
plt.plot(x_[0,:])
plt.ylabel('Mean of x[0]',fontsize=16)

plt.subplot(2,2,2)
plt.plot(x_[1,:])
plt.ylabel('Mean of x[1]',fontsize=16)

plt.subplot(2,2,3)
chi = plom.ac(x_[0,:(n_mc//2)])
plt.plot(chi[:chi.size]/chi[0])
plt.ylabel(r'$\chi_x(t)$',fontsize=16)
# plt.xlabel('MC Sweeps',fontsize=16)

plt.subplot(2,2,4)
chi = plom.ac(x_[1,:(n_mc//2)])
plt.plot(chi[:chi.size]/chi[0])
plt.ylabel(r'$\chi_x^{2}(t)$',fontsize=16)
# plt.xlabel('MC Sweeps',fontsize=16)
plt.show()

# plot of errors
plt.plot(np.arange(i+1), errors[:i+1])
plt.xlabel('# iteration')
plt.ylabel('error')
plt.show()

plt.semilogy(np.arange(i+1), errors[:i+1])
plt.xlabel('# iteration')
plt.ylabel('error')
plt.show()

#undo scaling
x_c = x_mean + phi.dot(np.diag(mu)).dot(eta_lambda)
x_c = np.diag(alpha).dot(x_c)+x_min
x = np.diag(alpha).dot(x)+x_min
plom.mean(x_c[:,:])
plom.mean(g_c(x_c))


#plot some histograms
import matplotlib.patches as mpatches
plt.plot(x_c[0,:], x_c[1,:], 'rx')
plt.plot(x[0], x[1], 'bo')
plt.xlabel('x')
plt.ylabel('y')
red_patch = mpatches.Patch(color='red', label='X_c')
blue_patch = mpatches.Patch(color='blue', label='X')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

import matplotlib.patches as mpatches
plt.xlabel('x1')
plt.subplot(2,1,1)
plt.hist(x_c[0], bins = 100, color = 'red')
plt.subplot(2,1,2)
plt.hist(x[0], bins = 100, color = 'blue')
plt.title('Histogram')
red_patch = mpatches.Patch(color='red', label='X_c')
blue_patch = mpatches.Patch(color='blue', label='X')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

import matplotlib.patches as mpatches
plt.xlabel('x2')
plt.subplot(2,1,1)
plt.hist(x_c[1], bins = 100, color = 'red')
plt.subplot(2,1,2)
plt.hist(x[1], bins = 100, color = 'blue')
plt.title('Histogram')
red_patch = mpatches.Patch(color='red', label='X new realizations')
blue_patch = mpatches.Patch(color='blue', label='X')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

import matplotlib.patches as mpatches
plt.xlabel('x3')
plt.subplot(2,1,1)
plt.hist(x_c[2], bins = 100, color = 'red')
plt.subplot(2,1,2)
plt.hist(x[2], bins = 100, color = 'blue')
plt.title('Histogram')
red_patch = mpatches.Patch(color='red', label='X new realizations')
blue_patch = mpatches.Patch(color='blue', label='X')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

import matplotlib.patches as mpatches
plt.xlabel('x4')
plt.subplot(2,1,1)
plt.hist(x_c[3], bins = 100, color = 'red')
plt.subplot(2,1,2)
plt.hist(x[3], bins = 100, color = 'blue')
plt.title('Histogram')
red_patch = mpatches.Patch(color='red', label='X new realizations')
blue_patch = mpatches.Patch(color='blue', label='X')
plt.legend(handles=[red_patch, blue_patch])
plt.show()

#import matplotlib.patches as mpatches
#plt.subplot(2,1,1)
#plt.hist(x_c[18], bins = 100, color = 'red')
#plt.subplot(2,1,2)
#plt.hist(x[18], bins = 100, color = 'blue')
#plt.title('Histogram of f(A)')
#red_patch = mpatches.Patch(color='red', label='X new realizations')
#blue_patch = mpatches.Patch(color='blue', label='X')
#plt.legend(handles=[red_patch, blue_patch])
#plt.show()

#import matplotlib.patches as mpatches
#plt.xlabel('f($A_B^1$)')
#plt.subplot(2,1,1)
#plt.hist(x_c[20], bins = 100, color = 'red')
#plt.subplot(2,1,2)
#plt.hist(x[20], bins = 100, color = 'blue')
#plt.title('Histogram of f($A_B^1$)')
#red_patch = mpatches.Patch(color='red', label='X new realizations')
#blue_patch = mpatches.Patch(color='blue', label='X')
#plt.legend(handles=[red_patch, blue_patch])
#plt.show()

#import matplotlib.patches as mpatches
#plt.xlabel('$A^5$')
#plt.subplot(2,1,1)
#plt.hist(x_c[5], bins = 100, color = 'red')
#plt.subplot(2,1,2)
#plt.hist(x[5], bins = 100, color = 'blue')
#plt.title('Histogram of $A^5$')
#red_patch = mpatches.Patch(color='red', label='X new realizations')
#blue_patch = mpatches.Patch(color='blue', label='X')
#plt.legend(handles=[red_patch, blue_patch])
#plt.show()



#calculate and plot the pdfs in R^2
# points = 100
#
# x1 = np.linspace(-2, 2, points)
# x2 = np.linspace(-2, 2, points)
#
# p = np.zeros((points,points))
# p_c = np.zeros((points,points))
#
# for i in range(0, points):
#     for j in range(0,points):
#         p[i,j] = plom.kde(np.array([[x1[i]],[x2[j]]]), eta, s_v, c_v, hat_s_v)
#         p_c[i,j] = plom.kde(np.array([[x1[i]],[x2[j]]]), eta_lambda)
#
# plt.contour(x1, x2, p)
# plt.colorbar()
# plt.show()
#
# plt.contour(x1, x2, p_c)
# plt.colorbar()
# plt.show()


# In[ ]: Countor plot gamma (topographical map) in R^2

# points = 10
#
# lambdas1 = np.linspace(-0.1, 0.1, points)
# lambdas2 = np.linspace(-0.1, 0.1, points)
#
# gamma = np.zeros((points,points))

# gradient_comp_zero = np.zeros((points,points))
# gradient_comp_one = np.zeros((points,points))
#
#
# for i in range(0, points):
#     for j in range(0,points):
#         # z_init.shape
#         # z_lambda, eta_i_j, y_lambda, nu_lambda = plom.generator(z_init, y_init, a, n_mc, x_mean, eta, s_v,\
#         #                          hat_s_v, mu, phi, g[:,0:m],  psi, np.array(lambdas1[i],lambdas2[j]), g_c)
#         # k=plom.gradient_gamma(b_c, eta_i_j, g_c, phi, mu, psi, x_mean)
#         # gradient_comp_zero[i,j] = k[0]
#         # gradient_comp_one[i,j] = k[1]
#         gamma[i,j] = plom.gamma(np.array([[lambdas1[i]],[lambdas2[j]]]), eta, s_v, hat_s_v, g_c, phi, mu, psi, x_mean, b_c)
#
#
# plt.contour(lambdas1, lambdas2, gamma)
# plt.colorbar()

# plt.contour(lambdas1, lambdas2, gradient_comp_zero)
# plt.colorbar()

# plt.contour(lambdas1, lambdas2, gradient_comp_one)
# plt.colorbar()

# plt.quiver(lambdas1, lambdas2, gradient_comp_zero, gradient_comp_one)
#look at the code
#plom no constraints, done
#countor plot of function gamma
