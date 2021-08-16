# -*- coding: utf-8 -*-
#JGA
import numpy as np
import random
from math import pi, sqrt
import PLoM_library as plom
import matplotlib.pyplot as plt
#export DISPLAY=localhost:0.0
from ctypes import *

class Data():
    def __init__(self, Xvalues, constraints = ''):
        self._Xvalues = LoadData(Xvalues) #matrix of data points
        self._npoints = Xvalues.shape[1]
        self._dimensions = Xvalues.shape[0]
        self._constraints = AddConstraints(constraints)
        self._x_mean = plom.mean(Xvalues)

    def AddConstraints(NewConstraints):
        self._constraints = NewConstraints

    def LoadData(Xvalues):
        #read .csv/.txt/.mat ...

        self._Xvalues = X

    def RunAlgorithm():
        #...

        #no constraints
        Hnewvalues, nu_lambda, x_, x_2 = plom.generator(z_init, y_init, a,\
                                    n_mc, x_mean, eta, s_v, hat_s_v, mu, phi, g[:,0:m],  psi,\
                                    lambda_i, g_c) #solve the ISDE in n_mc iterations
        Xnewvalues = x_mean + phi.dot(np.diag(mu)).dot(Hnewvalues)

    def Hreduction():
        #...PCA...
        tol = 1e-9
        (self._Hvalues, self._mu, self._phi) = plom.PCA(Xvalues, tol)
        self._nu = len(eta)

    def DiffMaps():
        #..diff maps basis...
        epsilon = 16
        K, b = plom.K(Hvalues,epsilon)
        g, eigenvalues = plom.g(K,b) #diffusion maps
        eigenvalues = eigenvalues
        m = plom.m(eigenvalues)
        a = g[:,0:m].dot(np.linalg.inv(np.transpose(g[:,0:m]).dot(g[:,0:m])))
        self._Zvalues = Hvalues.dot(a)
        
    def PostProcess():
    	#...output plots..




    	#plot some histograms
        import matplotlib.patches as mpatches
        plt.plot(Xnewvalues[0,:], Xnewvalues[1,:], 'rx')
        plt.plot(Xvalues[0], Xvalues[1], 'bo')
        plt.xlabel('x')
        plt.ylabel('y')
        red_patch = mpatches.Patch(color='red', label='X_c')
        blue_patch = mpatches.Patch(color='blue', label='X')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()

        import matplotlib.patches as mpatches
        plt.xlabel('x1')
        plt.subplot(2,1,1)
        plt.hist(Xnewvalues[0], bins = 100, color = 'red')
        plt.subplot(2,1,2)
        plt.hist(Xvalues[0], bins = 100, color = 'blue')
        plt.title('Histogram')
        red_patch = mpatches.Patch(color='red', label='X_c')
        blue_patch = mpatches.Patch(color='blue', label='X')
        plt.legend(handles=[red_patch, blue_patch])
        plt.shlow()

        import matplotlib.patches as mpatches
        plt.xlabel('x2')
        plt.subplot(2,1,1)
        plt.hist(Xnewvalues[1], bins = 100, color = 'red')
        plt.subplot(2,1,2)
        plt.hist(Xvalues[1], bins = 100, color = 'blue')
        plt.title('Histogram')
        red_patch = mpatches.Patch(color='red', label='X new realizations')
        blue_patch = mpatches.Patch(color='blue', label='X')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()

        import matplotlib.patches as mpatches
        plt.xlabel('x3')
        plt.subplot(2,1,1)
        plt.hist(Xnewvalues[2], bins = 100, color = 'red')
        plt.subplot(2,1,2)
        plt.hist(Xvalues[2], bins = 100, color = 'blue')
        plt.title('Histogram')
        red_patch = mpatches.Patch(color='red', label='X new realizations')
        blue_patch = mpatches.Patch(color='blue', label='X')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()

        import matplotlib.patches as mpatches
        plt.xlabel('x4')
        plt.subplot(2,1,1)
        plt.hist(Xnewvalues[3], bins = 100, color = 'red')
        plt.subplot(2,1,2)
        plt.hist(Xvalues[3], bins = 100, color = 'blue')
        plt.title('Histogram')
        red_patch = mpatches.Patch(color='red', label='X new realizations')
        blue_patch = mpatches.Patch(color='blue', label='X')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()


        
