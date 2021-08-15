# -*- coding: utf-8 -*-
#JGA
import numpy as np
import random
from math import pi, sqrt
import PLoM_library as plom
import matplotlib.pyplot as plt
#export DISPLAY=localhost:0.0
from ctypes import *

z_init.generator(z_init, y_init, a, n_mc, x_mean, eta, s_v, hat_s_v , mu, phi, g, psi = 0, lambda_i = 0, g_c = 0)


class Data():
    def __init__(self, Xvalues, constraints = ''):
        self._Xvalues = LoadData(Xvalues) #matrix of data points
        self._npoints = Xvalues.shape[1]
        self._dimensions = Xvalues.shape[0]
        self._constraints = AddConstraints(constraints)

    def AddConstraints(NewConstraints):
        self._constraints = NewConstraints

    def LoadData(Xvalues):
        #read .csv/.txt/.mat ...

        self._Xvalues = X

    def RunAlgorithm():
        #...

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
    	#...output plots...
        
