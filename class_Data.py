# -*- coding: utf-8 -*-
#JGA
import numpy as np
import random
from math import pi, sqrt
import PLoM_library as plom
import matplotlib.pyplot as plt
#export DISPLAY=localhost:0.0
from ctypes import *


class PLoM:
    def __init__(self, data='', seperator=',', col_header=False, constraints = ''):
        self.initialize_data(data, seperator, col_header)
        #self._npoints = Xvalues.shape[1]
        #self._dimensions = Xvalues.shape[0]
        #self._constraints = AddConstraints(constraints)

    def AddConstraints(self, NewConstraints):
        self._constraints = NewConstraints

    def load_data(self, filename, seperator=',', col_header=False):

        # initialize the matrix and data size
        X = []
        N = 0
        n = 0

        # check if the file exist
        import os
        if not os.path.exists(filename):
            print('load_data: Error - the input file {} is not found'.format(filename))
            return X, N, NewConstraints
        
        # read data
        if filename.split('.')[-1] in ['csv','dat','txt']:
            print(filename)
            # txt data
            import pandas as pd
            col = None
            if col_header:
                col = 0
            tmp = pd.read_table(filename, delimiter=seperator, header=col)
            print(tmp)
            # remove all-nan column if any
            for cur_col in tmp.columns:
                print(cur_col)
                if all(np.isnan(tmp.loc[:,cur_col])):
                    print(cur_col)
                    tmp.drop(columns=col_col)
            X = tmp.to_numpy()
            print(X)

        elif filename.split('.')[-1] in ['mat', 'json']:
            # json or mat
            if filename.split('.')[-1] == 'mat':
                import scipy.io as scio
                matdata = scio.loadmat(filename)
                var_names = [x for x in list(matdata.keys()) if not x.startswith('__')]
            else:
                import json
                with open(filename) as f:
                    jsondata = json.load(f)
                var_names = list(jsondata.keys())

            if len(var_names) == 1:
                # single matrix
                X = matdata[var_names[0]]
                tmp = pd.DataFrame(X, columns=['Var'+str(x) for x in X.shape[1]])
            else:
                n = len(var_names)
                # multiple columns
                for cur_var in var_names:
                    X.append(matdata[cur_var].tolist())
                X = np.array(X).T
                tmp = pd.DataFrame(X, columns=var_names)

        else:
            print('load_data: Error - the file format is not supported yet.')
            print('load_data: Warning - accepted data format: csv, dat, txt, mat, json.')

        # Update data sizes
        N, n = X.shape
        print(N)
        print(n)

        # Return data and data sizes
        return X, N, n

    #def check_var_name():

        
    def get_data(self):

        # return data and data sizes
        return self.X, self.N, self.n
    
    def add_data(self, filename, seperator=',', col_header=False):

        # load new data
        new_X, new_N, new_n = self.load_data(filename, seperator, col_header)
        # check data sizes
        if new_n != self._n:
            print('add_data: Error - incompatable column size when loading {}'.format(filename))
            return 1
        else:
            # update the X and N
            self.X = np.concatenate(self._X, new_X)
            self.N = self.N + new_N
        
        return 0

    def initialize_data(self, filename, seperator=',', col_header=False, constraints = ''):

        # initialize the data and data sizes
        try:
            self.X, self.N, self.n = self.load_data(filename, seperator, col_header)
        except:
            print('initialize_data: Error - cannot initialize data with {}'.format(filename))
            return 1

        return 0

    """
    def PlotDataMatrix():

    def RunAlgorithm():
        #...

    def Hreduction():
        #...PCA...
        self._Hvalues= PCA(Xvalues,....)

    def DiffMaps():
        #..diff maps basis...
        self._Zvalues= PCA(Hvalues,....)
    
    """

    def generator(z_init, y_init, a, n_mc, x_mean, eta, s_v, hat_s_v, mu, phi, g, psi = 0, lambda_i = 0, g_c = 0):
        delta_t = 2*pi*hat_s_v/20
        print('delta t: ', delta_t)
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
            L_l_half = L(z_l_half.dot(np.transpose(g)), g_c, x_mean, eta, s_v, hat_s_v, mu, phi, psi, lambda_i).dot(a)
            y_l_1 = (1-beta)*y_l/(1+beta) + delta_t*(L_l_half)/(1+beta) + sqrt(f_0)*w_l_1/(1+beta)
            z_l = z_l_half + delta_t*0.5*y_l_1
            y_l = y_l_1
        for l in range(M_0, M_0*(n_mc+1)):
            z_l_half = z_l + delta_t*0.5*y_l
            w_l_1 = np.random.normal(scale = sqrt(delta_t), size = (nu,N)).dot(a) #wiener process
            L_l_half = L(z_l_half.dot(np.transpose(g)), g_c, x_mean, eta, s_v, hat_s_v, mu, phi, psi, lambda_i).dot(a)
            y_l_1 = (1-beta)*y_l/(1+beta) + delta_t*(L_l_half)/(1+beta) + sqrt(f_0)*w_l_1/(1+beta)
            z_l = z_l_half + delta_t*0.5*y_l_1
            y_l = y_l_1
            if l%M_0 == M_0-1:
                eta_lambda[:,int(l/M_0)*N:(int(l/M_0)+1)*N] = z_l.dot(np.transpose(g))
                nu_lambda[:,int(l/M_0)*N:(int(l/M_0)+1)*N] = y_l.dot(np.transpose(g))
                x_[:,int(l/M_0)-1:int(l/M_0)] = mean(x_mean + phi.dot(np.diag(mu)).dot(eta_lambda[:,:(int(l/M_0)+1)*N]))
                x_2[:,int(l/M_0)-1:int(l/M_0)] = mean((x_mean + phi.dot(np.diag(mu)).dot(eta_lambda[:,:(int(l/M_0)+1)*N]))**2)
        return eta_lambda[:,N:], nu_lambda[:,N:], x_, x_2

    def L(y, g_c, x_mean, eta, s_v, hat_s_v, mu, phi, psi, lambda_i): #gradient of the potential
        nu = eta.shape[0]
        N = eta.shape[1]
        L = np.zeros((nu,N))
        for l in range(0,N):

            yl = np.resize(y[:,l],(len(y[:,l]),1))
            rho_ = rhoctypes(yl, np.resize(np.transpose(eta),(nu*N,1)),\
                    nu, N, s_v, hat_s_v)
            rho_ = 1e250*rho_
            if rho_ < 1e-250:
                print(rho_)
                closest = 1e30
                for i in range(0,N):
                    if closest > np.linalg.norm((hat_s_v/s_v)*np.resize(eta[:,i],yl.shape)-yl):
                        closest = np.linalg.norm((hat_s_v/s_v)*np.resize(eta[:,i],yl.shape)-yl)
                        vector = (hat_s_v/s_v)*np.resize(eta[:,i],yl.shape)-yl
                L[:,l] = (  np.resize(vector/(hat_s_v**2),(nu))\
                    -np.resize(np.diag(mu).dot(np.transpose(phi)).\
                            dot(D_x_g_c(x_mean+np.resize(phi.dot(np.diag(mu)).dot(yl), (x_mean.shape)))).\
                            dot(psi).dot(lambda_i), (nu)))

            else:
                array_pointer = cast(gradient_rhoctypes(np.zeros((nu,1)),yl,\
                    np.resize(np.transpose(eta),(nu*N,1)), nu, N, s_v, hat_s_v), POINTER(c_double*nu))
                gradient_rho = np.frombuffer(array_pointer.contents)
                    #np.resize(gradient_expo(yl)/expo(yl),(nu))\
                L[:,l] = np.resize(1e250*gradient_rho/rho_,(nu))\
                        -np.resize(np.diag(mu).dot(np.transpose(phi)).\
                                dot(D_x_g_c(x_mean+np.resize(phi.dot(np.diag(mu)).dot(yl), (x_mean.shape)))).\
                                    dot(psi).dot(lambda_i), (nu))
        return L
