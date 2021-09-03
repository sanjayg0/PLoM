# -*- coding: utf-8 -*-
#JGA
import numpy as np
import random
from math import pi, sqrt
import PLoM_library as plom
import matplotlib.pyplot as plt
#export DISPLAY=localhost:0.0
from ctypes import *
import importlib
from pathlib import Path
import sys

class PLoM:
    def __init__(self, data='', separator=',', col_header=False, constraints = None):
        self.initialize_data(data, separator, col_header)
        self.constraints = {}
        self.num_constraints = 0
        if self.add_constraints(constraints_file=constraints):
            print('PLoM: constraints input failed.')
        #self._npoints = Xvalues.shape[1]
        #self._dimensions = Xvalues.shape[0]
        #self._constraints = AddConstraints(constraints)

    def add_constraints(self, constraints_file = None):

        if constraints_file is None:
            self.g_c = None
            self.beta_c = []
            print('add_constraints: no user-defined constraint - please use add_constraints(constraints_file=X) to add new constraints if any.')
            return 0

        try:
            # path
            path_constraints = Path(constraints_file).resolve()
            sys.path.insert(0, str(path_constraints.parent)+'/')
            # load the function
            new_constraints = importlib.__import__(path_constraints.name[:-3], globals(), locals(), [], 0)
            #self.g_c = new_constraints.g_c
            #self.beta_c = new_constraints.beta_c()
        except:
            print('add_constraints: Error - could not add constraints {}'.format(constraints_file))
            return 1
        self.num_constraints = self.num_constraints+1
        self.constraints.update({
            'Constraint'+str(self.num_constraints): {
                'g_c': new_constraints.g_c,
                'beta_c': new_constraints.beta_c()
            }
        })
        self.g_c = new_constraints.g_c
        self.beta_c = new_constraints.beta_c()
        print('add_constraints: constraints added')
        return 0

        #self._constraints = NewConstraints

    def switch_constraints(self, constraint_tag = 1):

        if constraint_tag > self.num_constraints:
            print('get_constratins: sorry the maximum constraint tag is {}'.format(self.num_constraints))
            #return None, []

        try:
            g_c = self.constraints.get('Constraint'+str(constraint_tag)).get('g_c')
            beta_c = self.constraints.get('Constraint'+str(constraint_tag)).get('beta_c')
            self.g_c = new_constraints.g_c
            self.beta_c = new_constraints.beta_c()
            #return g_c, beta_c
        except:
            print('get_constratins: Error - cannot get constraints')
            #return None, []

    def delete_constraints(self):

        self.g_c = None
        self.beta_c = []

    def load_data(self, filename, separator=',', col_header=False):

        # initialize the matrix and data size
        X = []
        N = 0
        n = 0

        # check if the file exist
        import os
        import pandas as pd
        if not os.path.exists(filename):
            print('load_data: Error - the input file {} is not found'.format(filename))
            return X, N, NewConstraints

        # read data
        if filename.split('.')[-1] in ['csv','dat','txt']:
            # txt data
            col = None
            if col_header:
                col = 0
            tmp = pd.read_table(filename, delimiter=separator, header=col)
            # remove all-nan column if any
            for cur_col in tmp.columns:
                if all(np.isnan(tmp.loc[:,cur_col])):
                    tmp.drop(columns=cur_col)
            X = tmp.to_numpy()

        elif filename.split('.')[-1] in ['mat', 'json']:
            # json or mat
            if filename.split('.')[-1] == 'mat':
                import scipy.io as scio
                matdata = scio.loadmat(filename)
                var_names = [x for x in list(matdata.keys()) if not x.startswith('__')]
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
                    X = X[0,:,:]
                    tmp = pd.DataFrame(X, columns=var_names)
            else:
                import json
                with open(filename) as f:
                    jsondata = json.load(f)
                var_names = list(jsondata.keys())
                print(var_names)
                # multiple columns
                for cur_var in var_names:
                    X.append(jsondata[cur_var])
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
        #return self.X, self.N, self.n

    #def check_var_name():

    def get_data(self):

        # return data and data sizes
        return self.X, self.N, self.n

    def add_data(self, filename, separator=',', col_header=False):

        # load new data
        new_X, new_N, new_n = self.load_data(filename, separator, col_header)
        # check data sizes
        if new_n != self.n:
            print('add_data: Error - incompatable column size when loading {}'.format(filename))
            return 1
        else:
            # update the X and N
            print(self.X)
            print(new_X)
            self.X = np.concatenate((self.X, new_X))
            self.N = self.N + new_N
        
        return 0

    def initialize_data(self, filename, separator=',', col_header=False, constraints = ''):

        # initialize the data and data sizes
        try:
            self.X, self.N, self.n = self.load_data(filename, separator, col_header)
        except:
            print('initialize_data: Error - cannot initialize data with {}'.format(filename))
            return 1

        return 0

    """
    def PlotDataMatrix():
    """

    def RunAlgorithm(self):

        #scaling
        self.X_scaled, self.alpha, self.x_min = plom.scaling(self.X)
        self.x_mean = plom.mean(self.X_scaled)

        #PCA
        self.Hreduction(self)

        #parameters KDE
        (self.s_v, self.c_v, self.hat_s_v) = plom.parameters_kde(self.H)

        #diff maps
        self.DiffMaps(self)

        #no constraints
        Hnewvalues, nu_lambda, x_, x_2 = plom.generator(z_init, y_init, a,\
                                    n_mc, x_mean, eta, s_v, hat_s_v, mu, phi, g[:,0:m],  psi,\
                                    lambda_i, g_c) #solve the ISDE in n_mc iterations
        self.Xnew = self.x_mean + phi.dot(np.diag(mu)).dot(Hnewvalues)

    def Hreduction(self):
        #...PCA...
        tol = 1e-9
        (self.H, self.mu, self.phi) = plom.PCA(self.X_scaled, tol)
        self.nu = len(eta)
        return self.H, self.mu, self.phi, self.nu

    def DiffMaps(self):
        #..diff maps basis...
        self.Z = PCA(self.H)
        epsilon = 16
        self.K, self.b = plom.K(self.H, epsilon)
        self.g, self.eigenvalues = plom.g(self.K, self.b) #diffusion maps
        self.m = plom.m(eigenvalues)
        self.a = (self.g[:,0:m]).dot(np.linalg.inv(np.transpose(self.g[:,0:m]).dot(self.g[:,0:m])))
        self.Z = (self.H).dot(self.a)

        return self.Z, self.a, self.K, self.b, self.g, self.eigenvalues, self.m, self.a, self.Z
        
    def PostProcess():
    	#...output plots...

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


        
