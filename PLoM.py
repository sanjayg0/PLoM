# -*- coding: utf-8 -*-
#JGA
import numpy as np
import pandas as pd
import random
from math import pi, sqrt
import PLoM_library as plom
import matplotlib.pyplot as plt
#export DISPLAY=localhost:0.0
from ctypes import *
import importlib
from pathlib import Path
import sys
from general import Logfile, DBServer

class PLoM:
    def __init__(self, data='', separator=',', col_header=False, constraints = None, run_tag = False, num_rlz = 5, tol_pca = 1e-6, epsilon_kde = 25):
        # initialize logfile
        self.logfile = Logfile()
        # initialize database server
        self.dbserver = None
        try:
            self.dbserver = DBServer()
        except:
            self.logfile.write_msg(msg='PLoM: database server initialization failed.',msg_type='ERROR',msg_level=0)
        if self.dbserver:
            self.logfile.write_msg(msg='PLoM: database server initialized.',msg_type='RUNNING',msg_level=0)
        # initialize data
        if self.initialize_data(data, separator, col_header):
            self.logfile.write_msg(msg='PLoM: data loading failed.',msg_type='ERROR',msg_level=0)
        # initialize constraints
        self.constraints = {}
        self.num_constraints = 0
        if self.add_constraints(constraints_file=constraints):
            self.logfile.write_msg(msg='PLoM: constraints input failed.',msg_type='ERROR',msg_level=0)
        # run
        if run_tag:
            self.RunAlgorithm(n_mc = num_rlz, epsilon_pca = tol_pca, epsilon_kde = epsilon_kde)
        else:
            self.logfile.write_msg(msg='PLoM: using RunAlgorithm(n_mc=n_mc,epsilon_pca=epsilon_pca,epsilon_kde) to run simulations.',msg_type='RUNNING',msg_level=0)


    def add_constraints(self, constraints_file = None):

        if constraints_file is None:
            self.g_c = None
            self.beta_c = []
            self.lambda_i = 0
            self.psi = 0
            self.logfile.write_msg(msg='PLoM.add_constraints: no user-defined constraint - please use add_constraints(constraints_file=X) to add new constraints if any.',msg_type='WARNING',msg_level=0)
            return 0

        try:
            # path
            path_constraints = Path(constraints_file).resolve()
            sys.path.insert(0, str(path_constraints.parent)+'/')
            # load the function
            new_constraints = importlib.__import__(path_constraints.name[:-3], globals(), locals(), [], 0)
        except:
            self.logfile.write_msg(msg='PLoM.add_constraints: could not add constraints {}'.format(constraints_file),msg_type='ERROR',msg_level=0)
            return 1
        self.num_constraints = self.num_constraints+1
        self.constraints.update({
            'Constraint'+str(self.num_constraints): {
                'filename': constraints_file,
                'g_c': new_constraints.g_c,
                'beta_c': new_constraints.beta_c()
            }
        })
        self.g_c = new_constraints.g_c
        self.beta_c = new_constraints.beta_c()
        self.logfile.write_msg(msg='PLoM.add_constraints: constraints added.',msg_type='RUNNING',msg_level=0)
        return 0


    def switch_constraints(self, constraint_tag = 1):
        """
        Selecting different constraints
        - constraint_tag: the tag of selected constraint
        """

        if constraint_tag > self.num_constraints:
            self.logfile.write_msg(msg='PLoM.switch_constraints: sorry the maximum constraint tag is {}'.format(self.num_constraints),msg_type='ERROR',msg_level=0)
        try:
            self.g_c = self.constraints.get('Constraint'+str(constraint_tag)).get('g_c')
            self.beta_c = self.constraints.get('Constraint'+str(constraint_tag)).get('beta_c')
        except:
            self.logfile.write_msg(msg='PLoM.get_constraints: cannot get constraints',msg_type='ERROR',msg_level=0)


    def delete_constraints(self):
        """
        Removing all current constraints
        """

        self.g_c = None
        self.beta_c = []


    def load_data(self, filename, separator=',', col_header=False):

        # initialize the matrix and data size
        X = []
        N = 0
        n = 0

        # check if the file exist
        import os
        if not os.path.exists(filename):
            self.logfile.write_msg(msg='load_data: Error - the input file {} is not found'.format(filename),msg_type='ERROR',msg_level=0)
            return X, N, n

        # read data
        if filename.split('.')[-1] in ['csv','dat','txt']:
            # txt data
            col = None
            if col_header:
                col = 0
            self.X0_table = pd.read_table(filename, delimiter=separator, header=col)
            # remove all-nan column if any
            for cur_col in self.X0_table.columns:
                if all(np.isnan(self.X0_table.loc[:,cur_col])):
                    self.X0_table.drop(columns=cur_col)
            X = self.X0_table.to_numpy()

        elif filename.split('.')[-1] in ['mat', 'json']:
            # json or mat
            if filename.split('.')[-1] == 'mat':
                import scipy.io as scio
                matdata = scio.loadmat(filename)
                var_names = [x for x in list(matdata.keys()) if not x.startswith('__')]
                if len(var_names) == 1:
                    # single matrix
                    X = matdata[var_names[0]]
                    self.X0_table = pd.DataFrame(X, columns=['Var'+str(x) for x in X.shape[1]])
                else:
                    n = len(var_names)
                    # multiple columns
                    for cur_var in var_names:
                        X.append(matdata[cur_var].tolist())
                    X = np.array(X).T
                    X = X[0,:,:]
                    self.X0_table = pd.DataFrame(X, columns=var_names)
            else:
                import json
                with open(filename) as f:
                    jsondata = json.load(f)
                var_names = list(jsondata.keys())
                # multiple columns
                for cur_var in var_names:
                    X.append(jsondata[cur_var])
                X = np.array(X).T
                self.X0_table = pd.DataFrame(X, columns=var_names)

        else:
            self.logfile.write_msg(msg='PLoM.load_data: the file format is not supported yet.',msg_type='ERROR',msg_level=0)
            self.logfile.write_msg(msg='PLoM.load_data: accepted data formats - csv, dat, txt, mat, json.',msg_type='WARNING',msg_level=0)

        # Update data sizes
        N, n = X.shape
        self.logfile.write_msg(msg='PLoM.load_data: loaded data size = ({}, {}).'.format(N,n),msg_type='RUNNING',msg_level=0)

        # Return data and data sizes
        return X.T, N, n

    #def check_var_name():


    def get_data(self):

        # return data and data sizes
        return self.X, self.N, self.n


    def add_data(self, filename, separator=',', col_header=False):

        # load new data
        new_X, new_N, new_n = self.load_data(filename, separator, col_header)
        # check data sizes
        if new_n != self.n:
            self.logfile.write_msg(msg='PLoM.add_data: incompatible column size when loading {}'.format(filename),msg_type='ERROR',msg_level=0)
        else:
            # update the X and N
            self.X = np.concatenate((self.X, new_X))
            self.N = self.N + new_N
            self.X0_table.append(pd.DataFrame(new_X.T, columns=list(self.X0_table.columns)))

        self.logfile.write_msg(msg='PLoM.add_data: current X0 size = ({}, {}).'.format(self.N,self.n),msg_type='RUNNING',msg_level=0)


    def initialize_data(self, filename, separator=',', col_header=False, constraints = ''):

        # initialize the data and data sizes
        try:
            self.X, self.N, self.n = self.load_data(filename, separator, col_header)
        except:
            self.logfile.write_msg(msg='PLoM.initialize_data: cannot initialize data with {}'.format(filename),msg_type='ERROR',msg_level=0)
            return 1

        # Save to database
        self.dbserver.add_item(item_name = 'X0', col_names = list(self.X0_table.columns), item = self.X.T)
        self.dbserver.add_item(item_name = 'X0_size', col_names = ['N0', 'n0'], item = np.array([[self.N,self.n]]))
        self.logfile.write_msg(msg='PLoM.initialize_data: current X0 size = ({}, {}).'.format(self.N,self.n),msg_type='RUNNING',msg_level=0)
        self.logfile.write_msg(msg='PLoM.initialize_data: X0 and X0_size saved to database.',msg_type='RUNNING',msg_level=0)

        return 0


    def RunAlgorithm(self, n_mc = 5, epsilon_pca = 1e-6, epsilon_kde = 25):
        """
        Running the PLoM algorithm to train the model and generate new realizations
        - n_mc: realization/sample size ratio
        - epsilon_pca: tolerance for selecting the number of considered componenets in PCA
        - epsilon_kde: smoothing parameter in the kernel density estimation
        """

        #scaling
        self.X_scaled, self.alpha, self.x_min = plom.scaling(self.X)
        self.x_mean = plom.mean(self.X_scaled)
        self.logfile.write_msg(msg='PLoM.RunAlgorithm: data normalization completed.',msg_type='RUNNING',msg_level=0)
        self.dbserver.add_item(item_name = 'X_scaled', col_names = list(self.X0_table.columns), item = self.X_scaled.T)
        self.dbserver.add_item(item_name = 'X_scaled_mean', col_names = list(self.X0_table.columns), item = self.x_mean.T)
        self.logfile.write_msg(msg='PLoM.RunAlgorithm: X_scaled and X_scaled_mean saved.',msg_type='RUNNING',msg_level=0)

        #PCA
        self.H, self.mu, self.phi, self.nu = self._Hreduction(self.X_scaled, epsilon_pca)
        self.logfile.write_msg(msg='PLoM.RunAlgorithm: PCA completed.',msg_type='RUNNING',msg_level=0)
        self.dbserver.add_item(item_name = 'X_PCA', col_names = ['Component'+str(i+1) for i in range(self.H.shape[0])], item = self.H.T)
        self.dbserver.add_item(item_name = 'EigenValue_PCA', col_names = 'EigenValue_PCA', item = self.mu)
        self.dbserver.add_item(item_name = 'EigenVector_PCA', col_names = ['V'+str(i+1) for i in range(self.phi.shape[0])], item = self.phi)
        self.logfile.write_msg(msg='PLoM.RunAlgorithm: X_PCA, EigenValue_PCA and EigenVector_PCA saved.',msg_type='RUNNING',msg_level=0)

        #parameters KDE
        (self.s_v, self.c_v, self.hat_s_v) = plom.parameters_kde(self.H)
        self.K, self.b = plom.K(self.H, epsilon_kde)
        self.logfile.write_msg(msg='PLoM.RunAlgorithm: kernel density estimation completed.',msg_type='RUNNING',msg_level=0)
        self.dbserver.add_item(item_name = 'KDE', col_names = ['s_v','c_v','hat_s_v'], item = np.array([[self.s_v,self.c_v,self.hat_s_v]]))
        self.dbserver.add_item(item_name = 'X_KDE', item = self.K)
        self.dbserver.add_item(item_name = 'EigenValues_KDE', item = self.b)

        #diff maps
        self.g, self.m, self.a, self.Z = self._DiffMaps(self.H, self.K, self.b)
        self.logfile.write_msg(msg='PLoM.RunAlgorithm: diffusion maps completed.',msg_type='RUNNING',msg_level=0)
        self.dbserver.add_item(item_name = 'DiffMaps_g', item = self.g)
        self.dbserver.add_item(item_name = 'DiffMaps_m', item = np.array([self.m]))
        self.dbserver.add_item(item_name = 'DiffMaps_a', item = self.a)
        self.dbserver.add_item(item_name = 'DiffMaps_Z', item = self.Z)

        #constraints
        if self.constraints = None:

        #no constraints
        else:
            nu_init = np.random.normal(size=(self.nu,self.N))
            self.Y = nu_init.dot(self.a)
            Hnewvalues, nu_lambda, x_, x_2 = plom.generator(self.Z, self.Y, self.a,\
                                        n_mc, self.x_mean, self.H, self.s_v,\
                                        self.hat_s_v, self.mu, self.phi,\
                                        self.g[:,0:self.m],  psi=self.psi,\
                                        lambda_i=self.lambda_i, g_c=self.g_c) #solve the ISDE in n_mc iterations
            self.Xnew = self.x_mean + self.phi.dot(np.diag(self.mu)).dot(Hnewvalues)
        
        #unscale
        self.Xnew = np.diag(self.alpha).dot(self.Xnew)+self.x_min

        self.logfile.write_msg(msg='PLoM.RunAlgorithm: Realizations generated.',msg_type='RUNNING',msg_level=0)
        self.dbserver.add_item(item_name = 'X_new', col_names = list(self.X0_table.columns), item = self.Xnew.T)
        self.logfile.write_msg(msg='PLoM.RunAlgorithm: X_new saved.',msg_type='RUNNING',msg_level=0)


    def _Hreduction(self, X_origin, epsilon_pca):
        #...PCA...
        (H, mu, phi) = plom.PCA(X_origin, epsilon_pca)
        nu = len(H)
        self.logfile.write_msg(msg='PLoM._Hreduction: considered number of PCA components = {}'.format(nu),msg_type='RUNNING',msg_level=0)
        return H, mu, phi, nu


    def _DiffMaps(self, H, K, b):
        #..diff maps basis...
        #self.Z = PCA(self.H)
        try:
            g, eigenvalues = plom.g(K, b) #diffusion maps
            g = g.real
            m = plom.m(eigenvalues)
            a = g[:,0:m].dot(np.linalg.inv(np.transpose(g[:,0:m]).dot(g[:,0:m])))
            Z = H.dot(a)
        except:
            g = None
            m = 0
            a = None
            Z = None
            self.logfile.write_msg(msg='PLoM.__DiffMaps: diffusion maps failed.',msg_type='ERROR',msg_level=0)

        return g, m, a, Z


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


        
