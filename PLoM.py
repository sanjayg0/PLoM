# -*- coding: utf-8 -*-
#JGA
import os
import numpy as np
import pandas as pd
import random
from math import pi, sqrt
import PLoM_library as plom
import matplotlib.pyplot as plt
import warnings
#export DISPLAY=localhost:0.0
from ctypes import *
import importlib
from pathlib import Path
import sys
from general import *

class PLoM:
    def __init__(self, jobname='plom', data='', separator=',', col_header=False, constraints = None, run_tag = False, plot_tag = True, num_rlz = 5, tol_pca = 1e-6, epsilon_kde = 25):
        # basic setups
        self._basic_config(jobname=jobname)
        # initialize input data
        if self.initialize_data(data, separator, col_header):
            self.logfile.write_msg(msg='PLoM: data loading failed.',msg_type='ERROR',msg_level=0)
        else:
            # plot data matrix
            if plot_tag:
                smp = pd.plotting.scatter_matrix(self.X0, alpha=0.5, diagonal ='kde', figsize=(10,10))
                for ax in smp.ravel():
                    ax.set_xlabel(ax.get_xlabel(), fontsize = 6, rotation = 45)
                    ax.set_ylabel(ax.get_ylabel(), fontsize = 6, rotation = 45)
                plt.savefig(os.path.join(self.vl_path,'ScatterMatrix_X0.png'),dpi=480)
                self.logfile.write_msg(msg='PLoM: {} saved in {}.'.format('ScatterMatrix_X0.png',self.vl_path),msg_type='RUNNING',msg_level=0)
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

        
    def _basic_config(self, jobname=None):
        """
        Basic setups
        - jobname: job name (used for database name)
        """
        # initialize log and running directories
        self.dir_log = os.path.join(os.path.expanduser('~/Documents'),'PLoM')
        self.dir_run = os.path.join(os.path.expanduser('~/Documents'),'PLoM',jobname)
        # initialize logfile
        self.logfile = Logfile(logfile_dir = self.dir_log)
        try:
            os.makedirs(self.dir_run, exist_ok=True)
            self.logfile.write_msg(msg='PLoM: Running directory {} initialized.'.format(self.dir_run),msg_type='RUNNING',msg_level=0)
        except:
            self.logfile.write_msg(msg='PLoM: Running directory {} cannot be initialized.'.format(self.dir_run),msg_type='ERROR',msg_level=0)
        # initialize database server
        self.dbserver = None
        try:
            self.dbserver = DBServer(db_dir = self.dir_run, db_name=jobname+'.h5')
        except:
            self.logfile.write_msg(msg='PLoM: database server initialization failed.',msg_type='ERROR',msg_level=0)
        if self.dbserver:
            self.logfile.write_msg(msg='PLoM: database server initialized.',msg_type='RUNNING',msg_level=0)
        # initialize visualization output path
        self.vl_path = os.path.join(self.dir_run,'FigOut')
        try:
            os.makedirs(self.vl_path, exist_ok=True)
            self.logfile.write_msg(msg='PLoM: visualization folder {} initialized.'.format(self.vl_path),msg_type='RUNNING',msg_level=0)
        except:
            self.logfile.write_msg(msg='PLoM: visualization folder {} not initialized.'.format(self.vl_path),msg_type='WARNING',msg_level=0)        

    
    def add_constraints(self, constraints_file = None):

        if constraints_file is None:
            self.g_c = None
            self.D_x_g_c = None
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
                'D_x_g_c': new_constraints.D_x_g_c,
                'beta_c': new_constraints.beta_c()
            }
        })
        self.g_c = new_constraints.g_c
        self.D_x_g_c = new_constraints.D_x_g_c
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
            self.D_x_g_c = self.constraints.get('Constraint'+str(constraint_tag)).get('D_x_g_c')
            self.beta_c = self.constraints.get('Constraint'+str(constraint_tag)).get('beta_c')
        except:
            self.logfile.write_msg(msg='PLoM.get_constraints: cannot get constraints',msg_type='ERROR',msg_level=0)


    def delete_constraints(self):
        """
        Removing all current constraints
        """

        self.g_c = None
        self.D_x_g_c = None
        self.beta_c = []


    def load_data(self, filename, separator=',', col_header=False):

        # initialize the matrix and data size
        X = []
        N = 0
        n = 0

        # check if the file exist
        import os
        if not os.path.exists(filename):
            self.logfile.write_msg(msg='load_data: the input file {} is not found'.format(filename),msg_type='ERROR',msg_level=0)
            return X, N, n

        # read data
        if os.path.splitext(filename)[-1] in ['.csv','.dat','.txt']:
            # txt data
            col = None
            if col_header:
                col = 0
            self.X0 = pd.read_table(filename, delimiter=separator, header=col)
            # remove all-nan column if any
            for cur_col in self.X0.columns:
                if all(np.isnan(self.X0.loc[:,cur_col])):
                    self.X0.drop(columns=cur_col)
            X = self.X0.to_numpy()

        elif os.path.splitext(filename)[-1] in ['.mat', '.json']:
            # json or mat
            if os.path.splitext(filename)[-1] == '.mat':
                import scipy.io as scio
                matdata = scio.loadmat(filename)
                var_names = [x for x in list(matdata.keys()) if not x.startswith('__')]
                if len(var_names) == 1:
                    # single matrix
                    X = matdata[var_names[0]]
                    self.X0 = pd.DataFrame(X, columns=['Var'+str(x) for x in X.shape[1]])
                else:
                    n = len(var_names)
                    # multiple columns
                    for cur_var in var_names:
                        X.append(matdata[cur_var].tolist())
                    X = np.array(X).T
                    X = X[0,:,:]
                    self.X0 = pd.DataFrame(X, columns=var_names)
            else:
                import json
                with open(filename) as f:
                    jsondata = json.load(f)
                var_names = list(jsondata.keys())
                # multiple columns
                for cur_var in var_names:
                    X.append(jsondata[cur_var])
                X = np.array(X).T
                self.X0 = pd.DataFrame(X, columns=var_names)
        
        elif os.path.splitext(filename)[-1] in ['.h5']:
            # this h5 can be either formatted by PLoM or not
            # a separate method to deal with this file
            X = self.load_h5(filename)

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


    def _load_h5_plom(self, filename):
        """
        Loading PLoM-formatted h5 database 
        """
        try:
            store = pd.HDFStore(filename, 'r')
            for cur_var in store.keys():
                if cur_var in self.dbserver.get_item_adds() and ATTR_MAP[cur_var]:
                    # read in
                    cur_data = store[cur_var]
                    self.dbserver.add_item(item_name=cur_var.replace('/',''),col_names=list(cur_data.columns),item=cur_data)
            store.close()

        except:
            self.logfile.write_msg(msg='PLoM._load_h5_plom: data in {} not compatible.'.format(filename),msg_type='ERROR',msg_level=0)



    def _load_h5_data_X(self, filename):
        """
        Loading a h5 data which is expected to contain X data
        """
        try:
            store = pd.HDFStore(filename, 'r')
            # Note a table is expected for the variable
            self.X0 = store.get(store.keys()[0])
            store.close()
            self.dbserver.add_item(item_name='X0',col_name=list(self.X0.columns),item=self.X0)

            return self.X0.to_numpy()
        except:
            return None


    def _sync_data(self):
        """
        Sync database data to current attributes
        """
        avail_name_list = self.dbserver.get_name_list()
        if not avail_name_list:
            # empty database
            self.logfile.write_msg(msg='PLoM._sync_data: database is empty - no data to sync.',msg_type='WARNING',msg_level=0)
        else:
            for cur_item in avail_name_list:
                if type(ATTR_MAP[cur_item]) is str:
                    # only one data group in this data item:
                    self.__setattr__(ATTR_MAP[cur_item],self.dbserver.get_item(cur_item))
                    self.logfile.write_msg(msg='PLoM._sync_data: self.{} synced.'.format(ATTR_MAP[cur_item]),msg_type='RUNNING',msg_level=0)
                elif type(ATTR_MAP[cur_item]) is list:
                    # multiple data groups
                    for cur_var in ATTR_MAP[cur_item]:
                        self.__setattr__(cur_var,self.dbserver.get_item(cur_item).get(cur_var))
                        self.logfile.write_msg(msg='PLoM._sync_data: self.{} synced.'.format(cur_var),msg_type='RUNNING',msg_level=0)
                else:
                    # None type (this is the 'basic' - skipped)
                    self.logfile.write_msg(msg='PLoM._sync_data: data {} skipped.'.format(cur_item),msg_type='RUNNING',msg_level=0)


    def load_h5(self, filename):
        """
        Loading h5 database
        """
        try:
            self._load_h5_plom(filename)
            # sync data
            self._sync_data()
            if '/X0' in self.dbserver.get_name_list():
                return self.X0.to_numpy()
            else:
                self.logfile.write_msg(msg='PLoM.load_h5: the original X0 data not found in the loaded data.',msg_type='ERROR',msg_level=0)
                return None
        except:
            X = self._load_h5_data_X(filename)
            if X is None:
                self.logfile.write_msg(msg='PLoM.load_h5: cannot load {}.'.format(filename),msg_type='ERROR',msg_level=0)
                return None
            else:
                return X


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
            self.X0.append(pd.DataFrame(new_X.T, columns=list(self.X0.columns)))

        self.logfile.write_msg(msg='PLoM.add_data: current X0 size = ({}, {}).'.format(self.N,self.n),msg_type='RUNNING',msg_level=0)


    def initialize_data(self, filename, separator=',', col_header=False, constraints = ''):

        # initialize the data and data sizes
        try:
            self.X, self.N, self.n = self.load_data(filename, separator, col_header)
        except:
            self.logfile.write_msg(msg='PLoM.initialize_data: cannot initialize data with {}'.format(filename),msg_type='ERROR',msg_level=0)
            return 1

        # Save to database
        self.dbserver.add_item(item_name = 'X0', col_names = list(self.X0.columns), item = self.X.T)
        self.dbserver.add_item(item_name = 'X0_size', col_names = ['N', 'n'], item = np.array([[self.N,self.n]]))
        self.logfile.write_msg(msg='PLoM.initialize_data: current X0 size = ({}, {}).'.format(self.N,self.n),msg_type='RUNNING',msg_level=0)
        self.logfile.write_msg(msg='PLoM.initialize_data: X0 and X0_size saved to database.',msg_type='RUNNING',msg_level=0)

        return 0


    def RunAlgorithm(self, n_mc = 5, epsilon_pca = 1e-6, epsilon_kde = 25, tol_PCA2 = 1e-5, tol = 1e-6, max_iter = 50):
        """
        Running the PLoM algorithm to train the model and generate new realizations
        - n_mc: realization/sample size ratio
        - epsilon_pca: tolerance for selecting the number of considered componenets in PCA
        - epsilon_kde: smoothing parameter in the kernel density estimation
        - tol: tolerance in the PLoM iterations
        - max_iter: maximum number of iterations of the PLoM algorithm
        """

        #scaling
        self.X_scaled, self.alpha, self.x_min = plom.scaling(self.X)
        self.x_mean = plom.mean(self.X_scaled)
        self.logfile.write_msg(msg='PLoM.RunAlgorithm: data normalization completed.',msg_type='RUNNING',msg_level=0)
        self.dbserver.add_item(item_name = 'X_scaled', col_names = list(self.X0.columns), item = self.X_scaled.T)
        self.dbserver.add_item(item_name = 'X_scaled_mean', col_names = list(self.X0.columns), item = self.x_mean.T)
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
        if self.g_c:

            self.C_h_hat_eta = plom.covariance(self.g_c(self.x_mean+(self.phi).dot(np.diag(self.mu)).dot(self.H)))
            self.b_c, self.psi = plom.PCA2(self.C_h_hat_eta, self.beta_c, tol_PCA2)
            self.nu_c = len(self.b_c)


            self.hessian = plom.hessian_gamma(self.H, self.psi, self.g_c, self.phi, self.mu, self.x_mean)
            self.inverse = plom.solve_inverse(self.hessian)

            self.lambda_i = -(self.inverse)\
            .dot(plom.gradient_gamma(self.b_c, self.H, self.g_c, self.phi, self.mu, self.psi, self.x_mean))

            self.gradient = plom.gradient_gamma(self.b_c, self.H, self.g_c, self.phi, self.mu, self.psi, self.x_mean)
            self.errors = [plom.err(self.gradient, self.b_c)]
            iteration = 0
            nu_init = np.random.normal(size=(self.nu,self.N))
            self.Y = nu_init.dot(self.a)

            while (iteration < max_iter and self.errors[iteration] > tol):
                self.logfile.write_msg(msg='PLoM.RunAlgorithm: Running iteration {}.'.format(iteration+1),msg_type='RUNNING',msg_level=0)


                Hnewvalues, nu_lambda, x_, x_2 = plom.generator(self.Z, self.Y, self.a,\
                                            n_mc, self.x_mean, self.H, self.s_v,\
                                            self.hat_s_v, self.mu, self.phi,\
                                            self.g[:,0:self.m],  psi=self.psi,\
                                            lambda_i=self.lambda_i, g_c=self.g_c, D_x_g_c = self.D_x_g_c) #solve the ISDE in n_mc iterations

                self.gradient = plom.gradient_gamma(self.b_c, Hnewvalues, self.g_c, self.phi, self.mu, self.psi, self.x_mean)
                self.hessian = plom.hessian_gamma(Hnewvalues, self.psi, self.g_c, self.phi, self.mu, self.x_mean)
                self.inverse = plom.solve_inverse(self.hessian)

                self.lambda_i = self.lambda_i - 0.3*(self.inverse).dot(self.gradient)

                self.Z = Hnewvalues[:,-self.N:].dot(self.a)
                self.Y = nu_lambda[:,-self.N:].dot(self.a)
                iteration += 1

                (self.errors).append(plom.err(self.gradient, self.b_c))
            
            #saving data
            self.dbserver.add_item(item_name = 'Errors', item = np.array(self.errors))

            if iteration == max_iter:
                self.logfile.write_msg(msg='PLoM.RunAlgorithm: Max. iteration reached and convergence not achieved.',msg_type='WARNING',msg_level=0) 

        #no constraints
        else:
            nu_init = np.random.normal(size=(self.nu,self.N))
            self.Y = nu_init.dot(self.a)
            Hnewvalues, nu_lambda, x_, x_2 = plom.generator(self.Z, self.Y, self.a,\
                                        n_mc, self.x_mean, self.H, self.s_v,\
                                        self.hat_s_v, self.mu, self.phi,\
                                        self.g[:,0:self.m]) #solve the ISDE in n_mc iterations

        self.Xnew = self.x_mean + self.phi.dot(np.diag(self.mu)).dot(Hnewvalues)
        
        #unscale
        self.Xnew = np.diag(self.alpha).dot(self.Xnew)+self.x_min

        self.logfile.write_msg(msg='PLoM.RunAlgorithm: Realizations generated.',msg_type='RUNNING',msg_level=0)
        self.dbserver.add_item(item_name = 'X_new', col_names = list(self.X0.columns), item = self.Xnew.T)
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

    
    def export_results(self, data_list = [], file_format_list = ['csv']):
        """
        Exporting results by the data names
        - data_list: list of data names
        - filename_list: list of output filenames
        - file_format_list: list of output formats
        """
        avail_name_list = self.dbserver.get_name_list()
        if not data_list:
            # print available data names
            avail_name_str = ','.join(avail_name_list)
            self.logfile.write_msg(msg='PLoM.export_results: available data {}.'.format(avail_name_str),msg_type='RUNNING',msg_level=0)
        if not avail_name_list:
            # empty database
            self.logfile.write_msg(msg='PLoM.export_results: database is empty - no data exported.',msg_type='ERROR',msg_level=0)
        else:
            for tag, data_i in enumerate(data_list):
                if data_i not in avail_name_list:
                    self.logfile.write_msg(msg='PLoM.export_results: {} is not found and skipped.'.format(data_i),msg_type='WARNING',msg_level=0)
                else:
                    try:
                        ff_i = file_format_list[tag]
                    except:
                        ff_i = file_format_list[-1]
                    ex_flag = self.dbserver.export(data_name = data_i, file_format = ff_i)
                    if ex_flag == 1:
                        self.logfile.write_msg(msg='PLoM.export_results: {} is not found and skipped.'.format(data_i),msg_type='WARNING',msg_level=0)
                    elif ex_flag == 2:
                        self.logfile.write_msg(msg='PLoM.export_results: {} is not supported yest.'.format(ff_i),msg_type='ERROR',msg_level=0)
                    else:
                        self.logfile.write_msg(msg='PLoM.export_results: {} is exported in {}.'.format(data_i,self.dbserver.dir_export),msg_type='RUNNING',msg_level=0)


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


        
