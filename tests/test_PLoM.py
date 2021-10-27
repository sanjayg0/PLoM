"""
These are unit and integration tests on the PLoM module.
Note:
1. PLoM calls methods from general whose testing is addressed by test_general,
   so the those are not tested by this script
2. Nested functions are tested from bottom to top and testing low-level functions
   is not included (as duplicated) in testing a high-level method
3. Methods invoked from PLoM_library are not individually tested here
4. export-related functions are not explicitly tested here
"""

import pytest
import numpy as np
import json
import os
import inspect
import pandas as pd
import shutil
import matplotlib.pyplot as plt
plt.ion()
# Importing all in the general module
from PLoM import *
# Current directory
cur_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# Testing input directory
base_input_path = 'input_test'
base_constraints_path = 'constraints_test'

def test_PLoM_load_data():
    """
    Testing the PLoM.load_data()
    Six input formats are available and tested here
    """
    res = []
    test_format = ['.csv','.dat','.json','.txt','.mat','.h5']
    test_col = [True, False, False, False, False, False]
    test_sep = [',', '|', None, ' ', None, None]
    ref_output = [(447,11),(175000,14),(447,11),(875,15),(200,11),(475,27)]
    ref = [1,1,1,1,1,1]
    a = PLoM(data=None)

    for i in range(len(test_format)):
        cur_datafile = os.path.join(cur_dir,base_input_path,'input_test'+test_format[i])
        X, N, n = a.load_data(filename=cur_datafile, separator=test_sep[i], col_header=test_col[i])
        print(N,n)
        res.append((N,n)==ref_output[i])
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)


def test_PLoM_add_constraints():
    """
    Testing PLoM.add_constraint()
    """
    res = []
    test_input = ['constraints_test1.py','constraints_test2.py']
    ref_output = [(type(np.ndarray(1)), [[0.5],[0.3]], None),
                  (type(np.ndarray(1)), [[20],[150]], [[1,0],[0,1]]+[[0,0]]*9)]
    ref = [1,1]
    a = PLoM(data=os.path.join(cur_dir,base_input_path,'input_test.json'))

    for i in range(len(test_input)):
        cur_cfile = os.path.join(cur_dir,base_constraints_path,test_input[i])
        a.add_constraints(constraints_file=cur_cfile)
        res.append(int(type(a.g_c(a.X))==ref_output[i][0] and all(a.beta_c==ref_output[i][1]) and np.min(np.array(a.D_x_g_c(a.X)==ref_output[i][2]))))

    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)


def test_PLoM_delete_constraints():
    """
    Testing PLoM.delete_constraints()
    """
    res = []
    ref = [1]
    a = PLoM(data=None,constraints=os.path.join(cur_dir,base_constraints_path,'constraints_test1.py'))
    a.delete_constraints()
    res.append(int(a.g_c==None and a.beta_c==[] and a.D_x_g_c==None))

    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)


def test_PLoM_get_data():
    """
    Testing PLoM.get_data()
    """
    res = []
    ref_output = [(447,11)]
    ref = [1]
    a = PLoM(data=os.path.join(cur_dir,base_input_path,'input_test.json'))
    X, N, n = a.get_data()
    res.append((N,n)==ref_output[0])
    
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)


def test_PLoM_DataNormalization():
    """
    Testing PLoM.DataNormalization()
    """
    res = []
    test_input = np.array([[1,2],[2,1]])
    ref_output=[[[0,1],[1,0]],[1,1],[[1],[1]],[[0.5],[0.5]]]
    ref = [1,1,1,1]

    x_scaled, alpha, x_min, x_mean = PLoM.DataNormalization([],test_input)
    res.append(np.min(np.array(x_scaled==ref_output[0])))
    res.append(np.min(np.array(alpha==ref_output[1])))
    res.append(np.min(np.array(x_min==ref_output[2])))
    res.append(np.min(np.array(x_mean==ref_output[3])))

    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)


def test_PLoM_RunPCA():
    """
    Testing PLoM.RunPCA()
    """
    res = []
    test_input = np.array([[1,0],[0,1]])
    ref_output = [[-np.sqrt(0.5),np.sqrt(0.5)],[1],[[-np.sqrt(0.5)],[np.sqrt(0.5)]],1]
    ref = [1,1,1,1]

    a = PLoM(data=None)
    H, mu, phi, nu = a.RunPCA(test_input, 1e-5)
    res.append(np.min(np.isclose(H,ref_output[0],atol=1e-5)))
    res.append(np.min(np.isclose(mu,ref_output[1],atol=1e-5)))
    res.append(np.min(np.isclose(phi,ref_output[2],atol=1e-5)))
    res.append(np.min(np.isclose(nu,ref_output[3],atol=1e-5)))

    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)

    res = []
    test_input = np.array([[1,2,2],[1,2,1],[0,2,2]])
    ref_output = [[[1.14948699,-0.66966318,-0.47982381]],[1.32806725],[[-0.43276686],[-0.25211945],[-0.86553372]],1]
    ref = [1,1,1,1]

    H, mu, phi, nu = a.RunPCA(test_input, 0.5)
    res.append(np.min(np.isclose(H,ref_output[0],atol=1e-5)))
    res.append(np.min(np.isclose(mu,ref_output[1],atol=1e-5)))
    res.append(np.min(np.isclose(phi,ref_output[2],atol=1e-5)))
    res.append(np.min(np.isclose(nu,ref_output[3],atol=1e-5)))


def test_PLoM_RunKDE():
    """
    Testing PLoM.RunKDE()
    """
    res = []
    test_input = np.array([[1,1],[0,1],[2,4]])
    ref_output = [[0.8773066621237415, 0.13452737030512696, 0.7785858648409519],[[1,0],[0,1]],[[1,0],[0,1]]]
    ref = [1,1,1]

    a = PLoM(data=None)
    s_v, c_v, hat_s_v, K, b = a.RunKDE(test_input, 1e-5)
    res.append(np.min(np.isclose([s_v,c_v,hat_s_v],ref_output[0],atol=1e-5)))
    res.append(np.min(np.isclose(K,ref_output[1],atol=1e-5)))
    res.append(np.min(np.isclose(b,ref_output[2],atol=1e-5)))

    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)

    
def test_PLoM_DiffMaps():
    """
    Testing PLoM.DiffMaps()
    """

    res = []
    test_input = [np.array([[1,0.5],[0.5,1]]), np.array([[1,0.5],[0.5,1]]), np.array([[1.5, 0.], [0., 1.5]])]
    ref_output = [np.array([[ 0.57735027, -0.57735027],[0.57735027, 0.57735027]]), 3, [[0.8660254,-0.8660254],[0.8660254,0.8660254]],[[1.29903811,-0.4330127],[1.29903811,0.4330127]]]
    ref = [1,1,1,1]

    a = PLoM(data=None)
    g, m, a, Z = a.DiffMaps(test_input[0],test_input[1],test_input[2])
    res.append(np.min(np.isclose(g,ref_output[0],atol=1e-5)))
    res.append(np.min(np.isclose(m,ref_output[1],atol=1e-5)))
    res.append(np.min(np.isclose(a,ref_output[2],atol=1e-5)))
    res.append(np.min(np.isclose(Z,ref_output[3],atol=1e-5)))
    
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)


def test_PLoM_ISDEGeneration():
    """
    Testing PLoM.ISDEGeneration()
    """
    res = []
    ref_output = pd.read_csv(os.path.join(cur_dir,'other_test/X_new.csv'),header=0,index_col=0).values
    ref = [1]

    a = PLoM(data=os.path.join(cur_dir,base_input_path,'input_test.json'))
    a.ConfigTasks()
    a.RunAlgorithm(seed_num=10)
    res.append(np.min(np.isclose(a.Xnew.T,ref_output,1e-5)))

    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)
