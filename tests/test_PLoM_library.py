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
# Importing all in the general module
import PLoM_library as plom
# Current directory
cur_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# Testing input directory
base_input_path = 'input_test'
base_constraints_path = 'constraints_test'

def test_kernel():
    """
    Testing the kernel(x, y, epsilon)
    """
    
    res = plom.kernel(np.array([1,0]), np.array([1,0]), 0.5)
    ref = [1.0]
    np.testing.assert_allclose(res, ref, atol=1e-5)


def test_K():
    """
    K(eta, epsilon)
    """
    res = plom.K((np.array([[1,1],[1,1]])), 3)
    ref = (np.array([[1., 1.], [1., 1.]]), np.array([[2., 0.], [0., 2.]]))
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)
    
    
def test_g():
    """
    g(K, b)
    """
    [res1, res2] = plom.g((np.array([[1,0.5],[0.5,1]])), np.array([[1.5, 0.], [0., 1.5]]))
    [ref1, ref2]  = (np.array([[ 0.57735027, -0.57735027], [ 0.57735027,  0.57735027]]), np.array([1. , 0.33333333]))
    # Check
    np.testing.assert_allclose(res1, ref1, atol=1e-5)
    np.testing.assert_allclose(res2, ref2, atol=1e-5)
    
def test_m():
    """
    m(eigenvalues)
    """
    res = plom.m(np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025]))
    ref = 11
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)
    
def test_mean():
    """
    mean(x)
    """
    res=  plom.mean(np.array([[1,1],[0,1],[2,4]]))
    ref = np.array([[1. ], [0.5], [3. ]])
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)
    
def test_covariance():
    """
    covariance(x)
    """
    res=  plom.covariance(np.array([[1,1],[0,1],[2,4]]))
    ref = np.array([[0. , 0. , 0. ], [0. , 0.5, 1. ], [0. , 1. , 2. ]])
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)    

def test_PCA():
    """
    PCA(x, tol)
    """
    [res1, res2, res3] = plom.PCA(np.array([[1,1],[0,1],[2,4]]), 0.1)
    [ref1, ref2, ref3] = [np.array([[-0.70710678,  0.70710678]]), np.array([1.58113883]), np.array([[-1.13483031e-17], [ 4.47213595e-01], [ 8.94427191e-01]])]
    # Check
    np.testing.assert_allclose(res1, ref1, atol=1e-3) 
    np.testing.assert_allclose(res2, ref2, atol=1e-3) 
    np.testing.assert_allclose(res3, ref3, atol=1e-3) 
    
def test_parameters_kde():
    """
    parameters_kde(eta)
    """
    res = plom.parameters_kde(np.array([[1,1],[0,1],[2,4]]))
    ref = (0.8773066621237415, 0.13452737030512696, 0.7785858648409519)
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5) 

def test_kde():
    """
    kde(y, eta, s_v = None, c_v = None, hat_s_v = None)
    """
    res = plom.kde(np.array([[1, 2, 3]]), np.array([[1,1],[0,1],[2,4]]))
    ref = 0.01940049487135241
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5) 
    
def test_PCA2():
    """
    PCA2(C_h_hat_eta, beta, tol)
    """
    [res1, res2] = plom.PCA2(np.array([[1. , 1. , 1. ], [1. , 4.5, 1.5 ], [1. , 1.5 , 2. ]]), np.array([10, 1, 2]), 0.1)
    [ref1, ref2] = (np.array([-4.53648062,  5.2236145 ]), np.array([[-0.28104828,  0.42570005], [-0.85525695, -0.51768266], [-0.43537043,  0.74214832]]))
    # Check
    np.testing.assert_allclose(res1, ref1, atol=1e-5) 
    np.testing.assert_allclose(res2, ref2, atol=1e-5)
