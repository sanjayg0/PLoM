"""
These are unit and integration tests on the PLoM module.
Note:
1. PLoM calls methods from general whose testing is addressed by test_general,
   so the those are not tested by this script
2. Nested functions are tested from bottom to top and testing low-level functions
   is not included (as duplicated) in testing a high-level method
"""

import pytest
import numpy as np
import json
import os
import inspect
import pandas as pd
import shutil
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
        res.append(int(type(a.g_c(a.X))==ref_output[i][0] and all(a.beta_c==ref_output[i][1]) and np.max(np.array(a.D_x_g_c(a.X)==ref_output[i][2]))))

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

