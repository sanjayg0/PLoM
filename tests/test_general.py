"""
These are unit and integration tests on the general module.
"""

import pytest
import numpy as np
import json
import os
import inspect
import pandas as pd
import shutil
# Importing all in the general module
from general import *
# Current directory
cur_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# Testing input directory
base_input_path = 'resources'

def test_Logfile_init():
    """
    Testing the Logfile.__init__
    """
    res = []
    ref = [1]
    logdir = cur_dir
    logname = 'test.log'
    if os.path.isfile(os.path.join(logdir, logname)):
        os.remove(os.path.join(logdir, logname))
    test_log = Logfile(logfile_dir=logdir, logfile_name=logname, screen_msg=False)
    res.append(int(os.path.isfile(os.path.join(logdir, logname))))
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)
    # Clean
    if os.path.isfile(os.path.join(logdir, logname)):
        os.remove(os.path.join(logdir, logname))


def test_Logfile_write_msg():
    """
    Testing the Logfile.write_msg
    """
    res = []
    ref_output = ['pytest msg','TESTING','\t']
    ref = [1, 1, 1]
    logdir = cur_dir
    logname = 'test.log'
    if os.path.isfile(os.path.join(logdir, logname)):
        os.remove(os.path.join(logdir, logname))
    test_log = Logfile(logfile_dir=logdir, logfile_name=logname, screen_msg=False)
    test_log.write_msg(msg=ref_output[0])
    test_log.write_msg(msg_type=ref_output[1])
    test_log.write_msg(msg_level=1)
    with open(os.path.join(logdir, logname),'r') as f:
        lines = f.readlines()
        cur_line = 0
        for line in reversed(lines):
            cur_line = cur_line+1
            res.append(int(ref_output[-cur_line] in line))
            if cur_line == 3:
                break
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)
    # Clean
    if os.path.isfile(os.path.join(logdir, logname)):
        os.remove(os.path.join(logdir, logname))


def test_Logfile_delete_logfile():
    """
    Testing the Logfile.delete_logfile
    """
    res = []
    ref = [0]
    logdir = cur_dir
    logname = 'test.log'
    if os.path.isfile(os.path.join(logdir, logname)):
        os.remove(os.path.join(logdir, logname))
    test_log = Logfile(logfile_dir=logdir, logfile_name=logname, screen_msg=False)
    test_log.delete_logfile()
    res.append(int(os.path.isfile(os.path.join(logdir, logname))))
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)
    # Clean
    if os.path.isfile(os.path.join(logdir, logname)):
        os.remove(os.path.join(logdir, logname))


def test_DBServer_init():
    """
    Testing the DBServer.__init__()
    """
    res = []
    ref = [1, 1, 1, 1]
    dbdir = cur_dir
    dbname = 'test.h5'
    if os.path.isfile(os.path.join(dbdir, dbname)):
        os.remove(os.path.join(dbdir, dbname))
    if os.path.isdir(os.path.join(dbdir, 'DataOut')):
        shutil.rmtree(os.path.join(dbdir, 'DataOut'))
    test_db = DBServer(db_dir=dbdir, db_name=dbname)
    # Checking if the file exists
    res.append(int(os.path.isfile(os.path.join(dbdir, dbname))))
    # Checking if the basic property exists in the db
    store = pd.HDFStore(os.path.join(dbdir, dbname),'r')
    res.append('/basic' in store.keys())
    # Checking if the ConstraintsFile exists in the db
    res.append('/constraints_file' in store.keys())
    store.close()
    # Checking if the export directories exist
    res.append(int(os.path.isdir(os.path.join(dbdir, 'DataOut'))))
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)
    # Clean
    if os.path.isfile(os.path.join(dbdir, dbname)):
        os.remove(os.path.join(dbdir, dbname))
    if os.path.isdir(os.path.join(dbdir, 'DataOut')):
        shutil.rmtree(os.path.join(dbdir, 'DataOut'))


def test_DBServer_get_item_adds():
    """
    Testing the DBServer.get_item_adds()
    """
    res = []
    ref = [1]
    dbdir = cur_dir
    dbname = 'test.h5'
    if os.path.isfile(os.path.join(dbdir, dbname)):
        os.remove(os.path.join(dbdir, dbname))
    if os.path.isdir(os.path.join(dbdir, 'DataOut')):
        shutil.rmtree(os.path.join(dbdir, 'DataOut'))
    test_db = DBServer(db_dir=dbdir, db_name=dbname)
    res.append(test_db.get_item_adds()==ITEM_ADDS)
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)
    # Clean
    if os.path.isfile(os.path.join(dbdir, dbname)):
        os.remove(os.path.join(dbdir, dbname))
    if os.path.isdir(os.path.join(dbdir, 'DataOut')):
        shutil.rmtree(os.path.join(dbdir, 'DataOut'))


def test_DBServer_add_item():
    """
    Testing the DBServer.add_item()
    """
    res = []
    ref = [1, 1, 1, 1, 1, 1]
    test_item = [np.array([1]),np.array([[1],[1]]).T,np.array([[1],[1]])]
    dbdir = cur_dir
    dbname = 'test.h5'
    if os.path.isfile(os.path.join(dbdir, dbname)):
        os.remove(os.path.join(dbdir, dbname))
    if os.path.isdir(os.path.join(dbdir, 'DataOut')):
        shutil.rmtree(os.path.join(dbdir, 'DataOut'))
    test_db = DBServer(db_dir=dbdir, db_name=dbname)
    test_db.add_item(item_name='test0',item=test_item[0])
    test_db.add_item(item_name='test1',col_names=['c1','c2'],item=test_item[1],data_shape=test_item[1].shape)
    test_db.add_item(item_name='test2',item=test_item[2],data_shape=test_item[2].shape)
    store = pd.HDFStore(os.path.join(dbdir, dbname),'r')
    res.append(int(np.array_equal(store['/test0'].values[0],test_item[0])))
    res.append(int(np.array_equal(store['/test1'].values,test_item[1])))
    res.append(int(np.array_equal(store['/test2'].values,test_item[2])))
    res.append(int(tuple([x[0] for x in store['/DS_test0'].values.tolist()])==test_item[0].shape))
    res.append(int(tuple([x[0] for x in store['/DS_test1'].values.tolist()])==test_item[1].shape))
    res.append(int(tuple([x[0] for x in store['/DS_test2'].values.tolist()])==test_item[2].shape))
    store.close()
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)
    # Clean
    if os.path.isfile(os.path.join(dbdir, dbname)):
        os.remove(os.path.join(dbdir, dbname))
    if os.path.isdir(os.path.join(dbdir, 'DataOut')):
        shutil.rmtree(os.path.join(dbdir, 'DataOut'))


def test_DBServer_get_item():
    """
    Testing the DBServer.get_item()
    """
    res = []
    ref = [1, 1, 1]
    test_item = [np.array([1]),np.array([[1],[1]]).T,np.array([[1],[1]])]
    dbdir = cur_dir
    dbname = 'test.h5'
    if os.path.isfile(os.path.join(dbdir, dbname)):
        os.remove(os.path.join(dbdir, dbname))
    if os.path.isdir(os.path.join(dbdir, 'DataOut')):
        shutil.rmtree(os.path.join(dbdir, 'DataOut'))
    test_db = DBServer(db_dir=dbdir, db_name=dbname)
    test_db.add_item(item_name='test0',item=test_item[0])
    test_db.add_item(item_name='test1',col_names=['c1','c2'],item=test_item[1],data_shape=test_item[1].shape)
    test_db.add_item(item_name='test2',item=test_item[2],data_shape=test_item[2].shape)
    res.append(int(np.array_equal(test_db.get_item('test0'),test_item[0])))
    res.append(int(np.array_equal(test_db.get_item('test1'),test_item[1])))
    res.append(int(np.array_equal(test_db.get_item('test2'),test_item[2])))
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)
    # Clean
    if os.path.isfile(os.path.join(dbdir, dbname)):
        os.remove(os.path.join(dbdir, dbname))
    if os.path.isdir(os.path.join(dbdir, 'DataOut')):
        shutil.rmtree(os.path.join(dbdir, 'DataOut'))


def test_DBServer_get_item_shape():
    """
    Testing the DBServer.get_item_shape()
    """
    res = []
    ref = [1, 1, 1]
    test_item = [np.array([1]),np.array([[1],[1]]).T,np.array([[1],[1]])]
    dbdir = cur_dir
    dbname = 'test.h5'
    if os.path.isfile(os.path.join(dbdir, dbname)):
        os.remove(os.path.join(dbdir, dbname))
    if os.path.isdir(os.path.join(dbdir, 'DataOut')):
        shutil.rmtree(os.path.join(dbdir, 'DataOut'))
    test_db = DBServer(db_dir=dbdir, db_name=dbname)
    test_db.add_item(item_name='test0',item=test_item[0])
    test_db.add_item(item_name='test1',col_names=['c1','c2'],item=test_item[1],data_shape=test_item[1].shape)
    test_db.add_item(item_name='test2',item=test_item[2],data_shape=test_item[2].shape)
    res.append(int(np.array_equal(tuple([x[0] for x in test_db.get_item_shape('test0').values.tolist()]),test_item[0].shape)))
    res.append(int(np.array_equal(tuple([x[0] for x in test_db.get_item_shape('test1').values.tolist()]),test_item[1].shape)))
    res.append(int(np.array_equal(tuple([x[0] for x in test_db.get_item_shape('test2').values.tolist()]),test_item[2].shape)))
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)
    # Clean
    if os.path.isfile(os.path.join(dbdir, dbname)):
        os.remove(os.path.join(dbdir, dbname))
    if os.path.isdir(os.path.join(dbdir, 'DataOut')):
        shutil.rmtree(os.path.join(dbdir, 'DataOut'))


def test_DBServer_get_name_list():
    """
    Testing the DBServer.get_name_list()
    """
    res = []
    ref = [1, 1, 1, 1, 1, 1, 1, 1]
    test_item = [np.array([1]),np.array([[1],[1]]).T,np.array([[1],[1]])]
    test_name_list = ['/basic','/constraints_file','/test0','/test1','/test2','/DS_test0','/DS_test1','/DS_test2']
    dbdir = cur_dir
    dbname = 'test.h5'
    if os.path.isfile(os.path.join(dbdir, dbname)):
        os.remove(os.path.join(dbdir, dbname))
    if os.path.isdir(os.path.join(dbdir, 'DataOut')):
        shutil.rmtree(os.path.join(dbdir, 'DataOut'))
    test_db = DBServer(db_dir=dbdir, db_name=dbname)
    test_db.add_item(item_name='test0',item=test_item[0])
    test_db.add_item(item_name='test1',col_names=['c1','c2'],item=test_item[1],data_shape=test_item[1].shape)
    test_db.add_item(item_name='test2',item=test_item[2],data_shape=test_item[2].shape)
    for i in test_name_list:
        res.append(int(i in test_db.get_name_list()))
    # Check
    np.testing.assert_allclose(res, ref, atol=1e-5)
    # Clean
    if os.path.isfile(os.path.join(dbdir, dbname)):
        os.remove(os.path.join(dbdir, dbname))
    if os.path.isdir(os.path.join(dbdir, 'DataOut')):
        shutil.rmtree(os.path.join(dbdir, 'DataOut'))