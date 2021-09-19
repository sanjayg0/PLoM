# Constants, variables, and methods that are commonly used

import os
from datetime import datetime
import numpy as np
import pandas as pd
from collections import Counter


ITEM_LIST_DATANORM = ['X_range','X_min','X_scaled','X_scaled_mean']
ITEM_LIST_RUNPCA = ['X_PCA','EigenValue_PCA','EigenVector_PCA','NumComp_PCA']
ITEM_LIST_RUNKDE = ['s_v','c_v','hat_s_v','X_KDE','EigenValues_KDE']
ITEM_LIST_DIFFMAPS = ['DiffMaps_g','DiffMaps_m','DiffMaps_a','DiffMaps_Z']
ITEM_LIST_ISDEGENE = ['Errors','X_new']
ITEM_LIST = ['basic']+['constraints_file']+['X0','N','n']+ITEM_LIST_DATANORM+ITEM_LIST_RUNPCA \
    +ITEM_LIST_RUNKDE+ITEM_LIST_DIFFMAPS+ITEM_LIST_ISDEGENE # all variables in the database
ITEM_ADDS = ['/'+x for x in ITEM_LIST] # HDFStore ABSOLUTE path-names
ATTR_LIST = [None,None,'X','N','n',
    'alpha','x_min','X_scaled','x_mean',
    'H','mu','phi','nu',
    's_v','c_v','hat_s_v','K','b',
    'g','m','a','Z',
    'errors','Xnew']
ATTR_MAP = dict(zip(ITEM_ADDS, ATTR_LIST))
FULL_TASK_LIST = ['DataNormalization','RunPCA','RunKDE','DiffMaps','ISDEGeneration']
TASK_ITEM_MAP = {'DataNormalization': ITEM_LIST_DATANORM,
                 'RunPCA': ITEM_LIST_RUNPCA,
                 'RunKDE': ITEM_LIST_RUNKDE,
                 'DiffMaps': ITEM_LIST_DIFFMAPS,
                 'ISDEGeneration': ITEM_LIST_ISDEGENE}


class Logfile:
    def __init__(self, logfile_dir = './', logfile_name = 'plom.log', screen_msg = True):
        """
        Initializing the logfile
        - logfile_dir: default is the same path of the PLoM package
        - logfile_name: default is the "plom.log"
        - screen_msg: default is to show message on screen
        """
        self.logfile_dir = logfile_dir
        self.logfile_name = logfile_name
        self.logfile_path = os.path.join(self.logfile_dir, self.logfile_name)
        self.screen_msg = screen_msg
        # start the log
        self.write_msg(msg = '--NEW LOG STARTING FROM THIS LINE--', mode='w')
            
    
    def write_msg(self, msg = '', msg_type = 'RUNNING', msg_level = 0, mode='a'):
        """
        Writing running messages
        - msg: the message
        - msg_type: the type of message 'RUNNING', 'WARNING', 'ERROR'
        - msg_level: how many indent tags
        """
        indent_tabs = ''.join(['\t']*msg_level)
        decorated_msg = '{} {} {}-MSG {} '.format(datetime.utcnow(), indent_tabs, msg_type, msg)
        if self.screen_msg:
            print(decorated_msg)
        with open(self.logfile_path, mode) as f:
            f.write('\n'+decorated_msg)

    
    def delete_logfile(self):
        """
        Deleting the log file
        """
        if os.path.exists(self.logfile_path):
            os.remove(self.logfile_path)
        else:
            print('The logfile {} does not exist.'.format(self.logfile_path))


class DBServer:
    def __init__(self, db_dir = './', db_name = 'plom.h5'):
        """
        Initializing the database
        - db_dir: default is the same path of the PLoM package
        - db_name: default is "plom.h5"
        """
        self.db_dir = db_dir
        self.db_name = db_name
        self.db_path = os.path.join(self.db_dir, self.db_name)
        if os.path.exists(self.db_path):
            # deleting the old database
            os.remove(self.db_path)
        self.init_time = datetime.utcnow()
        self.item_name_list = []
        self.basic()
        self.dir_export = self._create_export_dir()
        self._item_list = ITEM_LIST
        self._item_adds = ITEM_ADDS
            
    
    def basic(self):
        """
        Writing basic info
        """
        df = pd.DataFrame.from_dict({
            'InitializedTime': [self.init_time],
            'LastEditedTime': [datetime.utcnow()],
            'DBName': [self.db_name],
        }, dtype=str)
        store = pd.HDFStore(self.db_path, 'a')
        df.to_hdf(store, 'basic', mode='a')
        store.close()
        self.add_item(item=[''],data_type='ConstraintsFile')


    def _create_export_dir(self):
        """
        Creating a export folder
        """
        dir_export = os.path.join(self.db_dir,'DataOut')
        try:
            os.makedirs(dir_export, exist_ok=True)
            return dir_export
        except:
            return None


    def get_item_adds(self):
        """
        Returning the full list of data items
        """
        return self._item_adds


    def add_item(self, item_name = None, col_names = None, item = [], data_shape = None, data_type='Data'):
        """
        Adding a new data item into database
        """
        if data_type == 'Data':
            if item.size > 1:
                df = pd.DataFrame(item, columns = col_names)
                dshape = pd.DataFrame(data_shape, columns=['DS_'+item_name])
            else:
                if col_names is None:
                    col_names = item_name
                df = pd.DataFrame.from_dict({
                    col_names: item.tolist()
                })
                dshape = pd.DataFrame.from_dict({
                    'DS_'+col_names: (1,)
                })
            if item_name is not None:
                store = pd.HDFStore(self.db_path, 'a')
                # data item
                df.to_hdf(store, item_name, mode='a')
                # data shape
                dshape.to_hdf(store, 'DS_'+item_name, mode='a')
                store.close()
        elif data_type == 'ConstraintsFile':
            # constraints filename
            cf = pd.DataFrame.from_dict({
                'ConstraintsFile': item
            }, dtype=str)
            store = pd.HDFStore(self.db_path, 'a')
            cf.to_hdf(store, 'constraints_file', mode='a')
            store.close()
        else:
            # Not supported data_type
            return False


    def get_item(self, item_name = None, table_like=False, data_type='Data'):
        """
        Getting a specific data item
        """
        if data_type == 'Data':
            if item_name is not None:
                store = pd.HDFStore(self.db_path, 'r')
                try:
                    item = store.get(item_name)
                    item_shape = tuple([x[0] for x in self.get_item_shape(item_name=item_name).values.tolist()])
                    if not table_like:
                        item = item.to_numpy().reshape(item_shape)                
                except:
                    item = None
                store.close()

                return item
        elif data_type == 'ConstraintsFile':
            store = pd.HDFStore(self.db_path, 'r')
            try:
                item = store.get('/constraints_file')
            except:
                item = None
            store.close()

            return item.values.tolist()[0][0]



    def get_item_shape(self, item_name = None):
        """
        Getting the shape of a specific data item
        """
        
        if item_name is not None:
            store = pd.HDFStore(self.db_path, 'r')
            try:
                item_shape = store.get('DS_'+item_name)
            except:
                item_shape = None
            store.close()

            return item_shape


    def get_name_list(self):
        """
        Returning the keys of the database
        """
        store = pd.HDFStore(self.db_path, 'r')
        try:
            name_list = store.keys()
        except:
            name_list = []
        store.close()
        return name_list


    def export(self, data_name = None, filename = None, file_format = 'csv'):
        """
        Exporting the specific data item
        - data_name: data tag
        - format: data format
        """
        d = self.get_item(item_name = data_name)
        if d is None:
            return 1
        if filename is None:
            filename = os.path.join(self.dir_export,str(data_name).replace('/','')+'.'+file_format)
        else:
            filename = os.path.join(self.dir_export,filename.split('.')[0]+'.'+file_format)
        if file_format == 'csv' or 'txt':
            d.to_csv(filename, header=True, index=True)
        elif file_format == 'json':
            with open(filename, 'w') as f:
                json.dump(d, f)
        else:
            return 2
        return 0


    def convert_format(self, item_name = None, cvt_task = None):
        """
        Converting data format
        - item_name: data item name
        - cvt_task: a string of conveting actions
        """
        



class Task:
    """
    This is a class for managering an individual task in 
    the PLoM running process
    """
    def __init__(self, task_name = None):
        """
        Initialization
        - task_name: name of the task
        """
        self.task_name = task_name # task name
        self.pre_task = None # previous task
        self.next_task = None # next task
        self.full_var_list = [] # key variable list
        self.avail_var_list = [] # current available variables
        self.status = False # task status

    
    def refresh_status(self):
        """
        Refreshing the current status of the task
        If any of the previous tasks is not completed, the current task is also not reliable
        """
        # check the previous task if any
        if self.pre_task:
            if not self.pre_task.refresh_status():
                # previous task not completed -> this task also needs to rerun
                self.status = False

                return self.status
    
        # self-check
        if Counter(self.avail_var_list)==Counter(self.full_var_list) and len(self.avail_var_list):
            self.status = True # not finished
        else:
            self.status = False # finished

        return self.status


class TaskList:
    """
    This is a class for managering a set of tasks
    in a specific order
    """
    def __init__(self):
        self.head_task = None # first task
        self.tail_task = None # last task
        self.status = False # status

    def add_task(self, new_task = None):
        if new_task is None:
            self.head_task = None
            return
        else:
            if self.head_task is None:
                # first task
                self.head_task = new_task
                self.tail_task = new_task
            else:
                # adding a new to the current list
                new_task.pre_task = self.tail_task
                self.tail_task.next_task = new_task
                self.tail_task = new_task

    
    def refresh_status(self):
        """
        Refreshing the tasks' status
        """
        if self.head_task:
            cur_task = self.head_task
            if not cur_task.status:
                self.status = False

                return self.status
            while cur_task.next_task:
                cur_task = cur_task.next_task
                if not cur_task.status:
                    self.status = False

                    return self.status
            
            self.status = True
            
            return self.status             
        
