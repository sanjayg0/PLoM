# Constants, variables, and methods that are commonly used

import os
from datetime import datetime
import numpy as np
import pandas as pd

# PLoM path
plom_dir = os.path.dirname(os.path.abspath(__file__))

class Logfile:
    def __init__(self, logfile_dir = plom_dir, logfile_name = 'plom.log', screen_msg = True):
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
        self.write_msg(msg = '--NEW LOG STARTING FROM THIS LINE--')
            
    
    def write_msg(self, msg = '', msg_type = 'RUNNING', msg_level = 0):
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
        with open(self.logfile_path, 'a') as f:
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
    def __init__(self, db_dir = plom_dir, db_name = 'plom.h5'):
        """
        Initializing the database
        - db_dir: default is the same path of the PLoM package
        - db_name: default is "plom.h5"
        """
        self.db_dir = db_dir
        self.db_name = db_name
        self.db_path = os.path.join(self.db_dir, self.db_name)
        self.init_time = datetime.utcnow()
        self._basic()

    
    def _basic(self):
        """
        Writing basic info
        """
        df = pd.DataFrame.from_dict({
            'InitializedTime': [self.init_time],
            'LastEditedTime': [datetime.utcnow()],
            'DBName': [self.db_name]
        }, dtype=str)
        store = pd.HDFStore(self.db_path, 'a')
        df.to_hdf(store, 'basic', mode='a')
        store.close()


    def add_item(self, item_name = None, col_names = None, item = []):
        """
        Adding a new data item into database
        """
        if item.ndim > 1:
            df = pd.DataFrame(item, columns = col_names)
        else:
            if col_names is None:
                col_names = item_name
            df = pd.DataFrame.from_dict({
                col_names: item.tolist()
            })
        if item_name is not None:
            store = pd.HDFStore(self.db_path, 'a')
            df.to_hdf(store, item_name, mode='a')
            store.close()


    def get_item(self, item_name = None):
        """
        Getting a specific data item
        """
        if item_name is not None:
            store = pd.HDFStore(self.db_path, 'r')
            try:
                item = store.get(item_name)
            except:
                item = None

            return item
