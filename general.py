# Constants, variables, and methods that are commonly used

import os
from datetime import datetime
import numpy as np
import pandas as pd

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
        self._basic()
        self.dir_export = self._create_export_dir()
            
    
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
            store.close()

            return item


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