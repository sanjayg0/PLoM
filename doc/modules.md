## Modules and API

### *general* module
The ***general*** module contains four major classes: **Logfile**, **DBServer**, **Task**, and **TaskList** which are commonly invoked through the 
calculation workflow and **PLoM** algorithm.

#### Logfile
```python
Logfile.__init__(self, logfile_dir = './', logfile_name = 'plom.log', screen_msg = True)
"""
Initializing the logfile
- logfile_dir: default is the same path of the PLoM package
- logfile_name: default is the "plom.log"
- screen_msg: default is to show message on screen
"""
```
```python
Logfile.write_msg(self, msg = '', msg_type = 'RUNNING', msg_level = 0, mode='a'):
"""
Writing running messages
- msg: the message
- msg_type: the type of message 'RUNNING', 'WARNING', 'ERROR'
- msg_level: how many indent tags
- mode: mode of writing the log file
"""
```
```python
Logfile.delete_logfile(self):
"""
Deleting the log file
"""
```

#### DBServer
```python
DBServer.__init__(self, db_dir = './', db_name = 'plom.h5'):
"""
Initializing the database
- db_dir: default is the same path of the PLoM package
- db_name: default is "plom.h5"
"""
```
```python
DBServer.basic(self):
"""
Writing 
"""
```
```python
DBServer._create_export_dir(self):
"""
Creating an export folder
"""
```
```python
DBServer.get_item_adds(self):
"""
Returning the full list of data items
"""
```
```python
DBServer.add_item(self, item_name = None, col_names = None, item = [], data_shape = None, data_type='Data'):
"""
Adding a new data item into database
- item_name: data item name
- col_names: a list of column names of the data
- item: the data item (in nparray)
- data_shape: the data shape (used for syncing data to attributes
- data_type: 'Data' or 'ConstraintsFile'
"""
```
```python
DBServer.get_item(self, item_name = None, table_like=False, data_type='Data'):
"""
Getting a specific data item
- item_name: data item name
- table_like: False -> return ndarray with the data shape; True - return the pandas dataframe
- data_type: 'Data' or 'ConstraintsFile'
"""
```
```python
DBServer.remove_item(self, item_name = None):
"""
Removing an item
- item_name: data item name
"""
```
```python
DBServer.get_item_shape(self, item_name = None):
"""
Getting the shape of a specific data item
- item_name: data item name
"""
```
```python
DBServer.get_name_list(self):
"""
Returning the keys of the database
"""
```
```python
DBServer.export(self, data_name = None, filename = None, file_format = 'csv'):
"""
Exporting the specific data item
- data_name: data tag
- filename: the name of exported file
- file_format: data format
"""
```

#### Task
```python
Task.__init__(self, task_name = None):
"""
Initialization
- task_name: name of the task
"""
```
```python
Task.refresh_status(self):
"""
Refreshing the current status of the task
If any of the previous tasks is not completed, the current task is also not reliable
"""
```
#### TaskList
```python
TaskList.__init__(self):
"""
Initialization
"""
```
```python
TaskList.add_task(self, new_task = None)
"""
Adding a new task to the list
- new_task: a Task()
"""
```
```python
TaskList.refresh_status(self):
"""
Refreshing the tasks' status
If and only if all tasks completed, the task list is completed
"""
```

### *PLoM* module
The ***PLoM*** module houses the major functionality of the PLoM package.

#### Initialization and configuration
```python
PLoM.__init__(self, model_name='plom', data='', separator=',', col_header=False, constraints = None, run_tag = False, plot_tag = False, num_rlz = 5, tol_pca = 1e-6, epsilon_kde = 25):
"""

"""
```

### *PLoM_library* module
