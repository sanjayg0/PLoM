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
PLoM.__init__(self, model_name='plom', data='', separator=',', col_header=False, constraints = None, run_tag = False, plot_tag = False, num_rlz = 5, tol_pca = 1e-6, epsilon_kde = 25, tol_PCA2 = 1e-5, tol = 1e-6, max_iter = 50, runDiffMaps = True):
"""
Creating a PLoM model
- model_name: user-specified model name
- data: input data path
- separator: separator (if used) in the data file
- col_header: if the data file has column headers
- constraints: constraints file path
- run_tag: if directly run the calculation to generate samples
- plot_tag: if plot figures during the run
- num_rlz: ratio of realizations/given sample size
- tol_pca: tolerance for selecting the number of considered componenets in PCA
- epsilon_kde: smoothing parameter in the kernel density estimation
- tol_PCA2: tolerance used in the PCA for the covariance matrix of constraint-function-valuated data
- tol: convergence tolerance in the iterative solutions of ISDE
- max_iter: maximum number of iterations in the iterative solutions of ISDE
- runDiffMaps: if use the diffusion maps in KDE
"""
```
```python
PLoM._basic_config(self, model_name=None):
"""
Basic setups: creating log file, database server, and the run directory
- model_name: job name (used for database name)
"""
```

#### Data loading
```python
PLoM.initialize_data(self, filename, separator=',', col_header=False, constraints = ''):
"""
Loading the data
- filename: input data path
- separator: separator (if used) in the data file
- col_header: if the data file has column headers
- constraints: constraints file path
"""
```
```python
PLoM.load_data(self, filename, separator=',', col_header=False):
"""
Loading the data
- filename: input data path
- separator: separator (if used) in the data file
- col_header: if the data file has column headers
"""
```
```python
PLoM.add_data(self, filename, separator=',', col_header=False):
"""
Adding new data
- filename: input data path
- separator: separator (if used) in the data file
- col_header: if the data file has column headers
"""
```
```python
PLoM.load_h5(self, filename):
"""
Loading a h5 database
- filename: input data path
"""
```
```python
PLoM._load_h5_plom(self, filename):
"""
Loading a PLoM-formatted h5 database 
- filename: input data path
"""
```
```python
PLoM._load_h5_data_X(self, filename):
"""
Loading a h5 data which is expected to contain X data
- filename: input data path
"""
```
```python
PLoM._sync_data(self):
"""
Sync database data to the current attributes in analysis
"""
```
```python
PLoM._sync_constraints(self):
"""
Sync constraints from dbserver to the attributes
"""
```

#### Constraints
```python
PLoM.add_constraints(self, constraints_file = None):
"""
Adding a set of constraints
- constraints_file: the file path of a python script defining the constraints
"""
```
```python
PLoM.delete_constraints(self):
"""
Removing all current constraints
"""
```

#### Configuring Tasks
```python
PLoM.ConfigTasks(self, task_list = FULL_TASK_LIST):
"""
Creating a task list object
- task_list: a string list of tasks to run
"""
```
```python
PLoM._init_indv_tasks(self):
"""
Initializing individual tasks
"""
```

#### Run Alogrithm
```python
PLoM.RunAlgorithm(self, n_mc = 5, epsilon_pca = 1e-6, epsilon_kde = 25, tol_PCA2 = 1e-5, tol = 1e-6, max_iter = 50, plot_tag = False, runDiffMaps = True):
- num_rlz: ratio of realizations/given sample size
- tol_pca: tolerance for selecting the number of considered componenets in PCA
- epsilon_kde: smoothing parameter in the kernel density estimation
- tol_PCA2: tolerance used in the PCA for the covariance matrix of constraint-function-valuated data
- tol: convergence tolerance in the iterative solutions of ISDE
- max_iter: maximum number of iterations in the iterative solutions of ISDE
- plot_tag: if plot figures during the run
- runDiffMaps: if use the diffusion maps in KDE
```
```python
PLoM.DataNormalization(self, X):
"""
Normalizing the X 
- X: the data matrix to be normalized
"""
```
```python
PLoM.RunPCA(self, X_origin, epsilon_pca):
"""
Principle Component Analysis
- X_origin: the data matrix
- epsilon_pca: tolerance for selecting the number of considered componenets in PCA
"""
```
```python
PLoM.RunKDE(self, X, epsilon_kde):
"""
Kernel Density Estimation
- X: the data matrix to be reduced
- epsilon_kde: smoothing parameter in the kernel density estimation
"""
```
```python
PLoM.DiffMaps(self, H, K, b):
"""
Diffusion Maps
"""
```
```python
PLoM.ISDEGeneration(self, n_mc = 5, tol_PCA2 = 1e-5, tol = 1e-6, max_iter = 50):
"""
Construction of a nonlinear Ito Stochastic Differential Equation (ISDE) to generate realizations of random variable H
- num_mc: ratio of realizations/given sample size
- tol_PCA2: tolerance used in the PCA for the covariance matrix of constraint-function-valuated data
- tol: convergence tolerance in the iterative solutions of ISDE
- max_iter: maximum number of iterations in the iterative solutions of ISDE
"""
```

#### Postprocessing
```python
PLoM.export_results(self, data_list = [], file_format_list = ['csv']):
"""
Exporting results by the data names
- data_list: list of data names
- file_format_list: list of output formats
"""
```
```python
PLoM.PostProcess(self):
"""
Plotting results
"""
```
