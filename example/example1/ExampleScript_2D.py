#!/usr/bin/env python
# coding: utf-8

# ## Probabilistic Learning on Manifolds (Simple example in 2 dimensions)
# 
# In this example, raw invented data is used as sample input. This is just a user guide to see how the package and algorithm work and see their possible applications.
# 
# The input sample included 300 data points and each features a unique vector of some independent variables and some other linearly related variables. The goal is to use PLoM learn the data structure and generate more samples whose key statistics (i.e., mean and covariance) are consistent with the input sample.

# ### Import python modules

# In[16]:


import numpy as np
import random
import time
from math import pi
import pandas as pd
from ctypes import *
import matplotlib.pyplot as plt
import sys
plt.ion()


# ### Import PLoM modules

# In[17]:


sys.path.insert(1, '../../')
from PLoM import *


# ### Load Data
# Data are loaded via a comma-separate value (csv) file. The first row contains column names for all the variables. The following rows are input sample data. Users are expected to specif the csv filename.

# In[18]:


# Filename
filename = './data/data_example2D.csv'
model = PLoM(model_name='Example0_constraints', data=filename, col_header=True, plot_tag=True, runDiffMaps = False)
model.add_constraints('./data/constraints_example2D.py')


# ### Configuring tasks
# Please specify tasks to run - the list of tasks can be run in sqeunce or invidivdual tasks can be run separately.

# In[19]:


tasks = ['DataNormalization','RunPCA','RunKDE','ISDEGeneration']


# ### Step 0: Scaling the data

# In[20]:


# Configure the task
model.ConfigTasks(['DataNormalization'])
# Launch the run
model.RunAlgorithm()


# ### Step 1: Principal Component Analysis (PCA)

# In[21]:


# Tolerance for truncating principal components
tol_pca = 1e-12
# Configure the task
model.ConfigTasks(['RunPCA'])
# Launch the run
model.RunAlgorithm(epsilon_pca=tol_pca)


# ### Step 2: Kernel Density Estimation (KDE)
# 
# In this step there is an optional argument for the RunAlgorithm process that involves the diffusion maps basis. This step can both be run or not, anddepending on the requirements it can be neglected. It can be noted that the default value of *runDiffMaps* is *True*.

# In[22]:


# Smoothing parameter in the KDE
sp = 25
# Configure the task
model.ConfigTasks(['RunKDE'])

# Launch the run
model.RunAlgorithm(epsilon_kde=sp)


# ### Step 3: Create the generator

# In[23]:


# Extra parameters for ISDE generation
new_sample_num_ratio = 10
tol_PCA2 = 1e-5
# Configure the task
model.ConfigTasks(['ISDEGeneration'])
# Launch the run
model.RunAlgorithm(n_mc = new_sample_num_ratio, tol_PCA2 = tol_PCA2, tol = 0.02, max_iter = 50)


# ### Step 4: Exporting data

# In[24]:


# Available data list
model.export_results()


# In[25]:


# Pick up the original and new realizations, X0 and X_new
model.export_results(data_list=['/X0','/X_new'], file_format_list=['csv','csv'])


# ### Post-processing
# We would like to check the basic statistics of the input sample (i.e., MSA) and the generated new realizations by PLoM. The key metrics include the median, standard deviation, and correlation coefficient matrix of different structural responses.

# In[26]:


# Load results

#write realtive paths

df_initial = pd.read_csv('../../RunDir/Example0_constraints/DataOut/X0.csv')
df_plom = pd.read_csv('../../RunDir/Example0_constraints/DataOut/X_new.csv')
print(df_initial.head)
print(df_plom.head)


# In[27]:


#plot of the first two variables
import matplotlib.patches as mpatches
plt.figure()
plt.plot(df_plom.iloc[:,1].to_numpy(), df_plom.iloc[:,2].to_numpy(), 'rx')
plt.plot(df_initial.iloc[:,1].to_numpy(), df_initial.iloc[:,2].to_numpy(), 'bo')
plt.xlabel('x')
plt.ylabel('y')
red_patch = mpatches.Patch(color='red', label='new X')
blue_patch = mpatches.Patch(color='blue', label='X')
plt.legend(handles=[red_patch, blue_patch])
plt.show()


# In[28]:


import matplotlib.patches as mpatches
plt.figure()
plt.plot(model.errors)
plt.xlabel('# iteration')
plt.ylabel('error')
plt.semilogy()
plt.show()


# In[29]:


model.errors


# In[ ]:




