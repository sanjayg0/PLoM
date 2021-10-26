#!/usr/bin/env python
# coding: utf-8

# ## Probabilistic Learning on Manifolds (MSA of a 12-story RC frame)
# 
# In this example, raw data from MSA of a 12-story RC frame is used as sample input. The interested engineering demand parameters include maximum story drift ratio and peak floor acceleration. The intensity measures for quantifying the ground motion characeteristics include pseudo spectral acceleration $Sa(T_1)$, response spectral shape measture $SaRatio$, and 5-75% significant duration $D_{S5-75}$.
# 
# The input sample included 472 data points (from six intensity levels in MSA) and each features an unique vector of intensity measures and engineering demand parameters. The goal is to use PLoM learn the data structure and generate more samples whose key statistics (i.e., mean and covariance) are consistent with the input sample.

# ### Import python modules

# In[1]:


import numpy as np
import random
import time
from math import pi
import pandas as pd
from ctypes import *
import matplotlib.pyplot as plt
import sys


# ### Import PLoM modules

# In[2]:


sys.path.insert(1, '../../')
from PLoM import *


# ### Load Multi-Stripe Analysis (MSA) Data
# MSA data are loaded via a comma-separate value (csv) file. The first row contains column names for both predictors (X) and responses (y). The following rows are input sample data. Users are expected to specif the csv filename.

# In[3]:


# Filename
filename = './data/response_rcf12_msa_la_nc.csv'
model = PLoM(model_name='FullMSA', data=filename, col_header=True, plot_tag=True)


# ### Configuring tasks
# Please specify tasks to run - the list of tasks can be run in sqeunce or invidivdual tasks can be run separately.

# In[4]:


tasks = ['DataNormalization','RunPCA','RunKDE','ISDEGeneration']


# ### Step 0: Scaling the data

# In[5]:


# Configure the task
model.ConfigTasks(['DataNormalization'])
# Launch the run
model.RunAlgorithm()


# ### Step 1: Principal Component Analysis (PCA)

# In[6]:


# Tolerance for truncating principal components
tol_pca = 1e-6
# Configure the task
model.ConfigTasks(['RunPCA'])
# Launch the run
model.RunAlgorithm(epsilon_pca=tol_pca)


# ### Step 2: Kernel Density Estimation (KDE)

# In[7]:


# Smoothing parameter in the KDE
sp = 25
# Configure the task
model.ConfigTasks(['RunKDE'])
# Launch the run
model.RunAlgorithm(epsilon_kde=sp)


# ### Step 3: Create the generator

# In[8]:


# Extra parameters for ISDE generation
new_sample_num_ratio = 10
tol_PCA2 = 1e-5
# Configure the task
model.ConfigTasks(['ISDEGeneration'])
# Launch the run
model.RunAlgorithm(n_mc = new_sample_num_ratio, tol_PCA2 = tol_PCA2)


# ### Step 4: Exporting data

# In[9]:


# Available data list
model.export_results()


# In[10]:


# Pick up the original and new realizations, X0 and X_new
model.export_results(data_list=['/X0','/X_new'], file_format_list=['csv','csv'])


# ### Post-processing
# We would like to check the basic statistics of the input sample (i.e., MSA) and the generated new realizations by PLoM. The key metrics include the median, standard deviation, and correlation coefficient matrix of different structural responses.

# In[11]:


# Load results
df_msa = pd.read_csv('../../RunDir/FullMSA/DataOut/X0.csv')
df_plom = pd.read_csv('../../RunDir/FullMSA/DataOut/X_new.csv')
print(df_msa.head)
print(df_plom.head)


# In[12]:


# Maximum story drift ratio in the building
# Note: the SDR are in the rows # 4 ~ # 15
sa_msa = df_msa['Sa']
sa_plom = df_plom['Sa']
col_list = ['SDR'+str(x+1) for x in range(12)]
sdr_max_msa = np.max(df_msa[col_list], axis=1)
sdr_max_plom = np.max(df_plom[col_list], axis=1)


# In[13]:


iml = [0.1690, 0.2594, 0.3696, 0.5492, 0.7131, 0.9000]
sdr_max_med_msa = [np.exp(np.mean(np.log(sdr_max_msa[df_msa['Sa']==iml_x]))) for iml_x in iml]
sdr_max_std_msa = [np.std(np.log(sdr_max_msa[df_msa['Sa']==iml_x])) for iml_x in iml]

iml2 = np.linspace(np.min(iml),np.max(iml),500)
num_nb = 100
sdr_max_med_plom = [np.exp(np.mean(np.log(sdr_max_plom[abs(np.log(df_plom['Sa'])-np.log(iml_x)).argsort()[0:num_nb]]))) for iml_x in iml2]
sdr_max_std_plom = [np.std(np.log(sdr_max_plom[abs(np.log(df_plom['Sa'])-np.log(iml_x)).argsort()[0:num_nb]])) for iml_x in iml2]

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(sdr_max_plom, sa_plom, 'bo', alpha=0.1, label='PLoM')
ax.plot(sdr_max_msa, sa_msa, 'ks', label='MSA')
ax.plot(sdr_max_med_msa, iml, 'k-', label='MSA median')
ax.plot(sdr_max_med_plom, iml2, 'r-', label='PLoM moving median')
ax.plot(sdr_max_med_msa*np.exp(sdr_max_std_msa), iml, 'k--', label='MSA $\pm$1$\sigma$')
ax.plot(sdr_max_med_plom*np.exp(sdr_max_std_plom), iml2, 'r:', label='PLoM moving $\pm$1$\sigma$')
ax.plot(sdr_max_med_msa/np.exp(sdr_max_std_msa), iml, 'k--')
ax.plot(sdr_max_med_plom/np.exp(sdr_max_std_plom), iml2, 'r:')

ax.set_xlabel('Max. SDR (in/in)')
ax.set_ylabel('Sa (g)')
ax.legend()
ax.grid()
plt.show()


# In[14]:


# Peak story acceleration in the building
# Note: the PFA are in the rows # 16 ~ # 27
col_list = ['PFA'+str(x+1) for x in range(12)]
pfa_max_msa = np.max(df_msa[col_list], axis=1)
pfa_max_plom = np.max(df_plom[col_list], axis=1)

iml = [0.1690, 0.2594, 0.3696, 0.5492, 0.7131, 0.9000]
pfa_max_med_msa = [np.exp(np.mean(np.log(pfa_max_msa[sa_msa==iml_x]))) for iml_x in iml]
pfa_max_std_msa = [np.std(np.log(pfa_max_msa[sa_msa==iml_x])) for iml_x in iml]

iml2 = np.linspace(np.min(iml),np.max(iml),500)
num_nb = 100
pfa_max_med_plom = [np.exp(np.mean(np.log(pfa_max_plom[abs(np.log(df_plom['Sa'])-np.log(iml_x)).argsort()[0:num_nb]]))) for iml_x in iml2]
pfa_max_std_plom = [np.std(np.log(pfa_max_plom[abs(np.log(df_plom['Sa'])-np.log(iml_x)).argsort()[0:num_nb]])) for iml_x in iml2]

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(pfa_max_plom, sa_plom, 'bo', alpha=0.1, label='PLoM')
ax.plot(pfa_max_msa, sa_msa, 'ks', label='MSA')
ax.plot(pfa_max_med_msa, iml, 'k-', label='MSA median')
ax.plot(pfa_max_med_plom, iml2, 'r-', label='PLoM moving median')
ax.plot(pfa_max_med_msa*np.exp(pfa_max_std_msa), iml, 'k--', label='MSA $\pm$1$\sigma$')
ax.plot(pfa_max_med_plom*np.exp(pfa_max_std_plom), iml2, 'r:', label='PLoM moving $\pm$1$\sigma$')
ax.plot(pfa_max_med_msa/np.exp(pfa_max_std_msa), iml, 'k--')
ax.plot(pfa_max_med_plom/np.exp(pfa_max_std_plom), iml2, 'r:')

ax.set_xlabel('Max. PFA (g)')
ax.set_ylabel('Sa (g)')
ax.legend()
ax.grid()
plt.show()


# In[15]:


# Story-based responses (SDR)
x0 = df_msa.iloc[:,1:].T
x_c = df_plom.iloc[:,1:].T
sa_msa = x0.iloc[0,:]
sa_plom = x_c.iloc[0,:]
iml = [0.1690, 0.2594, 0.3696, 0.5492, 0.7131, 0.9000]
for i in range(12):
    sdr_cur_msa = x0.iloc[i+3,:]
    sdr_cur_plom = x_c.iloc[i+3,:]
    sdr_cur_med_msa = [np.exp(np.mean(np.log(sdr_cur_msa[x0.iloc[0,:]==iml_x]))) for iml_x in iml]
    sdr_cur_std_msa = [np.std(np.log(sdr_cur_msa[x0.iloc[0,:]==iml_x])) for iml_x in iml]

    iml2 = iml
    num_nb = 100
    sdr_cur_med_plom = [np.exp(np.mean(np.log(sdr_cur_plom[abs(np.log(x_c.iloc[0,:])-np.log(iml_x)).argsort()[0:num_nb]]))) for iml_x in iml2]
    sdr_cur_std_plom = [np.std(np.log(sdr_cur_plom[abs(np.log(x_c.iloc[0,:])-np.log(iml_x)).argsort()[0:num_nb]])) for iml_x in iml2]
    
    plt.figure(figsize=(16,4))
    for j in range(6):
        plt.subplot(1,6,j+1)
        ax = plt.gca()
        pd.Series(sdr_cur_plom).plot.kde(color='b', bw_method=0.5)
        pd.Series(sdr_cur_msa).plot.kde(color='k', bw_method=0.5)
        ax.plot([sdr_cur_med_msa[j], sdr_cur_med_msa[j]], [0, 100], 'k-')
        ax.plot([sdr_cur_med_msa[j]*np.exp(sdr_cur_std_msa[j]), sdr_cur_med_msa[j]*np.exp(sdr_cur_std_msa[j])], [0, 100], 'k--')
        ax.plot([sdr_cur_med_msa[j]/np.exp(sdr_cur_std_msa[j]), sdr_cur_med_msa[j]/np.exp(sdr_cur_std_msa[j])], [0, 100], 'k--')
        ax.plot([sdr_cur_med_plom[j], sdr_cur_med_plom[j]], [0, 100], 'b-')
        ax.plot([sdr_cur_med_plom[j]*np.exp(sdr_cur_std_plom[j]), sdr_cur_med_plom[j]*np.exp(sdr_cur_std_plom[j])], [0, 100], 'b--')
        ax.plot([sdr_cur_med_plom[j]/np.exp(sdr_cur_std_plom[j]), sdr_cur_med_plom[j]/np.exp(sdr_cur_std_plom[j])], [0, 100], 'b--')
        ax.set_xlim(0.0, 0.05)
        #ax.set_ylim(0.0, 200)
        ax.grid()
        ax.set_xlabel('SDR (in/in)')
        if j > 0:
            ax.set_yticklabels('')
            ax.set_ylabel('')
        if j == 2:
            ax.set_title('Max. SDR (in/in) of Story #'+str(i+1))


# In[16]:


# Story-based responses (PFA)
iml = [0.1690, 0.2594, 0.3696, 0.5492, 0.7131, 0.9000]
for i in range(12):
    pfa_cur_msa = x0.iloc[i+15,:]
    pfa_cur_plom = x_c.iloc[i+15,:]
    pfa_cur_med_msa = [np.exp(np.mean(np.log(pfa_cur_msa[x0.iloc[0,:]==iml_x]))) for iml_x in iml]
    pfa_cur_std_msa = [np.std(np.log(pfa_cur_msa[x0.iloc[0,:]==iml_x])) for iml_x in iml]

    iml2 = iml
    num_nb = 100
    pfa_cur_med_plom = [np.exp(np.mean(np.log(pfa_cur_plom[abs(np.log(x_c.iloc[0,:])-np.log(iml_x)).argsort()[0:num_nb]]))) for iml_x in iml2]
    pfa_cur_std_plom = [np.std(np.log(pfa_cur_plom[abs(np.log(x_c.iloc[0,:])-np.log(iml_x)).argsort()[0:num_nb]])) for iml_x in iml2]
    
    plt.figure(figsize=(16,4))
    for j in range(6):
        plt.subplot(1,6,j+1)
        ax = plt.gca()
        pd.Series(pfa_cur_plom).plot.kde(color='b', bw_method=0.5)
        pd.Series(pfa_cur_msa).plot.kde(color='k', bw_method=0.5)
        ax.plot([pfa_cur_med_msa[j], pfa_cur_med_msa[j]], [0, 100], 'k-')
        ax.plot([pfa_cur_med_msa[j]*np.exp(pfa_cur_std_msa[j]), pfa_cur_med_msa[j]*np.exp(pfa_cur_std_msa[j])], [0, 100], 'k--')
        ax.plot([pfa_cur_med_msa[j]/np.exp(pfa_cur_std_msa[j]), pfa_cur_med_msa[j]/np.exp(pfa_cur_std_msa[j])], [0, 100], 'k--')
        ax.plot([pfa_cur_med_plom[j], pfa_cur_med_plom[j]], [0, 100], 'b-')
        ax.plot([pfa_cur_med_plom[j]*np.exp(pfa_cur_std_plom[j]), pfa_cur_med_plom[j]*np.exp(pfa_cur_std_plom[j])], [0, 100], 'b--')
        ax.plot([pfa_cur_med_plom[j]/np.exp(pfa_cur_std_plom[j]), pfa_cur_med_plom[j]/np.exp(pfa_cur_std_plom[j])], [0, 100], 'b--')
        ax.set_xlim(0.0, 1)
        ax.set_ylim(0.0, 10)
        ax.grid()
        ax.set_xlabel('PFA (g)')
        if j > 0:
            ax.set_yticklabels('')
            ax.set_ylabel('')
        if j == 2:
            ax.set_title('PFA (g) of Story #'+str(i+1))


# In[17]:


# Correlation coefficient matrix
c_msa = np.corrcoef(x0)
c_plom = np.corrcoef(x_c)
c_combine = c_msa
tmp = np.triu(c_plom).flatten()
tmp = tmp[tmp != 0]
c_combine[np.triu_indices(27)] = tmp

# Plot covariance matrix
fig, ax = plt.subplots(figsize=(8,6))
ctp = ax.contourf(c_combine[3:,3:], cmap=plt.cm.hot, levels=1000)
ctp.set_clim(0,1)
ax.plot([0, 23], [0, 23], 'k--')
ax.set_xticks(list(range(24)))
ax.set_yticks(list(range(24)))
ax.set_xticklabels(x0.index.tolist()[3:], fontsize=8, rotation=45)
ax.set_yticklabels(x0.index.tolist()[3:], fontsize=8, rotation=45)
ax.set_title('Covariance matrix comparison')
ax.grid()
cbar = fig.colorbar(ctp,ticks=[x/10 for x in range(11)])
plt.show()

# Plot the cross-section of correlation matrix
fig, ax = plt.subplots(figsize=(6,4))
ax.plot([0],[0],'k-',label='MSA')
ax.plot([0],[0],'r:',label='PLoM')
for i in range(24):
    ax.plot(np.array(range(24)),c_msa[i+3][3:],'k-')
    ax.plot(np.array(range(24)),c_plom[i+3][3:],'r:')
ax.set_xticks(list(range(24)))
ax.set_xticklabels(x0.index.tolist()[3:], fontsize=8, rotation=45)
ax.set_ylabel('Correlation coefficient')
ax.set_ylim([0,1])
ax.set_xlim([0,23])
ax.legend()
ax.grid()
plt.show()


# In[ ]:




