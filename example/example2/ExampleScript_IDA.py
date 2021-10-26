#!/usr/bin/env python
# coding: utf-8

# ## Probabilistic Learning on Manifolds (IDA of a 12-story RC frame)
# 
# In this example, raw data from Incremental Dynamic Analysis (IDA) of a 12-story RC frame are used as the input sample. Similar to the previous example (i.e., example1), the interested engineering demand parameters include maximum story drift ratio and peak floor acceleration. The intensity measures for quantifying the ground motion characeteristics include pseudo spectral acceleration $Sa(T_1)$, response spectral shape measture $SaRatio$, and 5-75% significant duration $D_{S5-75}$.
# 
# The IDA ground motions include 49 records with various $SaRati$ and $D_{S5-75}$, and the entire sample data include 478 points. The goal is (1) to use PLoM learn the data structure and generate more samples whose key statistics (i.e., mean and covariance) are consistent with the input samplem, and (2) to resample the PLoM-estimated EDPs given specific site hazard (i.e., a specific joint distribution of $Sa$, $SaRatio$, and $D_{S5-75}$).

# ### Import python modules

# In[35]:


import numpy as np
import random
import time
from math import pi
import pandas as pd
from ctypes import *
import matplotlib.pyplot as plt
import sys


# ### Import PLoM modules

# In[36]:


sys.path.insert(1, '../../')
from PLoM import *


# ### Load Incremental (IDA) Data
# MSA data are loaded via a comma-separate value (csv) file. The first row contains column names for both predictors (X) and responses (y). The following rows are input sample data. Users are expected to specif the csv filename.

# In[37]:


# Filename
filename = './data/response_frame12_ida_comb.csv'
model = PLoM(model_name='IDA', data=filename, col_header=True, plot_tag=True)


# ### Configuring tasks
# Please specify tasks to run - the list of tasks can be run in sqeunce or invidivdual tasks can be run separately.

# In[38]:


tasks = ['DataNormalization','RunPCA','RunKDE','ISDEGeneration']


# ### Step 0: Scaling the data

# In[39]:


# Configure the task
model.ConfigTasks(['DataNormalization'])
# Launch the run
model.RunAlgorithm()


# ### Step 1: Principal Component Analysis (PCA)

# In[40]:


# Tolerance for truncating principal components
tol_pca = 1e-6
# Configure the task
model.ConfigTasks(['RunPCA'])
# Launch the run
model.RunAlgorithm(epsilon_pca=tol_pca)


# ### Step 2: Kernel Density Estimation (KDE)

# In[41]:


# Smoothing parameter in the KDE
sp = 25
# Configure the task
model.ConfigTasks(['RunKDE'])
# Launch the run
model.RunAlgorithm(epsilon_kde=sp)


# ### Step 3: Create the generator

# In[42]:


# Extra parameters for ISDE generation
new_sample_num_ratio = 20
tol_PCA2 = 1e-5
# Configure the task
model.ConfigTasks(['ISDEGeneration'])
# Launch the run
model.RunAlgorithm(n_mc = new_sample_num_ratio, tol_PCA2 = tol_PCA2)


# ### Step 4: Exporting data

# In[43]:


# Available data list
model.export_results()


# In[44]:


# Pick up the original and new realizations, X0 and X_new
model.export_results(data_list=['/X0','/X_new'], file_format_list=['csv','csv'])


# ### Post-processing
# We would like to check the basic statistics of the input sample (i.e., IDA) and the generated new realizations by PLoM. The key metrics include the median, standard deviation, and correlation coefficient matrix of different structural responses.

# In[45]:


# Load results
df_ida = pd.read_csv('../../RunDir/IDA/DataOut/X0.csv')
df_plom = pd.read_csv('../../RunDir/IDA/DataOut/X_new.csv')
print(df_ida.head)
print(df_plom.head)


# In[46]:


x0 = df_ida.iloc[:,1:].T
x_c = df_plom.iloc[:,1:].T
x_name = x0.index.tolist()
x0 = np.array(x0)
x_c = np.array(x_c)
n = 27
# Correlation coefficient matrix
c_ida = np.corrcoef(x0)
c_plom = np.corrcoef(x_c)
c_combine = c_ida
tmp = np.triu(c_plom).flatten()
tmp = tmp[tmp != 0]
c_combine[np.triu_indices(27)] = tmp

# Plot covariance matrix
fig, ax = plt.subplots(figsize=(8,6))
ctp = ax.contourf(c_combine[3:,3:], cmap=plt.cm.hot, levels=1000)
ctp.set_clim(0,1)
ax.plot([0, 23], [0, 23], 'k--')
ax.set_xticks(list(range(n-3)))
ax.set_yticks(list(range(n-3)))
ax.set_xticklabels(x_name[3:], fontsize=8, rotation=45)
ax.set_yticklabels(x_name[3:], fontsize=8, rotation=45)
ax.set_title('Covariance matrix comparison')
ax.grid()
cbar = fig.colorbar(ctp,ticks=[x/10 for x in range(11)])
plt.show()

# Plot the cross-section of correlation matrix
fig, ax = plt.subplots(figsize=(6,4))
ax.plot([0],[0],'k-',label='MSA')
ax.plot([0],[0],'r:',label='PLoM')
for i in range(n-3):
    ax.plot(np.array(range(n-3)),c_ida[i+3][3:],'k-')
    ax.plot(np.array(range(n-3)),c_plom[i+3][3:],'r:')
ax.set_xticks(list(range(n-3)))
ax.set_xticklabels(x_name[3:], fontsize=8, rotation=45)
ax.set_ylabel('Correlation coefficient')
ax.set_ylim([0,1])
ax.set_xlim([0,n-4])
ax.legend()
ax.grid()
plt.show()


# ### Hazard Adjustment
# This section can be used to process the PLoM predictions from raw IDA training. Site specific hazard information is needed as an input. An example site hazard csv file is provided, the first column is the Sa intensity, the second column is the median SaRatio, the third column is the median duration, and the last four columns are covariance matrix entries.

# In[47]:


# Load site hazard information
shz = pd.read_csv('./data/site_hazard.csv')
sa_levels = shz['Sa']
print(shz)
print(np.array(shz.iloc[0]['cov11':]).reshape((2,2)))


# In[48]:


# Draw samples from the site distribution
num_rlz = 1000 # sample size
np.random.seed(1) # random seed for replicating results
rlz_imv = []
for i in range(len(shz.index)):
    rlz_imv.append(np.random.multivariate_normal(mean=[shz['mSaRatio'][i],shz['mDs'][i]],cov=np.array(shz.iloc[i]['cov11':]).reshape((2,2)),size=num_rlz))


# In[49]:


# Search nearest PLoM data points for each sample in rlz_imv
lnsa_plom = x_c[0]
lnsaratio_plom = x_c[1]
lnds_plom = x_c[2]

# Create the nearest interporator and interpolate data
from scipy.interpolate import NearestNDInterpolator
res_edp = []
for i in range(n-3):
    # Loop all EDPs
    interp_nn = NearestNDInterpolator(list(zip(lnsa_plom,lnsaratio_plom,lnds_plom)),x_c[3+i])
    pred_nn = []
    for j in range(len(shz.index)):
        # Loop all intensity levels
        pred_nn.append(interp_nn(np.ones(rlz_imv[j][:,0].shape)*np.log(shz['Sa'][j]),
                                 rlz_imv[j][:,0],rlz_imv[j][:,1]))
    res_edp.append(pred_nn)
        
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(rlz_imv[0][:,0],rlz_imv[0][:,1],'r.',label='Resample')
plt.show()


# In[50]:


ref_msa = pd.read_csv('./data/response_rcf12_msa_la_nc.csv')


# In[51]:


from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=2,weights='distance',algorithm='auto',p=2)
res = []
for i in range(n-3):
    # Loop all EDPs
    neigh.fit(np.transpose(x_c[0:3]),x_c[i+3])
    pred = []
    for j in range(len(shz.index)):
        # Loop all intensity levels
        pred.append(neigh.predict(np.array((np.ones(rlz_imv[j][:,0].shape)*np.log(shz['Sa'][j]),rlz_imv[j][:,0],rlz_imv[j][:,1])).T))
    res.append(pred)


# In[52]:


num_story = 12
num_sa = 6
sdr_cur_med_msa = np.zeros((num_story,num_sa))
sdr_cur_std_msa = np.zeros((num_story,num_sa))
sdr_cur_med_plom = np.zeros((num_story,num_sa))
sdr_cur_std_plom = np.zeros((num_story,num_sa))
for i in range(12):
    for j in range(6):
        sdr_cur_msa = ref_msa.loc[ref_msa['Sa']==shz['Sa'][j]][x_name[i+3][2:]]
        sdr_cur_med_msa[i,j] = np.exp(np.mean(np.log(sdr_cur_msa)))
        sdr_cur_std_msa[i,j] = np.std(np.log(sdr_cur_msa))
        sdr_cur_plom = np.exp(res[i][j])
        sdr_cur_med_plom[i,j] = np.exp(np.mean(res[i][j]))
        sdr_cur_std_plom[i,j] = np.std(res[i][j])
        
fig = plt.figure(figsize=(12,8))
story_list = list(range(1,num_story+1))
for i in range(6):
    plt.subplot(2,3,i+1)
    ax = plt.gca()
    ax.plot([0],[0],'k-',label='MSA')
    ax.plot([0],[0],'r-',label='PLoM-IDA \nHazard Adjusted')
    ax.plot(sdr_cur_med_msa[:,i],story_list,'k-')
    ax.plot(sdr_cur_med_msa[:,i]*np.exp(sdr_cur_std_msa[:,i]),story_list,'k--')
    ax.plot(sdr_cur_med_msa[:,i]/np.exp(sdr_cur_std_msa[:,i]),story_list,'k--')
    ax.plot(sdr_cur_med_plom[:,i],story_list,'r-')
    ax.plot(sdr_cur_med_plom[:,i]*np.exp(sdr_cur_std_plom[:,i]),story_list,'r--')
    ax.plot(sdr_cur_med_plom[:,i]/np.exp(sdr_cur_std_plom[:,i]),story_list,'r--')
    ax.set_xlim(0.0,0.05)
    ax.set_ylim(1,12)
    ax.grid()
    ax.legend()
    ax.set_xlabel('$SDR_{max}$ (in/in)')
    ax.set_ylabel('Story')


# In[53]:


num_story = 12
num_sa = 6
pfa_cur_med_msa = np.zeros((num_story,num_sa))
pfa_cur_std_msa = np.zeros((num_story,num_sa))
pfa_cur_med_plom = np.zeros((num_story,num_sa))
pfa_cur_std_plom = np.zeros((num_story,num_sa))
for i in range(12):
    for j in range(6):
        pfa_cur_msa = ref_msa.loc[ref_msa['Sa']==shz['Sa'][j]][x_name[i+15][2:]]
        pfa_cur_med_msa[i,j] = np.exp(np.mean(np.log(pfa_cur_msa)))
        pfa_cur_std_msa[i,j] = np.std(np.log(pfa_cur_msa))
        pfa_cur_plom = np.exp(res[i+12][j])
        pfa_cur_med_plom[i,j] = np.exp(np.mean(res[i+12][j]))
        pfa_cur_std_plom[i,j] = np.std(res[i+12][j])
        
fig = plt.figure(figsize=(12,8))
story_list = list(range(1,num_story+1))
for i in range(6):
    plt.subplot(2,3,i+1)
    ax = plt.gca()
    ax.plot([0],[0],'k-',label='MSA')
    ax.plot([0],[0],'r-',label='PLoM-IDA \nHazard Adjusted')
    ax.plot(pfa_cur_med_msa[:,i],story_list,'k-')
    ax.plot(pfa_cur_med_msa[:,i]*np.exp(pfa_cur_std_msa[:,i]),story_list,'k--')
    ax.plot(pfa_cur_med_msa[:,i]/np.exp(pfa_cur_std_msa[:,i]),story_list,'k--')
    ax.plot(pfa_cur_med_plom[:,i],story_list,'r-')
    ax.plot(pfa_cur_med_plom[:,i]*np.exp(pfa_cur_std_plom[:,i]),story_list,'r--')
    ax.plot(pfa_cur_med_plom[:,i]/np.exp(pfa_cur_std_plom[:,i]),story_list,'r--')
    ax.set_xlim(0.0,1)
    ax.set_ylim(1,12)
    ax.grid()
    ax.legend()
    ax.set_xlabel('$PFA$ (g)')
    ax.set_ylabel('Story')


# In[54]:


x0_ref = []
for i in range(n):
    x0_ref.append([np.log(x) for x in ref_msa.iloc[:, i].values.tolist()])

c_msa = np.corrcoef(x0_ref)
res_conct = []
for i in range(n-3):
    tmp = []
    for j in range(len(shz.index)):
        tmp = tmp+res[i][j].tolist()
    res_conct.append(tmp)
c_plom = np.corrcoef(res_conct)
# Plot correlation of resampled data
fig, ax = plt.subplots(figsize=(6,4))
ax.plot([0],[0],'k-',label='MSA')
ax.plot([0],[0],'r:',label='PLoM-IDA (Hazard Adjusted)')
for i in range(n-15):
    ax.plot(np.array(range(n-3)),c_msa[i+3][3:],'k-')
    ax.plot(np.array(range(n-3)),c_plom[i],'r:')
ax.set_xticks(list(range(n-3)))
ax.set_xticklabels(x_name[3:], fontsize=8, rotation=45)
ax.set_ylabel('Correlation coefficient')
ax.set_ylim([0,1])
ax.set_xlim([0,n-16])
ax.legend()
ax.grid()
plt.show()
fig.savefig('plom_vs_ida_cov.png',dpi=600)


# In[55]:


# Estimation errors
err_med = np.linalg.norm(np.log(sdr_cur_med_plom) - np.log(sdr_cur_med_msa),axis=0)/np.linalg.norm(np.log(sdr_cur_med_msa),axis=0)
err_std = np.linalg.norm(sdr_cur_std_plom - sdr_cur_std_msa,axis=0)/np.linalg.norm(sdr_cur_std_msa,axis=0)
# Plot
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(list(range(6)),err_med,'ko-',label='Mean EDP')
ax.plot(list(range(6)),err_std,'rs-',label='Standard deviation EDP')
ax.set_xticks(list(range(6)))
ax.set_xticklabels(['Sa = '+str(x)+'g' for x in sa_levels],rotation=30)
ax.set_xlim([0,5])
ax.set_ylim([0,1])
ax.set_ylabel('MSE')
ax.grid()
ax.legend()
plt.show()


# In[ ]:




