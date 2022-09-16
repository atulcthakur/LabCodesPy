#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This is to change the size of the cells in Jyupyter-notebook
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:99% }</style>"))


# In[2]:


import numpy as np
import pandas as pd
import scipy as sp
import numba
from tqdm.notebook import tqdm
from numba import jit, njit
from useful import read_aimd_pimd_xyz_faster, xyz_writer, make_whole, sort_traj, map_index
from numba.typed import List
from numba.types import float64, int64
import pickle
import os, shutil


# In[57]:


import matplotlib.pyplot as plt


# In[3]:


import P2Ethane  


# In[4]:


import ResidenceTimePC


# In[ ]:





# In[18]:


def P2_timeseries(traj, Id):
    #Carbons_from_zero = P2Ethane.substract_box(traj)
    carbon_vectors = P2Ethane.CC_vectors_Id(traj, Id=Id)
    carbon_timeseries = P2Ethane.make_timeseries(carbon_vectors)
    return carbon_timeseries


# In[ ]:





# In[19]:


def Theta_timeseries(traj):
    #Carbons_from_zero = ResidenceTimePC.substract_box(traj)
    trj, ls = ResidenceTimePC.select_first_layer(traj, molecule="ethane")
    theta_timeseries = ResidenceTimePC.make_theta(trj, ls)
    return theta_timeseries


# In[ ]:





# In[68]:


def Make_Traj_For_P2(xyz):
    out = {}
    ethane, natoms = read_aimd_pimd_xyz_faster(xyz)
    ethane_carbons = ResidenceTimePC.get_carbons(ethane, molecule="ethane")
    Carbons_from_zero = ResidenceTimePC.substract_box(ethane_carbons)
    trj, ls = ResidenceTimePC.select_first_layer(Carbons_from_zero, molecule="ethane")
    print(len(trj['time_0']))
    for counter, time in enumerate(tqdm(Carbons_from_zero)):
        frame = Carbons_from_zero[time].copy(deep=True)
        layer = frame[frame['mols'].isin(ls)]
        out[time] = layer
    return out, ls 


# In[ ]:





# In[69]:


traj_un, lsd = Make_Traj_For_P2("ethane.xyz")


# In[ ]:





# In[20]:


p2ts = P2_timeseries(traj_un, lsd)


# In[21]:


tts = Theta_timeseries(traj_un)


# In[ ]:





# In[49]:


def Weighted_timeseries(carbon_timeseries, theta_timeseries):
    out = {}
    for mol1, mol2 in zip(carbon_timeseries, theta_timeseries):
        carbs = carbon_timeseries[mol1]
        thetas = theta_timeseries[mol2]
        #print(len(carbs), len(thetas))
        weighted = carbs[['x', 'y', 'z']]*(np.array(thetas).reshape(-1, 1))
        out[mol1] = weighted
    return out


# In[53]:


pd.set_option("display.max_rows", None, "display.max_columns", None)


# In[50]:


w_ts = Weighted_timeseries(p2ts, tts)


# In[64]:


w_ts[2]


# In[72]:


weighted_p2 = P2Ethane.P2_Correlate(w_ts)


# In[73]:


plt.plot(weighted_p2)


# In[77]:


(weighted_p2*154)


# In[ ]:


cts = P2_timeseries("ethane.xyz")


# In[ ]:


theta_ts = Theta_timeseries("ethane.xyz")


# In[ ]:





# In[78]:


traj_un['time_1']


# In[ ]:





# In[ ]:




