#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 

import glob 
import h5py 

import numpy as np 


# In[2]:


import MDAnalysis as mda


# In[3]:


from MDAnalysis.analysis.rms import RMSD


# In[4]:


from utils import cm_to_cvae 


# In[8]:


omm_list = sorted(glob.glob('../vhp_exp.3rd/omm_runs_*'))


# # RMSD

# In[15]:


ref_pdb_file = '../vhp_exp.3rd/pdb/vhp1ww.pdb' 
start_point = '../vhp_exp.3rd/pdb/vhp1ww_solv.gro'


# In[17]:


RMSD_all = []
for omm in omm_list: 
    dcd_file = os.path.join(omm, 'output.dcd')
    mda_traj = mda.Universe(start_point, dcd_file)
    ref_traj = mda.Universe(ref_pdb_file)
    R = RMSD(mda_traj, ref_traj, select='protein and name CA')
    R.run() 
    RMSD_all.append(R.rmsd[:,2])


# In[19]:


RMSD_all = np.hstack(RMSD_all)


# # Embed

# In[22]:


model_weight = '../vhp_exp.3rd/CVAE_exps/cvae_weight.h5'


# In[23]:


cm_data_lists = [] 
num_frame = 0 
for omm in omm_list: 
    cm_file = os.path.join(omm, 'output_cm.h5')
    cm_h5 = h5py.File(cm_file, 'r') 
#     print cm_h5[u'contact_maps']
    cm_data_lists.append(cm_h5[u'contact_maps'].value) 
    num_frame += cm_h5[u'contact_maps'].shape[1]
    cm_h5.close() 


# In[24]:


num_frame


# In[25]:


cvae_input = cm_to_cvae(cm_data_lists)


# In[26]:


print cvae_input.shape


# In[27]:


from utils import predict_from_cvae
cm_predict = predict_from_cvae(model_weight, cvae_input, hyper_dim=3)


# In[28]:


cm_predict.shape


# # outliers 

# In[30]:


from utils import outliers_from_latent
eps = 0.2 

while True:
    outliers = np.squeeze(outliers_from_latent(cm_predict, eps=eps))
    n_outlier = len(outliers)
    print('dimension = {0}, eps = {1:.2f}, number of outlier found: {2}'.format(
        3, eps, n_outlier))
    if n_outlier > 200:
        eps = eps + 0.05
    else:
        outlier_list = outliers 
        break


# In[31]:


outlier_list


# In[32]:


h5_save = h5py.File('./latent3d_vhp.h5', 'w') 


# In[33]:


h5_save.create_dataset('cm_predict', data=cm_predict)  
h5_save.create_dataset('RMSD', data=RMSD_all) 
h5_save.create_dataset('outliers', data=outlier_list)  


# In[34]:


h5_save.close() 


# In[37]:


h5_save = h5py.File('./latent3d_vhp.h5', 'r') 


# In[38]:


h5_save.items()


# In[ ]:




