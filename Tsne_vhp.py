#!/usr/bin/env python
import os 
import glob 
import h5py 
import numpy as np 

from utils import cm_to_cvae 


# In[8]:


omm_list = sorted(glob.glob('/gpfs/alpine/bip179/proj-shared/vhp_exp.3rd/omm_runs_*'))


# # Embeddings 


model_weight = '../CVAE_exps_6/cvae_weight.h5'

cm_data_lists = [] 
num_frame = 0 
for omm in omm_list: 
    cm_file = os.path.join(omm, 'output_cm.h5')
    cm_h5 = h5py.File(cm_file, 'r') 
#     print cm_h5[u'contact_maps']
    cm_data_lists.append(cm_h5[u'contact_maps'].value) 
    num_frame += cm_h5[u'contact_maps'].shape[1]
    cm_h5.close() 

print 'Embedding %d frames. ' % num_frame


cvae_input = cm_to_cvae(cm_data_lists)


print cvae_input.shape


from utils import predict_from_cvae
cm_predict = predict_from_cvae(model_weight, cvae_input, hyper_dim=6)


print 'Embedded dimension as ', cm_predict.shape


# # T-sne embedding 
from sklearn.manifold import TSNE  
cm_tsne = TSNE(n_components=2).fit_transform(cm_predict) 

# # Saving results 

h5_save = h5py.File('./tsne_vhp.h5', 'w') 


h5_save.create_dataset('cm_predict', data=cm_predict)  
h5_save.create_dataset('tsne', data=cm_tsne) 


h5_save.close() 


# In[37]:


h5_save = h5py.File('./latent3d_vhp.h5', 'r') 

print h5_save.items()



# In[ ]:




