#!/usr/bin/env python
import os 
import sys
import glob 
import h5py 
import numpy as np 



cm_h5 = h5py.File("../cvae/cvae_15/latent.h5", 'r') 
cm_predict = cm_h5['latent'] 
print(('Embedded dimension as ', cm_predict.shape))

# # T-sne embedding 
from sklearn.manifold import TSNE  
cm_tsne = TSNE(n_components=3).fit_transform(cm_predict[::10]) 

# # Saving results 

h5_save = h5py.File('./tsne.h5', 'w') 
h5_save.create_dataset('cm_predict', data=cm_predict)  
h5_save.create_dataset('tsne', data=cm_tsne) 


h5_save.close() 


# In[37]:


h5_save = h5py.File('./tsne_fsp.h5', 'r') 

print(list(h5_save.items()))



# In[ ]:




