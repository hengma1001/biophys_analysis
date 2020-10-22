#!/usr/bin/env python
import os
import glob
import h5py
import numpy as np

h5_file = 'latent_cvae-14.h5'
cm_h5 = h5py.File(h5_file, 'r')
cm_predict = cm_h5['cvae']
print(cm_predict.shape)
print(np.max(cm_predict.value))

# # T-sne embedding 
from sklearn.manifold import TSNE
cm_tsne = TSNE(n_components=3).fit_transform(cm_predict[::10])

h5_save = h5py.File('./tsne.h5', 'w')

h5_save.create_dataset('cm_predict', data=cm_predict)
h5_save.create_dataset('tsne', data=cm_tsne)
h5_save.close()

h5_save = h5py.File('./tsne.h5', 'r')
print h5_save.items()

