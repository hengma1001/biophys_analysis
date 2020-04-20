import os 
import h5py 
import numpy as np 

import sys 
sys.path.append('../py_modules') 

from sklearn.cluster import MiniBatchKMeans 
from sklearn.cluster import KMeans 
from sklearn.externals import joblib 

from cvae.CVAE import CVAE 

model_weight = '../../CVAE_exps/cvae_runs_03_1585670249/cvae_weight.h5'
h5_file = '../traj_analysis/contact_maps.h5' 

cm_h5 = h5py.File(h5_file, 'r')
cvae_input = cm_h5['contact_maps']

def predict_from_cvae(model_weight, cvae_input, hyper_dim=3): 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(2)  
    cvae = CVAE(cvae_input.shape[1:], hyper_dim) 
    cvae.model.load_weights(model_weight)
    cm_predict = cvae.return_embeddings(cvae_input) 
    del cvae 
    return cm_predict


cm_predict = predict_from_cvae(model_weight, cvae_input, hyper_dim=3) 

# # KMeans clustering 
# kmeans = KMeans(n_clusters=1000, random_state=0).fit(cm_predict) 
# kmeans_labels = kmeans.labels_ 
# kmeans_file = 'kmeans.joblib' 
# joblib.dum(kmeans, kmeans_file) 
# 
# # MiniBatchKMeans clustering 
# mbkmeans = MiniBatchKMeans(n_clusters=1000, random_state=0, max_iter=50).fit(cm_predict) 
# mbkmeans_labels = mbkmeans.labels_ 
# mbkmeans_file = 'mbkmeans.joblib' 
# joblib.dum(mbkmeans, mbkmeans_file) 


# Save everything 
h5_result = h5py.File('latent_cvae.h5', 'w') 
h5_result.create_dataset('cvae', data=cm_predict) 
# h5_result.create_dataset('kmeans', data=kmeans_labels) 
# h5_result.create_dataset('mbkmeans', data=mbkmeans_labels) 
h5_result.close() 
