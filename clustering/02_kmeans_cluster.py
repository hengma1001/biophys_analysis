import os 
import h5py 
import logging
import pandas as pd 
import numpy as np
from sklearn.cluster import MiniBatchKMeans

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

h5_file = h5py.File('../cvae_comp/cvae_40/latent.h5', 'r') 

n_clusters = [30, 50, 100, 200, 250, 500, 1000, 1500, 3000, 5000] 
cm_predict = np.array(h5_file[u'latent'])
# cm_predict = np.load('./LA_Z_emb.npy')

cluster_labels = []
for n_cluster in n_clusters: 
    logger.info(f"Starting to cluster the embeddings into {n_cluster} clusters")
    kmeans = MiniBatchKMeans(
                n_clusters=n_cluster, init='k-means++', 
                random_state=0
                ).fit(cm_predict[::])
    local_rec = {'n_cluster': n_cluster}
    local_rec['labels'] = kmeans.labels_
    # local_rec['labels_all'] = kmeans.predict(cm_predict)
    local_rec['centers'] = kmeans.cluster_centers_
    # local_rec['center_indices'] = kmeans.medoid_indices_
    
    cluster_labels.append(local_rec) 

df = pd.DataFrame(cluster_labels) 
df.to_pickle('cluster_kmeans.pkl')
