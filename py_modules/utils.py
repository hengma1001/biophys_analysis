import os 
import numpy as np
import h5py 
import errno 
import MDAnalysis as mda 
# from cvae.CVAE import CVAE
# from keras import backend as K 
from sklearn.cluster import DBSCAN 

def triu_to_full(cm0):
    num_res = int(np.ceil((len(cm0) * 2) ** 0.5))
    iu1 = np.triu_indices(num_res, 1)

    cm_full = np.zeros((num_res, num_res))
    cm_full[iu1] = cm0
    cm_full.T[iu1] = cm0
    np.fill_diagonal(cm_full, 1)
    return cm_full


def read_h5py_file(h5_file): 
    cm_h5 = h5py.File(h5_file, 'r', libver='latest', swmr=True)
    if 'contact_maps' in cm_h5.keys(): 
        return cm_h5[u'contact_maps'] 
    elif 'contacts' in cm_h5.keys(): 
        return cm_h5['contacts'] 
    else: 
        return [] 


def cm_to_cvae(cm_data_lists): 
    """
    A function converting the 2d upper triangle information of contact maps 
    read from hdf5 file to full contact map and reshape to the format ready 
    for cvae
    """
    cm_all = np.hstack(cm_data_lists)

    # transfer upper triangle to full matrix 
    cm_data_full = np.array([triu_to_full(cm_data) for cm_data in cm_all.T]) 

    # padding if odd dimension occurs in image 
    pad_f = lambda x: (0,0) if x%2 == 0 else (0,1) 
    padding_buffer = [(0,0)] 
    for x in cm_data_full.shape[1:]: 
        padding_buffer.append(pad_f(x))
    cm_data_full = np.pad(cm_data_full, padding_buffer, mode='constant')

    # reshape matrix to 4d tensor 
    cvae_input = cm_data_full.reshape(cm_data_full.shape + (1,))   
    
    return cvae_input


def stamp_to_time(stamp): 
    import datetime
    return datetime.datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S') 
    

def find_frame(traj_dict, frame_number=0): 
    local_frame = frame_number
    for key in sorted(traj_dict.keys()): 
        if local_frame - int(traj_dict[key]) < 0: 
#             dir_name = os.path.dirname(key) 
            if os.path.isdir(key):                 
                traj_file = os.path.join(key, 'output.dcd') 
            else: 
                traj_file = key 
            return traj_file, local_frame
        else: 
            local_frame -= int(traj_dict[key])
    raise Exception('frame %d should not exceed the total number of frames, %d' % (frame_number, sum(np.array(traj_dict.values()).astype(int))))
    
    
def write_pdb_frame(traj_file, pdb_file, frame_number, output_pdb): 
    mda_traj = mda.Universe(pdb_file, traj_file)
    mda_traj.trajectory[frame_number] 
    PDB = mda.Writer(output_pdb)
    PDB.write(mda_traj.atoms)     
    return output_pdb

def make_dir_p(path_name): 
    try:
        os.mkdir(path_name)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

# def predict_from_cvae(model_weight, cvae_input, hyper_dim=3): 
#     os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"]=str(0)  
#     cvae = CVAE(cvae_input.shape[1:], hyper_dim) 
#     cvae.model.load_weights(model_weight)
#     cm_predict = cvae.return_embeddings(cvae_input) 
#     del cvae 
#     K.clear_session()
#     return cm_predict

def outliers_from_latent(cm_predict, eps=0.35): 
    db = DBSCAN(eps=eps, min_samples=10).fit(cm_predict)
    db_label = db.labels_
    outlier_list = np.where(db_label == -1)
    return outlier_list
