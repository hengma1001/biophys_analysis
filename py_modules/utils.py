import os
import h5py 
import errno 
# from platform import python_implementation 
import numpy as np
from numpy import linalg as LA
import MDAnalysis as mda 
from sklearn.cluster import DBSCAN 

def dist_pbc(a, b, box=None):
    """
    calculate distance between two points
    in PBC box
    """
    assert len(a) == len(b)
    box = box[:len(a)]
    a = a % box
    b = b % box
    dist_vec = np.abs(a - b)
    dist_vec = np.abs(dist_vec - box * (dist_vec > box/2))
    return LA.norm(dist_vec)

def get_angle(pos1, pos2, pos3): 
    """get the angle of 1-2-3"""
    vec1 = pos1 - pos2
    vec2 = pos3 - pos2
    ang = np.arccos(sum(vec1*vec2) / (LA.norm(vec1) * LA.norm(vec2)))
    return ang * 180 / np.pi


def triu_to_full(cm0):
    num_res = int(np.ceil((len(cm0) * 2) ** 0.5))
    iu1 = np.triu_indices(num_res, 1)

    cm_full = np.zeros((num_res, num_res))
    cm_full[iu1] = cm0
    cm_full.T[iu1] = cm0
    np.fill_diagonal(cm_full, 1)
    return cm_full

def sparse_to_full(sparse_cm):
    full_cm = np.zeros(max(sparse_cm) + 1) 
    full_cm[sparse_cm] = 1 
    return full_cm

def read_h5py_file(h5_file): 
    cm_h5 = h5py.File(h5_file, 'r', libver='latest', swmr=True)
    return cm_h5[u'contact_maps'] 

def coord_polar_to_euc(r, theta, phi): 
    x = r * np.sin(phi) * np.cos(theta) 
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z
    
def coord_euc_to_polar(x, y, z): 
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arccos(z / r)
    theta = np.arcsin(y / (r * np.sin(phi)))
    return r, theta, phi

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
