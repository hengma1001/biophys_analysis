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
    with h5py.File(h5_file, 'r', libver='latest', swmr=True) as cm_h5: 
        if 'contact_maps' in cm_h5.keys(): 
            return cm_h5[u'contact_maps'].value 
        elif 'contacts' in cm_h5.keys(): 
            return cm_h5['contacts'].value
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



def trim_cm(contact_map, resnum_intersted): 
    for i in np.arange(contact_map.shape[0]): 
        if i not in resnum_intersted: 
            contact_map[i] = 10 
            contact_map[:,i] = 10 
    contact_map = contact_map[contact_map != 10] 
    assert len(contact_map) == len(resnum_intersted) ** 2
    return contact_map.reshape(len(resnum_intersted), len(resnum_intersted))


def cm_to_cvae_trim(cm_data_lists, resnum_intersted): 
    """
    A function converting the 2d upper triangle information of contact maps 
    read from hdf5 file to full contact map of interested residues and 
    reshape to the format ready for cvae
    """
    cm_all = np.hstack(cm_data_lists)

    # transfer upper triangle to full matrix 
    cm_data_full = np.array([trim_cm(triu_to_full(cm_data), resnum_intersted) for cm_data in cm_all.T]) 

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


def read_h5py_len(h5_file):
    with h5py.File(h5_file, 'r', libver='latest', swmr=True) as cm_h5:
        if 'contact_maps' in cm_h5.keys():
            return cm_h5[u'contact_maps'].shape[1]
        elif 'contacts' in cm_h5.keys():
            return cm_h5['contacts'].shape[1]
        else:
            return 0


def cm_generator(cm_files, batch_size, resnum, shuffle=True): 
#     samples_per_epoch = len(cm_files) 
    number_of_files = len(cm_files)  
    counter = 0 
    if shuffle: 
        np.random.shuffle(cm_files) 
    cm_data_lists = []
    
    while 1: 
        # assuming batch size is way smaller than number of cm in each file 
        # therefore, it's sufficient to load only one extra while current is 
        # exhausted 
        if counter >= len(cm_files) and len(cm_data_lists) < batch_size: 
            # Reset counter to next epoch, abandon last batch of frames here 
            # cvae_input, cm_data_lists = cm_data_lists, [] 
            counter = 0 
        else: 
            if len(cm_data_lists) < batch_size: 
                # read a new h5 file when the leftover is smaller than batch size 
                new_cm = cm_to_cvae_trim([read_h5py_file(cm_files[counter])], resnum) 
#                 print(new_cm.shape) 
                counter += 1 
                if cm_data_lists == []: 
                    cm_data_lists = new_cm
                else: 
                    cm_data_lists = np.array(np.vstack([cm_data_lists, new_cm])) 
            cvae_input, cm_data_lists = cm_data_lists[:batch_size], cm_data_lists[batch_size:]
        
        yield cvae_input, cvae_input 
