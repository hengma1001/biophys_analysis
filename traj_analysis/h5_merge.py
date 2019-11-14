import h5py 
import glob 
import numpy as np 

h5_files = glob.glob('./contact_maps*') 

contact_maps = []
for h5_filepath in h5_files: 
	h5_file = h5py.File(h5_filepath, 'r') 
	contact_maps.append(h5_file['contact_maps'].value) 
	h5_file.close() 


contact_maps = np.concatenate(contact_maps, axis=0) 
cm_h5 = h5py.File('contact_maps_all.h5', 'w') 
cm_h5.create_dataset('contact_maps', data=contact_maps) 
cm_h5.close() 
