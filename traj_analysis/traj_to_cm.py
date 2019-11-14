import os
import sys 
import glob 
import h5py 
import numpy as np 
from tqdm import tqdm 

import MDAnalysis as mda 
from MDAnalysis.analysis import distances 
from utils import triu_to_full

print "Calculating contact maps from MD trajectories" 

pdb_file = '../2F4K-0-protein/2F4K.pdb' 
dcd_files = sorted(glob.glob('../2F4K-0-protein/2F4K-0-protein*.dcd'))

contact_maps = []
mda_traj = mda.Universe(pdb_file, dcd_files)  
protein_ca = mda_traj.select_atoms('protein and name CA') 
		
for _ in tqdm(mda_traj.trajectory): 
	contact_map = triu_to_full(
		(distances.self_distance_array(protein_ca.positions) < 8.0) * 1.0
		)
	contact_maps.append(contact_map) 

contact_maps = np.array(contact_maps)

# padding if odd dimension occurs in image
pad_f = lambda x: (0,0) if x%2 == 0 else (0,1)
padding_buffer = [(0,0)]
for x in contact_maps.shape[1:]:
    padding_buffer.append(pad_f(x))
contact_maps = np.pad(contact_maps, padding_buffer, mode='constant')

contact_maps = contact_maps.reshape((contact_maps.shape) + (1,))

cm_h5 = h5py.File('contact_maps_2F4K-0.h5', 'w') 
cm_h5.create_dataset('contact_maps', data=contact_maps) 
cm_h5.close() 

print 'Done'
# print dcd_files 
