import os
import sys 
import glob 
import h5py 
import json 
import numpy as np 
from tqdm import tqdm 

import MDAnalysis as mda 
from MDAnalysis.analysis import distances 

print "Calculating contact maps from MD trajectories" 

pdb_files = sorted(glob.glob('../../*/*_input.pdb'))
json_files = sorted(glob.glob('../../*/*_input.json')) 
dcd_files = sorted(glob.glob('../../*/run_*/output.dcd'))

print(pdb_files, json_files, dcd_files) 

contact_maps = []
for i, dcd_file in enumerate(dcd_files): 
    # get the right pdb and json 
    pdb_file = pdb_files[i//2] 
    json_file = json_files[i//2]
    xy = json.load(open(json_file, 'r'))
    x = xy['x'] 
    y = xy['y'] 
    mda_traj = mda.Universe(pdb_file, dcd_file)  
    protein_ca = mda_traj.select_atoms('protein and name CA') 
                    
    for _ in tqdm(mda_traj.trajectory): 
        contact_map = (distances.distance_array(protein_ca[x].positions, protein_ca[y].positions) < 8.0) * 1.0
        contact_maps.append(contact_map) 

contact_maps = np.array(contact_maps)

# padding if odd dimension occurs in image
padding = 4 
pad_f = lambda x: (0,0) if x%padding == 0 else (0,padding-x%padding)
padding_buffer = [(0,0)]
for x in contact_maps.shape[1:]:
    padding_buffer.append(pad_f(x))
contact_maps = np.pad(contact_maps, padding_buffer, mode='constant')

contact_maps = contact_maps.reshape((contact_maps.shape) + (1,))

cm_h5 = h5py.File('contact_maps_mpro.h5', 'w') 
cm_h5.create_dataset('contact_maps', data=contact_maps) 
cm_h5.close() 

print 'Done'
# print dcd_files 
