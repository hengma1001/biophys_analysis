import os
import sys 
import glob 
import h5py 
import json 
import numpy as np 
from tqdm import tqdm 

import MDAnalysis as mda 
from MDAnalysis.analysis import distances 
sys.path.append("../py_modules") 
from utils import triu_to_full

print("Calculating contact maps from MD trajectories") 

pdb_files = sorted(glob.glob('../../Deepdrive_mds/omm_runs_*/comp.pdb'))
dcd_files = sorted(glob.glob('../../Deepdrive_mds/omm_runs_*/output.dcd'))
np_file = './mpro_res.npy' 

contact_maps = []
failed = []
# mda_traj = mda.Universe(pdb_file, dcd_files)  
# protein_ca = mda_traj.select_atoms('protein and name CA') 
		
for pdb, dcd in tqdm(zip(pdb_files, dcd_files)): 
    try: 
        mda_traj = mda.Universe(pdb, dcd) 
    except OSError: 
        failed += [pdb] 
        continue 
    protein_ca = mda_traj.select_atoms('protein and name CA')
    resl_1 = np.load(np_file) 

    for _ in mda_traj.trajectory: 
            contact_map = (distances.distance_array(protein_ca.positions[resl_1], protein_ca.positions) < 8.0) * 1.0
            contact_maps.append(contact_map) 

print("failed MD on ", failed) 
contact_maps = np.array(contact_maps)

# padding if odd dimension occurs in image
padding = 4 
pad_f = lambda x: (0,0) if x%padding == 0 else (0,padding-x%padding)
padding_buffer = [(0,0)]
for x in contact_maps.shape[1:]:
    padding_buffer.append(pad_f(x))
contact_maps = np.pad(contact_maps, padding_buffer, mode='constant')

contact_maps = contact_maps.reshape((contact_maps.shape) + (1,))

cm_h5 = h5py.File('contact_maps.h5', 'w') 
cm_h5.create_dataset('contact_maps', data=contact_maps) 
cm_h5.close() 

print('Done')
# print dcd_files 
