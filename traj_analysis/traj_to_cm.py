import os
import sys 
import glob 
import h5py 
import numpy as np 
from tqdm import tqdm 

import MDAnalysis as mda 
from MDAnalysis.analysis import distances 
sys.path.append("../py_modules") 
from utils import triu_to_full

print("Calculating contact maps from MD trajectories") 

# pdb_file = '../../md_runs/cpep.gro' 
# dcd_files = sorted(glob.glob('../../md_runs/rex_0*/output.dcd'))
pdb_files = sorted(glob.glob('../../Deepdrive_mds/omm_runs_*/comp.pdb'))
dcd_files = sorted(glob.glob('../../Deepdrive_mds/omm_runs_*/output.dcd'))

contact_maps = []
failed = [] 
# labels = []
# label_kinds = set()
# mda_traj = mda.Universe(pdb_file, dcd_files)  
# protein_ca = mda_traj.select_atoms('protein and name CA') 
		
for pdb_file, traj_file in tqdm(zip(pdb_files, dcd_files)): 
#     mda_traj = mda.Universe(pdb_file, dcd) 
    try: 
        mda_traj = mda.Universe(pdb_file, traj_file) 
    except OSError:
        failed += [pdb_file] 
        continue
    protein_ca = mda_traj.select_atoms('protein and name CA')
#     label = os.path.basename(os.path.dirname(pdb)).split('_')[2]
#     label_kinds.add(label) 

    for _ in mda_traj.trajectory[::10]: 
            contact_map = triu_to_full(
                    (distances.self_distance_array(protein_ca.positions) < 8.0) * 1
                    )
            contact_maps.append(contact_map) 
#             labels.append(len(label_kinds)-1) 

print("failed MD cases: ", failed) 
contact_maps = np.array(contact_maps)

# padding if odd dimension occurs in image
padding = 4 
pad_f = lambda x: (0,0) if x%padding == 0 else (0,padding-x%padding)
padding_buffer = [(0,0)]
for x in contact_maps.shape[1:]:
    padding_buffer.append(pad_f(x))
contact_maps = np.pad(contact_maps, padding_buffer, mode='constant')
print(contact_maps.shape) 

contact_maps = contact_maps.reshape((contact_maps.shape) + (1,))

cm_h5 = h5py.File('contact_maps.h5', 'w') 
cm_h5.create_dataset('contact_maps', data=contact_maps) 
# cm_h5.create_dataset('system', data=labels) 
cm_h5.close() 

print('Done')
# print dcd_files 
