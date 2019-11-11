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

pdb_file = '../1FME-0-protein/1FME-0.pdb' 
dcd_files = sorted(glob.glob('../1FME-0-protein/1FME-0-protein-*.dcd'))

contact_maps = []
mda_traj = mda.Universe(pdb_file, dcd_files)  
protein_ca = mda_traj.select_atoms('protein and name CA') 
		
for _ in tqdm(mda_traj.trajectory): 
	contact_map = triu_to_full(
		(distances.self_distance_array(protein_ca.positions) < 8.0) * 1.0
		)
	contact_maps.append(contact_map) 

contact_maps = np.array(contact_maps)
contact_maps = contact_maps.reshape((contact_maps.shape) + (1,))

cm_h5 = h5py.File('contact_maps_%s.h5' % os.path.basename(pdb_file)[:-4], 'w') 
cm_h5.create_dataset('contact_maps', data=contact_maps) 
cm_h5.close() 

print 'Done'
# print dcd_files 