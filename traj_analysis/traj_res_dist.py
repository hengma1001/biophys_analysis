import os 
import sys 
import glob 
import h5py

from tqdm import tqdm
import numpy as np 
import MDAnalysis as mda 
from MDAnalysis.analysis import distances 

traj_dirname = "../../MD_trajs/*_run*" 

traj_dirs = sorted(glob.glob(traj_dirname))

pdb_files = [glob.glob(traj_dir + '/*pdb')[0] for traj_dir in traj_dirs] 
traj_files = [glob.glob(traj_dir + '/*dcd')[0] for traj_dir in traj_dirs]

mda_traj = mda.Universe(pdb_files[0], traj_files[0]) 
n_atoms = mda_traj.atoms.n_atoms 

dist_prof = [] 
for pdb, traj in zip(pdb_files, traj_files): 
    mda_traj = mda.Universe(pdb, traj) 
    protein_ca = mda_traj.select_atoms('protein and name CA') 
    res_nums = np.array([145, 189, 451, 495]) - 1 
    # print protein_ca[res_nums]
    for _ in tqdm(mda_traj.trajectory): 
        dist = distances.distance_array(protein_ca.positions[res_nums], protein_ca.positions[res_nums]) 
        dist_prof.append([dist[0, 1], dist[2, 3]]) 

dist_prof = np.array(dist_prof) 
print dist_prof.shape 

dist_h5 = h5py.File('dist_sel.h5', 'w') 
dist_h5.create_dataset('dist', data=dist_prof) 
dist_h5.close() 

print "done" 
