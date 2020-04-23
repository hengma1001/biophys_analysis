import os 
import sys 
import glob 

from tqdm import tqdm
import numpy as np 
import MDAnalysis as mda 

traj_dirname = "../../MD_trajs/*CoV_run*" 

traj_dirs = sorted(glob.glob(traj_dirname))

pdb_files = [glob.glob(traj_dir + '/*pdb')[0] for traj_dir in traj_dirs] 
traj_files = [glob.glob(traj_dir + '/*dcd')[0] for traj_dir in traj_dirs]

mda_traj = mda.Universe(pdb_files[0], traj_files[0]) 
n_atoms = mda_traj.atoms.n_atoms 

with mda.Writer('traj_full.dcd', n_atoms) as W: 
    for pdb, traj in zip(pdb_files, traj_files): 
        mda_traj = mda.Universe(pdb, traj) 
        for _ in tqdm(mda_traj.trajectory): 
            W.write(mda_traj.atoms) 

print "done" 
