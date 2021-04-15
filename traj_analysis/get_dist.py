import os
import sys 
import glob 
import h5py 
import numpy as np 
from tqdm import tqdm 

import MDAnalysis as mda 
from numpy import linalg as LA
sys.path.append("../py_modules") 
from utils import triu_to_full

def dist_pbc(a, b, box):
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
    print(dist_vec)

print("Calculating distance between complexes from MD trajectories") 

# pdb_file = '../../md_runs/cpep.gro' 
# dcd_files = sorted(glob.glob('../../md_runs/rex_0*/output.dcd'))
pdb_files = sorted(glob.glob('/lus/theta-fs0/projects/RL-fold/msalim/production-runs/pasc/nsp1016_448_gpu.1/md_runs/run*_*/*.pdb'))
dcd_files = sorted(glob.glob('/lus/theta-fs0/projects/RL-fold/msalim/production-runs/pasc/nsp1016_448_gpu.1/md_runs/run*_*/*.dcd'))

assert len(pdb_files) == len(dcd_files) 
dists = []
# labels = []
# label_kinds = set()
# mda_traj = mda.Universe(pdb_file, dcd_files)  
# protein_ca = mda_traj.select_atoms('protein and name CA') 
		
for pdb_file in tqdm(pdb_files): 
    traj_file = dcd_files[pdb_files.index(pdb_file)]
#     mda_traj = mda.Universe(pdb_file, dcd) 
    try: 
        mda_traj = mda.Universe(pdb_file, traj_file) 
    except OSError:
        print(pdb_file) 
        continue
    protein_ca = mda_traj.select_atoms('protein and name CA')
    nsp16 = mda_traj.segments[0].atoms
    nsp10 = mda_traj.segments[1].atoms
#     label = os.path.basename(os.path.dirname(pdb)).split('_')[2]
#     label_kinds.add(label) 

    for _ in mda_traj.trajectory[::5]: 
        dist = dist_pbc(
                nsp10.center_of_mass(), 
                nsp16.center_of_mass(), 
                nsp16.dimensions) 
        dists += [dist]

dists = np.array(dists)

np.save('dist.npy', dists)
print('Done')
# print dcd_files 
