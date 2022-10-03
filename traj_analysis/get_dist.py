import os 
import glob
import numpy as np 
import pandas as pd 
import MDAnalysis as mda
from tqdm import tqdm
from numpy import linalg as LA

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
    

# a = np.array([1., 2.6]); b = np.array([3.6, 2])
# box = np.array([4., 4.]); c = np.array([10., 9.])
# dist_pbc(a, b, box)
# dist_pbc(a, c, box)

# exit()
runs = glob.glob("./nsp10_16_*") 

dist_collect = [] 
for run in runs: 
    pdbs = glob.glob(f"{run}/*pdb") 
    for pdb in tqdm(pdbs):
        sys_name = os.path.basename(pdb)[:-4]
        dcd = pdb.replace('pdb', 'dcd') 

        mda_traj = mda.Universe(pdb, dcd) 
        nsp16 = mda_traj.segments[0].atoms
        nsp10 = mda_traj.segments[1].atoms

        dist_list = []
        for ts in mda_traj.trajectory: 
            box = ts.dimensions 
            nsp10_c = nsp10.center_of_mass() 
            nsp16_c = nsp16.center_of_mass() 
            dist_list += [dist_pbc(nsp10_c, nsp16_c, box)] 
        dist_collect.append({"run": run, "sys_name": sys_name, 
                "dist": dist_list}) 

df = pd.DataFrame(dist_collect) 
df.to_pickle('nsp10_16_dist.pkl') 
        



