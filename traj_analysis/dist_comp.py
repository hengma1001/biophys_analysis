import os 
import glob

import numpy as np
import pandas as pd
import MDAnalysis as mda
from tqdm import tqdm
from numpy import linalg as LA

runs = sorted(glob.glob("../../MD_exps/omm_runs_*"))

df = [] 
for run in runs: 
    pdb_file = run + '/comp.pdb'
    dcd_file = run + '/output.dcd' 

    mda_traj = mda.Universe(pdb_file, dcd_file) 

    nsp16 = mda_traj.segments[0].atoms
    nsp10 = mda_traj.segments[1].atoms

    for ts in tqdm(mda_traj.trajectory): 
        dist_vec = nsp10.center_of_mass() - nsp16.center_of_mass()
        dist_vec = mda.lib.distances.apply_PBC(dist_vec, ts.dimensions)
        dist = LA.norm(dist_vec)
        
        df.append({'sys_name': os.path.basename(run), 
                   'frame': ts.frame, 
                   'dist_vec': dist_vec, 
                   'dist': dist}) 

df = pd.DataFrame(df) 
df.to_pickle('nsp10_16_distance.pkl', protocol=2) 
