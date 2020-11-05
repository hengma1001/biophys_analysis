import os 
import glob

import numpy as np
import pandas as pd
import MDAnalysis as mda
from tqdm import tqdm
from numpy import linalg as LA

runs = sorted(glob.glob("../../MD_exps/omm_runs_*"))

df = [] 
dist_list = [] 
for run in tqdm(runs): 
    pdb_file = run + '/comp.pdb'
    dcd_file = run + '/output.dcd' 

    run_base = os.path.basename(run) 
    run_label = run_base.split("_")[3]
    if int(run_label) < 14: 
        continue

    mda_traj = mda.Universe(pdb_file, dcd_file) 

    nsp16 = mda_traj.segments[0].atoms
    nsp10 = mda_traj.segments[1].atoms
    no_sol = mda_traj.select_atoms("protein") 

    for ts in mda_traj.trajectory: 
        box_edge = ts.dimensions[0]
        box_center = box_edge / 2
        trans_vec = box_center - np.array(nsp16.center_of_mass())
        no_sol.atoms.translate(trans_vec).wrap()
        trans_vec = box_center - np.array(no_sol.center_of_mass())
        no_sol.atoms.translate(trans_vec).wrap()
        dist_vec = no_sol.segments[0].atoms.center_of_mass() - no_sol.segments[1].atoms.center_of_mass()
        dist = LA.norm(dist_vec)
        
        dist_list += [dist]
        df.append({'sys_name': os.path.basename(run), 
                   'frame': ts.frame, 
                   'dist_vec': dist_vec, 
                   'dist': dist}) 

df = pd.DataFrame(df) 
df.to_pickle('nsp10_16_distance_skip14.pkl', protocol=2) 
np.save("dist_skip14.npy", dist_list) 
