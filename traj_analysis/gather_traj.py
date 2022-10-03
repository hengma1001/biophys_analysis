import os 
import glob 

import numpy as np 
import MDAnalysis as mda 
from tqdm import tqdm
# from MDAnalysis.analysis import align
from MDAnalysis import transformations

# runs = [\
#         "/lambda_stor/homes/heng.ma/Research/covid19/Nsp10_Nsp16/nsp10_16_dist_sam/md_runs"
# ]

def wrap_atmgrp(atoms): 
    atom_CoM = atoms.center_of_mass()
    target = atom_CoM % atoms.dimensions[:3]
    trans_vec = target - atom_CoM
    atoms.translate(trans_vec) 


runs = glob.glob("/lambda_stor/homes/heng.ma/Research/insulins/conoINS-P1*/md_runs") 
#         /lambda_stor/homes/heng.ma/Research/covid19/Nsp10_Nsp16/nsp10_16_dist_s*/md_runs")
print(runs)

for run in runs: 
    if os.path.basename(run) == 'md_runs': 
        sims = glob.glob(run + '/run_*hexa*')
        save_path = run.split('/')[-2]
    else: 
        sims = glob.glob(run + '/omm_runs_*') 
        save_path = run.split('/')[-3]
    sims = sorted(sims) 

    print(f"processing {save_path}...") 
    os.makedirs(save_path, exist_ok=True) 
    
    n_sim = len(sims)
    for sim in tqdm(sims[:]): 
        sim_label = os.path.basename(sim).replace("run_", "").replace("omm_runs_", "")
        pdb_file = glob.glob(sim + '/comp*.pdb')[0]
        dcd_file = sim + '/output.dcd' 

        save_pdb = save_path + f"/{sim_label}.pdb" 
        save_dcd = save_path + f"/{sim_label}.dcd" 

        mda_traj = mda.Universe(pdb_file, dcd_file) 
        protein = mda_traj.select_atoms('protein or resname GLM') 

        box_edge = protein.dimensions[:3]
        box_center = box_edge / 2
        for i, seg in enumerate(protein.segments): 
            chain = seg.atoms
            trans_vec = box_center - np.array(chain.center_of_mass())
            protein.atoms.translate(trans_vec)
            for seg in protein.segments[i+1:]: 
                wrap_atmgrp(seg.atoms) 

        trans_vec = box_center - np.array(protein.center_of_mass())
        protein.atoms.translate(trans_vec)
        for seg in protein.segments[:]: 
            wrap_atmgrp(seg.atoms) 
        protein.write(save_pdb) 

        with mda.Writer(save_dcd, protein.n_atoms) as w:
            for ts in tqdm(mda_traj.trajectory[::]): 
                box_edge = ts.dimensions[:3]
                box_center = box_edge / 2
                for i, seg in enumerate(protein.segments): 
                    chain = seg.atoms
                    trans_vec = box_center - np.array(chain.center_of_mass())
                    protein.atoms.translate(trans_vec)
                    # protein.segments[i:].atoms.wrap()
                    for seg in protein.segments[i+1:]: 
                        wrap_atmgrp(seg.atoms) 

                trans_vec = box_center - np.array(protein.center_of_mass())
                protein.atoms.translate(trans_vec)
                for seg in protein.segments[:]: 
                    wrap_atmgrp(seg.atoms) 
                w.write(protein)


    # print(save_path, len(sims) )

