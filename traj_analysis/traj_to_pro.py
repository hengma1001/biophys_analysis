import os 
import glob 
import numpy as np 
import MDAnalysis as mda 

from tqdm import tqdm 
from MDAnalysis.analysis import align

runs = sorted(glob.glob('../../md_runs/run_*'))

ref_pdb = '../../md_runs/run_protein/prot.pdb'
ref = mda.Universe(ref_pdb) 
ref_pro = ref.select_atoms("protein")

save_path = 'protein_DCDs' 
os.makedirs(save_path, exist_ok=True) 

for run in runs: 
    print(run) 
    run_label = os.path.basename(run).replace("run_", "")

    if run_label == 'protein': 
        pdb_file = run + '/prot.pdb'
    else: 
        pdb_file = run + '/comp.pdb' 
    traj_file = run + '/output.dcd' 
    
    mda_traj = mda.Universe(pdb_file, traj_file) 
    protein = mda_traj.select_atoms("protein")

    if run_label == "pep": 
        n_oxt = np.where(protein.atoms.names == 'OXT')[0]
        protein = protein.atoms[:n_oxt[0]+1]

    save_pdb = save_path + f"/{run_label}.pdb" 
    save_dcd = save_path + f"/{run_label}.dcd" 
    protein.write(save_pdb)
    with mda.Writer(save_dcd, protein.n_atoms) as W: 
        for _ in tqdm(mda_traj.trajectory): 
            align.alignto(protein, ref_pro, 
                    select="protein and name CA", 
                    weights="mass") 
            W.write(protein.atoms) 
