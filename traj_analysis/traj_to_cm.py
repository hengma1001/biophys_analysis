import os
import sys 
import glob 
import h5py 
import numpy as np 
from tqdm import tqdm 

import MDAnalysis as mda 
from MDAnalysis.analysis import distances 
from MDAnalysis.analysis.rms import RMSD
sys.path.append("../py_modules") 
from utils import triu_to_full

print("Calculating contact maps from MD trajectories") 

runs = glob.glob("../../md_runs/run_*")

h5_output = "h5_output" 
os.makedirs(h5_output, exist_ok=True) 

traj_out = 'traj_out'
os.makedirs(traj_out, exist_ok=True)

contact_maps = []
rmsd_profs = [] 
for run in runs: 
    label = os.path.basename(run)[4:]
    print(f"Running system {label}...") 
    pdb_file = run + f"/comp.pdb" 
    traj_file = run + "/output.dcd"
    mda_traj = mda.Universe(pdb_file, traj_file) 

    # prepare trajecory
    pdb_out = traj_out + f'/{label}_no_sol.pdb'
    dcd_out = traj_out + f'/{label}_no_sol.dcd'
    no_sol = mda_traj.select_atoms('not resname SOL and not resname NA and not resname CL')
    no_sol.write(pdb_out)
    if label.startswith("comp"):
        nsp10 = mda_traj.segments[1].atoms 
        nsp16 = mda_traj.segments[0].atoms 
        with mda.Writer(dcd_out, no_sol.n_atoms) as W: 
            for ts in tqdm(mda_traj.trajectory): 
                box_edge = ts.dimensions[0]
                box_center = box_edge / 2
                trans_vec = box_center - np.array(nsp16.center_of_mass())
                no_sol.atoms.translate(trans_vec).wrap()
                trans_vec = box_center - np.array(no_sol.center_of_mass())
                no_sol.atoms.translate(trans_vec).wrap()
                W.write(no_sol)
    else: 
        with mda.Writer(dcd_out, no_sol.n_atoms) as W:
            for ts in tqdm(mda_traj.trajectory):
                W.write(no_sol)
    mda_traj = mda.Universe(pdb_out, dcd_out)

    # prep ref traj 
    if label.startswith("comp"):
        ref_pdb = "../../md_runs/run_comp_000/comp.pdb"
    else:
        ref_pdb = pdb_file
    ref_traj = mda.Universe(ref_pdb) 

    rmsd_inst = RMSD(mda_traj, ref_traj, 
            select='protein and name CA',
            verbose=1) 
    rmsd_inst.run() 
    rmsd_profs.append(rmsd_inst.rmsd)
    
    protein_ca = mda_traj.select_atoms('protein and name CA')
    for _ in tqdm(mda_traj.trajectory): 
            contact_map = triu_to_full(
                    (distances.self_distance_array(protein_ca.positions) < 8.0) * 1
                    )
            contact_maps.append(contact_map) 
contact_maps = np.array(contact_maps)
rmsd_profs = np.array(rmsd_profs) 

# padding if odd dimension occurs in image
padding = 4 
pad_f = lambda x: (0,0) if x%padding == 0 else (0,padding-x%padding)
padding_buffer = [(0,0)]
for x in contact_maps.shape[1:]:
    padding_buffer.append(pad_f(x))
contact_maps = np.pad(contact_maps, padding_buffer, mode='constant')
print(contact_maps.shape) 

contact_maps = contact_maps.reshape((contact_maps.shape) + (1,))

h5_file = h5_output + f"/cm_{label}.h5"
cm_h5 = h5py.File(h5_file, 'w') 
cm_h5.create_dataset('contact_maps', data=contact_maps) 
cm_h5.create_dataset('rmsd', data=rmsd_profs) 
cm_h5.close() 

print('Done')
# print dcd_files 
