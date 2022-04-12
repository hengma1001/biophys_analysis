import os
import sys
import glob
import logging
# import numpy as np

import pandas as pd
import MDAnalysis as mda

from tqdm import tqdm
from mpi4py import MPI
from itertools import chain
from MDAnalysis.analysis import align

sys.path.append('../py_modules/')
from grid_analysis import atoms_grids


for _ in logging.root.manager.loggerDict:
    logging.getLogger(_).setLevel(logging.CRITICAL)

debug = 1
logger_level = logging.DEBUG if debug else logging.INFO
logging.basicConfig(level=logger_level, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

grid_size = 0.25 # unit \AA
cutoff = 6 

ref_top_file = '../../traj_save/top_files/comp_sam.top'
ref_pdb_file = '../../traj_save/Nsp10_Nsp16/comp_sam_0.pdb'

runs = sorted(glob.glob("../../traj_save/Nsp10_Nsp16*/*.pdb"))

# get sam occupying grids 
if rank == 0: 
    logger.info("Processing crystal structure for baseline")
ref_mda_u = mda.Universe(ref_pdb_file)
sam_atoms = ref_mda_u.select_atoms('resname SAM')
sam_ags =  atoms_grids(sam_atoms, ref_top_file, grid_size)
logger.debug(f"{len(sam_ags)} grids from SAM molecule: {len(sam_ags)* grid_size**3} A^3")

# get the baseline
sam_neighs = ref_mda_u.select_atoms(f'around {cutoff} (resname SAM)')
sam_neighs = sam_neighs.select_atoms('segid A')
sam_neighs = sam_neighs.residues.atoms
pro_ags = atoms_grids(sam_neighs, ref_top_file, grid_size)
grids_overlap = sam_ags.occupied_grids.intersection(pro_ags.occupied_grids)
logger.debug(f"{len(grids_overlap)} grids overlapping between SAM and protein: {len(grids_overlap)* grid_size**3} A^3")

exit()

# create holding list
df = []
# get topology store path
top_path = os.path.dirname(ref_top_file)
for run in runs: 
    pdb_file = run
    dcd_file = run.replace("pdb", "dcd")

    # get labels and skip cases 
    run_label = os.path.basename(pdb_file)[:-4]
    label_split = run_label.split('_')
    if len(label_split) == 2 and label_split[0] != 'nsp10': 
        top_file = f'{top_path}/{label_split[0]}.top'
    elif 'sam' in run_label: 
        top_file = f'{top_path}/{run_label[:-2]}.top'
    else: 
        continue

    if rank == 0:
        logger.info(f'Running system {run_label}...')

    if not os.path.exists(top_file):
        logger.error(f"{run_label}: Missing topology file...")

    mda_u = mda.Universe(pdb_file, dcd_file)
    sam_neighs_local = mda_u.atoms[sam_neighs.indices]

    # nsp16 align str
    align_str = 'name CA'
    if run_label.startswith('comp'): 
        align_str = f'{align_str} and segid A'
    elif run_label.startswith('nsp'):
        align_str = f'{align_str} and (segid A or segid SYST)'
    
    # align the traj to the ref structure 
    aligner = align.AlignTraj(
                mda_u, ref_mda_u, 
                select=align_str, in_memory=True).run()

    for ts in tqdm(mda_u.trajectory[rank::size]): 
        pro_ags = atoms_grids(sam_neighs_local, top_file, grid_size)
        grids_overlap = sam_ags.occupied_grids.intersection(pro_ags.occupied_grids)
        df.append({
                'sys_name': run_label,
                'frame': ts.frame,
                'n_grids': len(grids_overlap)
                })
        logger.debug(f"At frame {ts.frame}, {len(grids_overlap)} grids overlapping between SAM and protein: {len(grids_overlap)* grid_size**3} A^3")

    if rank == 0: 
        logger.info(f"Finished {run_label}...")

logger.info(f"Finished run on thread {rank}...")
logger.debug(df)
master_df = comm.gather(df, root=0)
if rank == 0:
    master_df = list(chain(*master_df))
    logger.debug(master_df)
    master_df = pd.DataFrame(master_df)
    master_df.to_pickle('occlusion_nsp16.pkl')
    
