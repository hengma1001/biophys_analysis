# -*- coding: future_fstrings -*-
import os
import glob
import tempfile
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import MDAnalysis as mda
from tqdm import tqdm
from numpy import linalg as LA


runs = sorted(glob.glob("../../traj_save/*/*.pdb"))

shiftx_exe = '/homes/heng.ma/miniconda3/envs/py2/bin/shiftx2.py'
shiftx_ph = 6.5
shiftx_temp = 303

result_save = 'chem_shift_save'
if not os.path.exists(result_save):
    os.makedirs(result_save)

df = []

for run in runs[:1]:
    pdb_file = run
    dcd_file = run.replace("pdb", "dcd")

    run_base = os.path.basename(run)[:-4]
    print(run_base)
    run_label = run_base.replace("run_", "")
    local_save = result_save + f'/{run_label}'
    if not os.path.exists(local_save):
        os.makedirs(local_save)

    mda_traj = mda.Universe(pdb_file, dcd_file)
    protein = mda_traj.select_atoms("protein")
    nsp16 = protein.segments[0].atoms
    nsp10 = protein.segments[1].atoms
    
    for ts in tqdm(mda_traj.trajectory[:300:100]):
        nsp16_save = f"{local_save}/nsp16_{ts.frame:06}.pdb"
        nsp16.write(nsp16_save)
        env_command = "bash -c 'source activate /homes/heng.ma/miniconda3/envs/py2'"
        command = f"{env_command} && {shiftx_exe} -i {nsp16_save} -p {shiftx_ph} -t {shiftx_temp}"
        command_run = subprocess.Popen(command, 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                shell=True)
        command_run.wait()
        os.remove(nsp16_save)
