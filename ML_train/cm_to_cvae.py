#!/usr/bin/env python  

import os 
import sys 
import h5py 
import glob 
import numpy as np 

sys.path.append('../py_modules') 
import utils 
from utils import read_h5py_file, cm_to_cvae 

# Get all the contact map h5 files 
cm_files = sorted(glob.glob('../../1_WT_bb_traj/contacts/contacts-*.h5'))

# Pick the 1st 100 for testing purpose 
cm_files = cm_files[:10] 

cm_data_lists = [read_h5py_file(cm_file) for cm_file in cm_files] 
print(cm_data_lists[0].shape) 
cvae_input = cm_to_cvae(cm_data_lists)
print(cvae_input.shape) 


# Create .h5 as cvae input
cvae_input_file = 'cvae_input.h5'
cvae_input_save = h5py.File(cvae_input_file, 'w')
cvae_input_save.create_dataset('contact_maps', data=cvae_input)
cvae_input_save.close()
