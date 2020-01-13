import os, sys
import h5py, warnings 
import argparse
import numpy as np 
from glob import glob

sys.path.append('../py_modules/')
from utils import cm_to_cvae, read_h5py_file

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--sim_path", dest='f', help="Input: OpenMM simulation path") 
parser.add_argument("-o", "--output", dest='o', help="Output: CVAE 2D contact map h5 input file")
parser.add_argument("-n", "--num_frame", dest='n', default="all", help="Option: Number of frame to use for h5") 

# Let's say I have a list of h5 file names 
args = parser.parse_args() 

if args.f: 
    cm_filepath = os.path.abspath(os.path.join(args.f, 'omm*/*_cm.h5')) 
else: 
    warnings.warn("No input dirname given, using current directory...") 
    cm_filepath = os.path.abspath(os.path.join('.', 'omm*/*_cm.h5'))

cm_files = sorted(glob(cm_filepath))[:60] 
if cm_files == []: 
    raise IOError("No h5 file found, recheck your input filepath") 
# Get a list of opened h5 files 
cm_data_lists = [read_h5py_file(cm_file) for cm_file in cm_files] 

# Compress all .h5 files into one in cvae format 
cvae_input = cm_to_cvae(cm_data_lists)


# Create .h5 as cvae input
cvae_input_file = args.o
cvae_input_save = h5py.File(cvae_input_file, 'w')
if args.n == "all": 
    cvae_input_save.create_dataset('contact_maps', data=cvae_input)
else: 
    cvae_input_save.create_dataset('contact_maps', data=cvae_input[:int(args.n)])
cvae_input_save.close()
