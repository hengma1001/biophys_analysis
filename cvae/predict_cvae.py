import os 
import time
import h5py 
import numpy as np
# import sys, errno
import argparse 
import json 
from cvae.CVAE import CVAE, train_cvae, predict_from_cvae, h5py_save 

cvae_input = '/homes/heng.ma/Research/FoldingTraj/1FME-0/1FME/contact_maps_1FME-0.h5'
# '../traj_analysis/contact_maps.h5'
cvae_weight = '/lambda_stor/homes/heng.ma/Research/bba/theta_runs/BBA_102/best.h5'
hyper_dim = 10 
input_shape = (28, 28, 1)

if __name__ == '__main__':
    # cvae_info_dict = {} 
    # cm_h5 = h5py.File(cvae_input, 'r', libver='latest', swmr=True)
    # cm_data_input = cm_h5[u'contact_maps'].value

    # Prediction using trained CVAE 
    start_predict = time.time() 
    # input_shape = cm_data_input.shape
    cvae = CVAE(input_shape, hyper_dim)
    cm_embedded, predict_time = predict_from_cvae(
        cvae, cvae_input)
    end_predict = time.time() 

    # Collect information for performance assessment of CVAE 
    np.save('bba_emb_anton.npy', cm_embedded)
    # print('Predicting time is ', predict_time)
