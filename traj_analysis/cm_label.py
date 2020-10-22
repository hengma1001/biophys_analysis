# coding: utf-8
import h5py 
cm_h5 = h5py.File('./contact_maps.h5', 'r') 
cm_h5.keys() 

system_key = cm_h5['system'] 
cm_sys_key = h5py.File('./sys_labels.h5', 'w') 
cm_sys_key.create_dataset('labels', data=system_key.value)  
cm_sys_key.close() 

cm_sys_key = h5py.File('./sys_labels.h5', 'r') 
cm_sys_key['labels'] 
cm_sys_key.close() 
cm_h5.close() 
