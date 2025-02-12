#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 4

@author: saad
"""

# Save training, testing and validation datasets to be read by jobs on cluster

import os
import re
import h5py
import numpy as np
from model.data_handler_ej import load_data_allLightLevels_cb, save_h5Dataset
from model import data_handler_ej
from model import handler_maps
from collections import namedtuple
from model.utils_si import h5_tostring
Exptdata = namedtuple('Exptdata', ['X', 'y'])
Exptdata_spikes = namedtuple('Exptdata', ['X', 'y','spikes'])


def stim_vecToMat(data,num_checkers_y,num_checkers_x):
    X = data.X
    X = np.reshape(X,(X.shape[0],num_checkers_y,num_checkers_x),order='F')    # convert stim frames back into spatial dimensions
    data = Exptdata_spikes(X,data.y,data.spikes)
    return data


datasetsToLoad = ['mesopic',]#,'photopic'];    #['scotopic','photopic','scotopic_photopic']
N_split = 0
frac_train_units = 0.90

STIM = 'CB' # 'CB'  'NATSTIM'
t_frame = 8
sig = 4
file_suffix = ''
NORM_STIM = 0
NORM_RESP = True
D_TYPE = 'f4'


expFold = 'RGB-16-2-0.48-11111' #'RGB-8-1-0.48-11111'
path_expFold = os.path.join('/home/saad/postdoc_db/analyses/data_ej',expFold)

expList = []
with open(os.path.join(path_expFold,'datasets.txt'), 'r') as file:
    for line in file:
        filename = line.strip()
        expList.append(filename[:-4])


exp_ctr = 0
exp = expList[0]
# exp='2007-03-02-0'
# %

for exp in expList:
    exp_ctr = exp_ctr+1
    print('File %d of %d'%(exp_ctr,len(expList)))
    
    expDate = exp #'2018-02-09-3'    
    path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_ej/datasets/')
    path_rfs = os.path.join('/home/saad/postdoc_db/analyses/data_ej/stas/')
    path_save = os.path.join('/home/saad/postdoc_db/analyses/data_ej/datasets/')
    # path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_kiersten/',expDate,'gradient_analysis/datasets/')
        
    
    fname_dataFile = os.path.join(path_dataset,(expDate+'_dataset_'+STIM+'_'+str(t_frame)+'ms_sig-'+str(sig)+'.h5'))
    fname_rf = os.path.join(path_rfs,(expDate+'_rfs.h5'))
    
    
    filt_temporal_width = 0
    idx_cells = None
    thresh_rr = 0
    
    frac_val = 0.05
    frac_test = 0.05 
    
    dataset = datasetsToLoad[0]

    for dataset in datasetsToLoad:
    
        data_train,data_val,data_test,data_quality,dataset_rr,resp_orig = load_data_allLightLevels_cb(fname_dataFile,dataset,frac_val=frac_val,frac_test=frac_test,
                                                                                                   filt_temporal_width=filt_temporal_width,idx_cells_orig=idx_cells,
                                                                                                   resp_med_grand=None,thresh_rr=thresh_rr,N_split=N_split,
                                                                                                   CHECK_CONTAM = False,NORM_RESP=NORM_RESP)
        
        
        with h5py.File(fname_dataFile) as f:
            obs_noise = np.zeros(len(f['units']))
    
        data_quality['var_noise'] =  obs_noise
    
        fname_data_train_val_test = os.path.join(path_save,(expDate+'_dataset_train_val_test_'+STIM+'_'+dataset+'_'+D_TYPE+'_'+str(t_frame)+'ms_sig-'+str(sig)+'_MAPS'))
        
        f = h5py.File(fname_dataFile,'r')
        samps_shift = 0#np.array(f[dataset]['val']['spikeRate'].attrs['samps_shift'])
        if 'num_checkers_x' in f[dataset]['train']['stim_frames'].attrs.keys():
            num_checkers_x = np.array(f[dataset]['train']['stim_frames'].attrs['num_checkers_x'])
            num_checkers_y = np.array(f[dataset]['train']['stim_frames'].attrs['num_checkers_y'])
            checkSize_um = np.array(f[dataset]['train']['stim_frames'].attrs['checkSize_um'])
            stimResampFac = np.array(f[dataset]['train']['stim_frames'].attrs['stimResampFac'])
        else:
            num_checkers_x = np.array(f[dataset]['train']['stim_frames'].shape[2])
            num_checkers_y = np.array(f[dataset]['train']['stim_frames'].shape[1])
            checkSize_um = 3.8  # 3.8 um/pixel
            
        
        # rf_centers = np.array(f[dataset]['train']['rf_centers'])
        unames =  h5_tostring(f['units'])
        unit_types = handler_maps.get_unit_types(unames)

        with h5py.File(fname_rf,'r') as f_rf:
            rf_centers = np.array(f_rf['centers'])
            
        unit_locs = np.round(rf_centers).astype('int')
        

        
        # data_train,rf_center_int,unit_types = buildRespMap(data_train,rf_centers,unames)
        # data_val,_,_ = buildRespMap(data_val,rf_centers,unames)
        # data_test,_,_ = buildRespMap(data_test,rf_centers,unames)
        

        t_frame_inData = np.array(f[dataset]['train']['stim_frames'].attrs['t_frame'])
        parameters = {
        't_frame': t_frame_inData,
        'filt_temporal_width': filt_temporal_width,
        'frac_val': frac_val,
        'frac_test':frac_test,
        'thresh_rr': thresh_rr,
        'samps_shift': samps_shift,
        'num_checkers_x': num_checkers_x,
        'num_checkers_y': num_checkers_y,
        'checkSize_um': np.array(checkSize_um,dtype='float32'),
        'unames':unames,
        'unit_locs': unit_locs,
        'unit_types': unit_types,
        'frac_train_units':frac_train_units
        }
        
        
        dinf = {}
        dinf['unit_locs'] = parameters['unit_locs']
        dinf['unit_types'] = parameters['unit_types']
        dinf['unames'] = parameters['unames']

        dinf,data_train,data_val = handler_maps.remove_boundary_units(dinf,data_train,data_val)
        info_unitSplit = handler_maps.units_validation_split(dinf['unames'],dinf['unit_locs'],frac_train_units)
        
        parameters['unames'] = dinf['unames']
        parameters['unit_locs'] = dinf['unit_locs']
        parameters['unit_types'] = dinf['unit_types']
        
        del info_unitSplit['unames_val_perType']
        del info_unitSplit['unames_train_perType']
        

        f.close()
        
        save_h5Dataset(fname_data_train_val_test+'.h5',data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig=resp_orig,dtype=D_TYPE)
        data_handler_ej.save_info_unitSplit(fname_data_train_val_test+'.h5',info_unitSplit)
