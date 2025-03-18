#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:06:03 2025

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

import numpy as np
import os
import time
import csv
import h5py
import glob
import re
import matplotlib.pyplot as plt
import cloudpickle
import gc
import datetime
from collections import namedtuple

from model.data_handler import prepare_data_cnn2d_maps,isintuple
from model import data_handler_ej
from model.data_handler_mike import load_h5Dataset
from model.performance import model_evaluate_new, estimate_noise
import model.utils_si
import seaborn as snb

# %% Single retina models - Test retinas

list_suffix = '20241115M' #'8M'#'20241115M'
testList = 'testList_'+list_suffix
trainList = 'trainList_'+list_suffix

path_dataset_base = '/home/saad/postdoc_db/analyses/data_ej/'
# path_base_single =  '/home/saad/data_hdd/analyses/data_ej/models_correct/cluster/'
path_base_single =  '/home/saad/data/analyses/data_ej/models/cluster/'


with open(os.path.join(path_dataset_base,'datasets',testList+'.txt'), 'r') as f:
    expDates_test = f.readlines()
expDates_test = [line.strip() for line in expDates_test][1:]

with open(os.path.join(path_dataset_base,'datasets',trainList+'.txt'), 'r') as f:
    expDates_train = f.readlines()
expDates_train = [line.strip() for line in expDates_train][1:]


# trainingDurs = np.array([0.03,0.04,0.15,1,2.5,5,10,15,20,25,30])
trainingDurs = np.array([30])
select_rgctype = np.array([1,2])

U=762
fev_test_lastEpoch_AllExps = np.zeros((len(expDates_test),len(trainingDurs),U));fev_test_lastEpoch_AllExps[:]=np.nan
corr_test_lastEpoch_AllExps = np.zeros((len(expDates_test),len(trainingDurs),U));corr_test_lastEpoch_AllExps[:]=np.nan
num_rgcs_allExps = np.zeros((len(expDates_test)));num_rgcs_allExps[:]=np.nan


path_mdlFold = os.path.join(path_base_single,'testing_retinas')
i=0;j=0
fname_missing_scratch = []
for i in range(len(expDates_test)):
    exp = expDates_test[i]

    path_single = os.path.join(path_mdlFold,exp,'CNN2D_FT')
    rgb = os.listdir(path_single)
    params = model.performance.getModelParams(rgb[0])
    nrgcs = params['U']


    for j in range(len(trainingDurs)):

        fname_model = 'U-%d_T-080_C1-20-03_C2-30-03_C3-30-03_C4-30-03_BN-1_MP-2_LR-0.001_TRSAMPS-%03d_TR-01'%(nrgcs,trainingDurs[j])
    
        fname_perf = os.path.join(path_single,fname_model,'performance','%s_%s.pkl'%(exp,fname_model))
        if os.path.exists(fname_perf):

            with open(fname_perf,'rb') as f:
                rgb = cloudpickle.load(f)
            perf_data = rgb[3]
            
            fev_test_allEpochs = perf_data['fev_test_allUnits_allEpochs']
            corr_test_allEpochs = perf_data['corr_test_allUnits_allEpochs']
            
            
            fev_test_lastEpoch = perf_data['fev_test_allUnits_bestEpoch']
            corr_test_lastEpoch = perf_data['corr_test_allUnits_bestEpoch']
                
            n_epochs_exp = 45#len(fev_val_allEpochs)

            fev_test_lastEpoch_AllExps[i,j,:len(fev_test_lastEpoch)] = fev_test_lastEpoch
            corr_test_lastEpoch_AllExps[i,j,:len(corr_test_lastEpoch)] = corr_test_lastEpoch

        else:
            fname_missing_scratch.append(fname_perf)
    
# %% Single retina models - Training retinas



with open(os.path.join(path_dataset_base,'datasets',trainList+'.txt'), 'r') as f:
    expDates_train = f.readlines()
expDates_train = [line.strip() for line in expDates_train][1:]


# trainingDurs = np.array([0.03,0.04,0.15,1,2.5,5,10,15,20,25,30])
select_rgctype = np.array([1,2])

U=762
fev_train_lastEpoch_AllExps = np.zeros((len(expDates_train),len(trainingDurs),U));fev_train_lastEpoch_AllExps[:]=np.nan
corr_train_lastEpoch_AllExps = np.zeros((len(expDates_train),len(trainingDurs),U));corr_train_lastEpoch_AllExps[:]=np.nan
num_rgcs_allExps_train = np.zeros((len(expDates_train)));num_rgcs_allExps_train[:]=np.nan


path_mdlFold = os.path.join(path_base_single,'training_retinas')
i=0;j=0
fname_missing_scratch = []
for i in range(len(expDates_train)):
    exp = expDates_train[i]


    path_single = os.path.join(path_mdlFold,exp,'CNN2D_FT')
    rgb = os.listdir(path_single)
    params = model.performance.getModelParams(rgb[0])
    nrgcs = params['U']


    for j in range(len(trainingDurs)):
        fname_model = 'U-%d_T-080_C1-20-03_C2-30-03_C3-30-03_C4-30-03_BN-1_MP-2_LR-0.001_TRSAMPS--01_TR-01'%nrgcs
    
        fname_perf = os.path.join(path_single,fname_model,'performance','%s_%s.pkl'%(exp,fname_model))
        if os.path.exists(fname_perf):

            with open(fname_perf,'rb') as f:
                rgb = cloudpickle.load(f)
            perf_data = rgb[3]
            
            fev_test_allEpochs = perf_data['fev_allUnits_allEpochs']
            corr_test_allEpochs = perf_data['predCorr_allUnits_allEpochs']
            
            
            fev_test_lastEpoch = perf_data['fev_allUnits_bestEpoch']
            corr_test_lastEpoch = perf_data['predCorr_allUnits_bestEpoch']
                
            n_epochs_exp = 45#len(fev_val_allEpochs)

            fev_train_lastEpoch_AllExps[i,j,:len(fev_test_lastEpoch)] = fev_test_lastEpoch
            corr_train_lastEpoch_AllExps[i,j,:len(corr_test_lastEpoch)] = corr_test_lastEpoch

        else:
            fname_missing_scratch.append(fname_perf)

# %% MAPS
path_base = '/home/saad/data/analyses/data_ej/models/cluster'
# f_mamlmaps = os.path.join(path_base,'metalzero',trainList,'poisson',
#                           'CNN2D_MAP2/U-71_T-070_C1-64-03_C2-64-03_C3-128-03_C4-128-03_BN-1_MP-0_LR-0.001_TRSAMPS--01_TR-01/performance/metalzero_perf_trainingExps_lastEpoch.pkl')

f_mamlmaps = os.path.join(path_base,'metalzero',trainList,'poisson',
                          'CNN2D_MAP2/U-71_T-070_C1-64-03_C2-64-03_C3-128-03_C4-128-03_BN-1_MP-0_LR-0.001_TRSAMPS--01_TR-01/performance/metalzero_perf_testExps_lastEpoch.pkl')

with open(os.path.join(f_mamlmaps), 'rb') as f:
    perf_mamlmaps = cloudpickle.load(f)

med_mamlmaps = perf_mamlmaps[0]['predCorr_medUnits_allExps']
# idx_train = np.arange(24,95)

# med_mamlmaps = med_mamlmaps[idx_train]
# %%
med_singleRetVec_test = np.nanmedian(corr_test_lastEpoch_AllExps,axis=-1)
med_singleRetVec_train = np.nanmedian(corr_train_lastEpoch_AllExps,axis=-1)

med_singleRetVec = med_singleRetVec_test#np.concatenate((med_singleRetVec_test,med_singleRetVec_train),axis=0)


lim_y = [0,0.7]
lim_x = lim_y
select_dur = -1      #trainingDurs = np.array([0.03,0.04,0.15,1,2.5,5,10,15,20,25,30])
fig,ax = plt.subplots(1,1,figsize=(4,4));fig.suptitle('Prediction correlations | Exp: %s'%(list_suffix))
cols = ['black','red']
ax.scatter(med_singleRetVec[:,select_dur],med_mamlmaps,20,color='k')
ax.plot([-1,1],[-1,1],'--b')
ax.set_ylabel('Zero-Shot | Maps ')
ax.set_xlabel('Single Retina trained with %d mins | Vecs'%trainingDurs[select_dur])
ax.legend(loc='lower right')
ax.set_xlim(lim_x)
ax.set_ylim(lim_y)

# %%
diff = med_mamlmaps-med_singleRetVec[:,0]
diff_mean = np.nanmean(diff)
plt.bar(np.arange(len(med_mamlmaps)), diff, color=['red' if d < 0 else 'blue' for d in diff])
plt.plot([0,71],[diff_mean,diff_mean],'--k')
plt.xlabel('retinas');plt.ylabel('corr diff')

# %%

fname_ft = '/mnt/devices/nvme-2tb/Dropbox/postdoc/projects/RetinaPredictors/run_jax/test_rets_ft.pkl'
with open(os.path.join(fname_ft), 'rb') as f:
    perf_ft = cloudpickle.load(f)

med_ft = perf_ft[0]

lim_y = [0,0.7]
lim_x = lim_y
select_dur = -1      #trainingDurs = np.array([0.03,0.04,0.15,1,2.5,5,10,15,20,25,30])
fig,ax = plt.subplots(1,1,figsize=(4,4));fig.suptitle('Prediction correlations | Exp: %s'%(list_suffix))
cols = ['black','red']
ax.scatter(med_ft,med_mamlmaps[:len(expDates_test)],20,color='k')
ax.plot([-1,1],[-1,1],'--b')
ax.set_ylabel('Zero-Shot | Maps ')
ax.set_xlabel('Fine-tuned | Vecs'%trainingDurs[select_dur])
ax.legend(loc='lower right')
ax.set_xlim(lim_x)
ax.set_ylim(lim_y)

