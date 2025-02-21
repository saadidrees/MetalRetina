#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 17:42:54 2021

@author: saad
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from metalmap_run_models import run_model
import socket
hostname=socket.gethostname()
if hostname=='sandwolf':
    base = '/home/saad/data/'
elif hostname=='sandhound':
    base = '/home/saad/postdoc_db/'

# base = '/home/saad/postdoc_db/'
base = '/home/saad/data/'


data_pers = 'ej'
expDate = ('trainList_test',) #('trainList_20241115a',)

APPROACH = 'metalzero' 
expFold = APPROACH 
subFold = 'test'
dataset = 'CB_mesopic_f4_8ms_sig-4'#'NATSTIM6_CORR2_mesopic-Rstar_f4_8ms',)#'NATSTIM3_CORR_mesopic-Rstar_f4_8ms  CB_CORR_mesopic-Rstar_f4_8ms
idx_unitsToTake = 0#np.arange(0,230) #np.array([0,1,2,3,4,5,6,7,8,9])
frac_train_units = 0.95

#np.arange(0,50)#idx_units_ON_train #[0] #idx_units_train
select_rgctype=0
mdl_subFold = ''
mdl_name = 'CNN2D_MAP' 
pr_params_name = ''
path_existing_mdl = ''
transfer_mode = ''
info = ''
idxStart_fixedLayers = 0#1
idxEnd_fixedLayers = -1#15   #29 dense; 28 BN+dense; 21 conv+dense; 15 second conv; 8 first conv
CONTINUE_TRAINING = 0

lr = 0.00001
lr_fac = 1# how much to divide the learning rate when training is resumed
use_lrscheduler=1
lrscheduler='constant' #'exponential_decay' #dict(scheduler='stepLR',drop=0.01,steps_drop=20,initial_lr=lr)
USE_CHUNKER=1
pr_temporal_width = 0
temporal_width=60
thresh_rr=0
chans_bp = 0
chan1_n=64#15
filt1_size=3
filt1_3rdDim=0
chan2_n=64#30
filt2_size=3
filt2_3rdDim=0
chan3_n=128#40
filt3_size=3
filt3_3rdDim=0
chan4_n=128#50
filt4_size=3
filt4_3rdDim=0
nb_epochs=10#42         # setting this to 0 only runs evaluation
bz_ms=2#64#10000#5000
BatchNorm=1
MaxPool=2
runOnCluster=0
num_trials=1

BatchNorm_train = 1
saveToCSV=1
trainingSamps_dur = 1#1#20 #-1 #0.05 # minutes per dataset
validationSamps_dur=0.5
testSamps_dur=0.5
USE_WANDB = 0



dataset_nameForPaths = ''
if 'trainList' in expDate[0]:
    dataset_nameForPaths = expDate[0]
else:
    for i in range(len(expDate)):
        dataset_nameForPaths = dataset_nameForPaths+expDate[i]+'+'
    dataset_nameForPaths = dataset_nameForPaths[:-1]

    
path_model_save_base = os.path.join(base,'analyses/data_'+data_pers+'/','models',subFold,expFold,dataset_nameForPaths,mdl_subFold)
path_dataset_base = os.path.join('/home/saad/postdoc_db/analyses/data_'+data_pers+'/')


if 'trainList' in expDate[0]:
    fname_data_train_val_test = os.path.join(path_dataset_base,'datasets',expDate[0]+'.txt')
else:
    fname_data_train_val_test = ''
    i=0
    for i in range(len(expDate)):
        name_datasetFile = expDate[i]+'_dataset_train_val_test_'+dataset+'.h5'
        fname_data_train_val_test = fname_data_train_val_test+os.path.join(path_dataset_base,'datasets',name_datasetFile) + '+'
    fname_data_train_val_test = fname_data_train_val_test[:-1]
    

c_trial = 1

if path_existing_mdl=='' and idxStart_fixedLayers>0:
    raise ValueError('Transfer learning set. Define existing model path')
    
# %%
for c_trial in range(1,num_trials+1):
    model_performance,mdl = run_model(expFold,mdl_name,path_model_save_base,fname_data_train_val_test,
                            path_dataset_base=path_dataset_base,
                            saveToCSV=saveToCSV,runOnCluster=0,
                            temporal_width=temporal_width, thresh_rr=thresh_rr,frac_train_units=frac_train_units,
                            pr_temporal_width=pr_temporal_width,pr_params_name=pr_params_name,
                            chans_bp=chans_bp,
                            chan1_n=chan1_n, filt1_size=filt1_size, filt1_3rdDim=filt1_3rdDim,
                            chan2_n=chan2_n, filt2_size=filt2_size, filt2_3rdDim=filt2_3rdDim,
                            chan3_n=chan3_n, filt3_size=filt3_size, filt3_3rdDim=filt3_3rdDim,
                            nb_epochs=nb_epochs,bz_ms=bz_ms,
                            BatchNorm=BatchNorm,BatchNorm_train = BatchNorm_train,MaxPool=MaxPool,c_trial=c_trial,USE_CHUNKER=USE_CHUNKER,
                            path_existing_mdl = path_existing_mdl, idxStart_fixedLayers=idxStart_fixedLayers, idxEnd_fixedLayers=idxEnd_fixedLayers,transfer_mode=transfer_mode,
                            CONTINUE_TRAINING=CONTINUE_TRAINING,info=info,
                            trainingSamps_dur=trainingSamps_dur,validationSamps_dur=validationSamps_dur,idx_unitsToTake=idx_unitsToTake,
                            lr=lr,lr_fac=lr_fac,use_lrscheduler=use_lrscheduler,lrscheduler=lrscheduler,USE_WANDB=USE_WANDB,APPROACH=APPROACH)
    
plt.plot(model_performance['fev_medianUnits_allEpochs']);plt.ylabel('FEV');plt.xlabel('Epochs')
print('FEV = %0.2f' %(np.nanmax(model_performance['fev_medianUnits_allEpochs'])*100))
