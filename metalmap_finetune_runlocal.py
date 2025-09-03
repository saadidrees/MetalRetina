#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 11:34:15 2025

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import metalmap_finetune
from model.utils_si import modelFileName
import socket
from model.models_jax import getModelParams
hostname=socket.gethostname()
if hostname=='sandwolf':
    base = '/home/saad/data/'
elif hostname=='sandhound':
    base = '/home/saad/postdoc_db/'
    
    

data_pers = 'ej'
list_suffix = '20241115RstarM' #'8M'#'20241115M'
testList = 'testlist_'+list_suffix
trainList = 'trainlist_'+list_suffix


path_dataset_base = '/home/saad/data/Dropbox/postdoc/analyses/data_ej/'
fold = 'cluster2'
path_model_base = os.path.join('/home/saad/data/analyses/data_ej/models/',fold)

ft_path_model_base = '/home/saad/data/analyses/data_ej/models/finetuned_models/'

with open(os.path.join(path_dataset_base,'datasets',testList+'.txt'), 'r') as f:
    expDates = f.readlines()
expDates_test = [line.strip() for line in expDates][1:]

with open(os.path.join(path_dataset_base,'datasets',trainList+'.txt'), 'r') as f:
    expDates_train = f.readlines()
expDates_train = [line.strip() for line in expDates_train]
dataset_suffix = expDates_train[0]
expDates_train = expDates_train[1:]


ft_expDate = expDates_test[2] #[22, 17, 18, 21, 12, 14,  6,  1, 16,  8, 11,  3, 15,  5, 19,  2,  7,13,  0, 20, 10, 23,  9,  4]

# dataset_suffix = 'CB_mesopic_f4_8ms_sig-4_MAPS'

pt_mdl_name = 'PRFR_CNN2D_MAP' 

ft_mdl_name = pt_mdl_name
ft_trainingSamps_dur = 15
batch_size = 64
nb_epochs= 10
ft_lr = 1e-4
ft_lrscheduler = 'constant'
validationSamps_dur = 0.5
CONTINUE_TRAINING=1

# Pre-trained model params
# APPROACH='metalzero'
# LOSS_FUN='mad'
# pt_mdl_name = pt_mdl_name
# U = len(expDates_train)#32#32
# lr_pretrained = 0.001
# pr_temporal_width=100
# temporal_width=80
# chan1_n=32; filt1_size=7
# chan2_n=64; filt2_size=7
# chan3_n=128; filt3_size=7
# chan4_n=0; filt4_size=0
# MaxPool=1
# trainingSamps_dur = 15 #1#20 #-1 #0.05 # minutes per dataset


# fname_model,dict_params =   modelFileName(U=U,P=pr_temporal_width,T=temporal_width,CB_n=0,
#                                                     C1_n=chan1_n,C1_s=filt1_size,C1_3d=0,
#                                                     C2_n=chan2_n,C2_s=filt2_size,C2_3d=0,
#                                                     C3_n=chan3_n,C3_s=filt3_size,C3_3d=0,
#                                                     C4_n=chan4_n,C4_s=filt4_size,C4_3d=0,
#                                                     BN=1,MP=MaxPool,LR=lr_pretrained,TR=1,TRSAMPS=trainingSamps_dur)

path_pretrained = '/home/saad/data/analyses/data_ej/models/cluster2/metalzero/trainlist_20241115RstarM/mad/PRFR_CNN2D_MAP/fr_cones_gammalarge/U-71_P-100_T-080_C1-32-07_C2-64-07_C3-128-07_BN-1_MP-0_LR-0.0001_TRSAMPS-015_TR-01/'
assert os.path.exists(path_pretrained), 'Model does not exist'


ft_fname_data_train_val_test = os.path.join(path_dataset_base,'datasets',ft_expDate+'_dataset_train_val_test_'+dataset_suffix+'.h5')

# %%

performance_finetuning,params_orig,params_final = metalmap_finetune.run_finetune(ft_expDate,path_pretrained,ft_fname_data_train_val_test,ft_mdl_name,
                                                                                 ft_path_model_base=ft_path_model_base,ft_trainingSamps_dur=ft_trainingSamps_dur,
                                                            validationSamps_dur=validationSamps_dur,nb_epochs=nb_epochs,ft_lr=ft_lr,batch_size=batch_size,
                                                            job_id=0,saveToCSV=1,CONTINUE_TRAINING=CONTINUE_TRAINING)
