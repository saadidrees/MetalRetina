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
hostname=socket.gethostname()
if hostname=='sandwolf':
    base = '/home/saad/data/'
elif hostname=='sandhound':
    base = '/home/saad/postdoc_db/'
    
    

data_pers = 'ej'
list_suffix = '20241115M' #'8M'#'20241115M'
testList = 'testList_'+list_suffix
trainList = 'trainList_'+list_suffix


path_dataset_base = '/home/saad/data/Dropbox/postdoc/analyses/data_ej/'
fold = 'local'
path_model_base = os.path.join('/home/saad/data/analyses/data_ej/models/',fold)
path_single_base =  '/home/saad/data/analyses/data_ej/models/cluster/'


with open(os.path.join(path_dataset_base,'datasets',testList+'.txt'), 'r') as f:
    expDates = f.readlines()
expDates_test = [line.strip() for line in expDates][1:]

with open(os.path.join(path_dataset_base,'datasets',trainList+'.txt'), 'r') as f:
    expDates_train = f.readlines()
expDates_train = [line.strip() for line in expDates_train][1:]


ft_expDate = expDates_test[18] #[22, 17, 18, 21, 12, 14,  6,  1, 16,  8, 11,  3, 15,  5, 19,  2,  7,13,  0, 20, 10, 23,  9,  4]
# [0.49523473, 0.41217554, 0.4841738 , 0.39615703, 0.51754504,
#         0.22054938, 0.57381499, 0.38592315, 0.27572663, 0.43924108,
#         0.37556186, 0.52105573, 0.58093581, 0.49063322, 0.34417585,
#         0.44153339, 0.2997238 , 0.35031742, 0.35891035, 0.08973839,
#         0.31206261, 0.40177846, 0.49752712, 0.16824725]

dataset_suffix = 'CB_mesopic_f4_8ms_sig-4_MAPS'

pt_mdl_name = 'CNN2D_MAP2' 

ft_mdl_name = pt_mdl_name
ft_trainingSamps_dur = -1
batch_size = 256
nb_epochs= 10
ft_lr = 1e-4
ft_lrscheduler = 'constant'
validationSamps_dur = 0.5
CONTINUE_TRAINING=0

# Pre-trained model params
APPROACH='metalzero'
LOSS_FUN='mad'
pt_mdl_name = pt_mdl_name
U = len(expDates_train)#32#32
lr_pretrained = 0.001
temporal_width=70
chan1_n=64; filt1_size=3
chan2_n=64; filt2_size=3
chan3_n=128; filt3_size=3
chan4_n=128; filt4_size=3
MaxPool=0
trainingSamps_dur = 2 #1#20 #-1 #0.05 # minutes per dataset


fname_model,dict_params =   modelFileName(U=U,P=0,T=temporal_width,CB_n=0,
                                                    C1_n=chan1_n,C1_s=filt1_size,C1_3d=0,
                                                    C2_n=chan2_n,C2_s=filt2_size,C2_3d=0,
                                                    C3_n=chan3_n,C3_s=filt3_size,C3_3d=0,
                                                    C4_n=chan4_n,C4_s=filt4_size,C4_3d=0,
                                                    BN=1,MP=MaxPool,LR=lr_pretrained,TR=1,TRSAMPS=trainingSamps_dur)

path_pretrained = os.path.join(path_model_base,APPROACH,trainList,LOSS_FUN,pt_mdl_name,fname_model+'/')
fname_pretrained = os.path.split(path_pretrained[:-1])[-1]

assert os.path.exists(path_pretrained), 'Model does not exist'


ft_fname_data_train_val_test = os.path.join(path_dataset_base,'datasets',ft_expDate+'_dataset_train_val_test_'+dataset_suffix+'.h5')

# %%

performance_finetuning,params_orig,params_final = metalmap_finetune.run_finetune(ft_expDate,path_pretrained,ft_fname_data_train_val_test,ft_mdl_name,ft_trainingSamps_dur=ft_trainingSamps_dur,
                                                            validationSamps_dur=validationSamps_dur,nb_epochs=nb_epochs,ft_lr=ft_lr,batch_size=batch_size,
                                                            job_id=0,saveToCSV=1,CONTINUE_TRAINING=CONTINUE_TRAINING)
