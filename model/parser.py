#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:31:24 2021

@author: saad
"""
import argparse

def parser_run_model():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('expFold',type=str)
    parser.add_argument('mdl_name',type=str)
    parser.add_argument('path_model_save_base',type=str)
    parser.add_argument('fname_data_train_val_test',type=str)
    parser.add_argument('--select_rgctype',type=str)
    parser.add_argument('--path_dataset_base',type=str,default='/home/saad/data/analyses/data_kiersten')
    parser.add_argument('--path_existing_mdl',type=str,default='')
    parser.add_argument('--transfer_mode',type=str,default='finetuning')
    parser.add_argument('--saveToCSV',type=int,default=1)
    parser.add_argument('--temporal_width',type=int,default=40)
    parser.add_argument('--pr_temporal_width',type=int,default=180)
    parser.add_argument('--pr_params_name',type=str,default='')
    parser.add_argument('--thresh_rr',type=str2int,default=0)
    parser.add_argument('--chan1_n',type=int,default=8)
    parser.add_argument('--filt1_size',type=int,default=13)
    parser.add_argument('--filt1_3rdDim',type=int,default=20)
    parser.add_argument('--chan2_n',type=int,default=0)
    parser.add_argument('--filt2_size',type=int,default=0)
    parser.add_argument('--filt2_3rdDim',type=int,default=0)
    parser.add_argument('--chan3_n',type=int,default=0)    
    parser.add_argument('--filt3_size',type=int,default=0)
    parser.add_argument('--filt3_3rdDim',type=int,default=0)
    parser.add_argument('--chan4_n',type=int,default=0)    
    parser.add_argument('--filt4_size',type=int,default=0)
    parser.add_argument('--filt4_3rdDim',type=int,default=0)
    parser.add_argument('--nb_epochs',type=int,default=100)
    parser.add_argument('--bz_ms',type=int,default=10000)
    parser.add_argument('--runOnCluster',type=int,default=0)
    parser.add_argument('--BatchNorm',type=int,default=1)
    parser.add_argument('--MaxPool',type=int,default=1)
    parser.add_argument('--c_trial',type=int,default=1)
    parser.add_argument('--USE_CHUNKER',type=int,default=0)
    parser.add_argument('--trainingSamps_dur',type=str2int,default=-1)
    parser.add_argument('--validationSamps_dur',type=str2int,default=0.3)
    parser.add_argument('--testSamps_dur',type=str2int,default=0.3)
    parser.add_argument('--CONTINUE_TRAINING',type=int,default=0)
    parser.add_argument('--idxStart_fixedLayers',type=int,default=0)
    parser.add_argument('--idxEnd_fixedLayers',type=int,default=-1)
    parser.add_argument('--info',type=str,default='')
    parser.add_argument('--lr',type=str2int,default=0.01)
    parser.add_argument('--lr_fac',type=str2int,default=1)
    parser.add_argument('--use_lrscheduler',type=str2int,default=1)
    parser.add_argument('--lrscheduler',type=str,default='constant')
    parser.add_argument('--idx_unitsToTake',type=int,default=0)
    parser.add_argument('--job_id',type=int,default=0)
    parser.add_argument('--USE_WANDB',type=int,default=0)
    parser.add_argument('--APPROACH',type=str,default='maml')

    

    args = parser.parse_args()
    
    return args

def parser_finetune():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('ft_expDate',type=str)
    parser.add_argument('pretrained_expDates',type=str)
    parser.add_argument('path_model_base',type=str)
    parser.add_argument('path_pretrained',type=str)
    parser.add_argument('ft_fname_data_train_val_test',type=str)
    parser.add_argument('ft_mdl_name',type=str)
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--ft_trainingSamps_dur',type=str2int,default=-1)
    parser.add_argument('--validationSamps_dur',type=str2int,default=0.5)
    parser.add_argument('--testSamps_dur',type=str2int,default=0.5)
    parser.add_argument('--ft_nb_epochs_A',type=int,default=2)
    parser.add_argument('--ft_nb_epochs_B',type=int,default=18)
    parser.add_argument('--ft_lr_A',type=str2int,default=0.1)
    parser.add_argument('--ft_lr_B',type=str2int,default=0.05)
    parser.add_argument('--saveToCSV',type=int,default=1)

    args = parser.parse_args()
    
    return args





def int2bool(v):
    print(v)
    if isinstance(v, bool):
       return v
    elif v == 1:
        return True
    elif v==0:
        return False
    # if v.lower() in ('yes', 'true', 't', 'y', '1'):
    #     return True
    # elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    #     return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
def str2int(v):
    if isinstance(v, str):
       return float(v)

    else:
        return v

        
        
        
