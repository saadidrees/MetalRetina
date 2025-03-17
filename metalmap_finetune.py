#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 08:16:18 2025

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""


# %% import needed modules
import numpy as np
import os
import time
import csv
import h5py
import glob
import re
import matplotlib.pyplot as plt
import cloudpickle
import jax.numpy as jnp
import jax
import optax
import orbax
import gc
import datetime
from collections import namedtuple
from tqdm.auto import tqdm
import shutil

from model.data_handler import prepare_data_cnn2d_maps,isintuple
from model import data_handler_ej
from model.data_handler_mike import load_h5Dataset
from model.performance import model_evaluate_new, estimate_noise
import model.paramsLogger
import model.utils_si

from model import models_jax, train_singleretunits, dataloaders,handler_maps
from model import train_metalmaps
from torch.utils.data import DataLoader

from model.performance import getModelParams


import seaborn as snb

Exptdata = namedtuple('Exptdata', ['X', 'y'])
Exptdata_spikes = namedtuple('Exptdata', ['X', 'y','spikes'])

# %

list_suffix = '20241115M' #'8M'#'20241115M'
testList = 'testList_'+list_suffix
trainList = 'trainList_'+list_suffix

path_dataset_base = '/home/saad/data/Dropbox/postdoc/analyses/data_ej/'

with open(os.path.join(path_dataset_base,'datasets',testList+'.txt'), 'r') as f:
    expDates = f.readlines()
expDates_test = [line.strip() for line in expDates][1:]

with open(os.path.join(path_dataset_base,'datasets',trainList+'.txt'), 'r') as f:
    expDates_train = f.readlines()
expDates_train = [line.strip() for line in expDates_train][1:]


expDates = expDates_test#expDates_train

fold = 'test'
path_base = os.path.join('/home/saad/data/analyses/data_ej/models/',fold)
path_base_single =  '/home/saad/data/analyses/data_ej/models/cluster/'

dataset_suffix = 'CB_mesopic_f4_8ms_sig-4_MAPS'
validationSamps_dur = 0.5

# Pretrained model params
APPROACH='metalzero'
LOSS_FUN='mad'
mdl_name = 'CNN2D_MAP2'
U = len(expDates_train)#32#32
lr_pretrained = 0.001
temporal_width=70
chan1_n=64; filt1_size=3
chan2_n=64; filt2_size=3
chan3_n=128; filt3_size=3
chan4_n=128; filt4_size=3
MaxPool=0
trainingSamps_dur = 2 #1#20 #-1 #0.05 # minutes per dataset


fname_model,dict_params = model.utils_si.modelFileName(U=U,P=0,T=temporal_width,CB_n=0,
                                                    C1_n=chan1_n,C1_s=filt1_size,C1_3d=0,
                                                    C2_n=chan2_n,C2_s=filt2_size,C2_3d=0,
                                                    C3_n=chan3_n,C3_s=filt3_size,C3_3d=0,
                                                    C4_n=chan4_n,C4_s=filt4_size,C4_3d=0,
                                                    BN=1,MP=MaxPool,LR=lr_pretrained,TR=1,TRSAMPS=trainingSamps_dur)

path_pretrained = os.path.join(path_base,APPROACH,trainList,LOSS_FUN,mdl_name,fname_model+'/')
fname_pretrained = os.path.split(path_pretrained[:-1])[-1]

assert os.path.exists(path_pretrained), 'Model does not exist'

expDates = expDates[22:23]        #[ 1,  3,  5,  6,  8, 11, 12, 14, 15, 16, 17, 18, 21, 22]

# Finetuning
trainingSamps_dur = -1
bz = 64

# %% load train val and test datasets from saved h5 file
"""
    load_h5dataset is a function to load training and validation data from h5 dataset. We can extract all data or a subset using the nsamps arguments.
    data_train, val and test are named tuples. data_train.X contains the stimulus with dimensions [samples,y pixels, x pixels]
    and data_train.y contains the spikerate normalized by median [samples,numOfCells]
"""
BUILD_MAPS = False
MAX_RGCS = model.train_metalmaps.MAX_RGCS


    
# Check whether the filename has multiple datasets that need to be merged
    
fname_data_train_val_test_all = []
i=1
for i in range(len(expDates)):
    name_datasetFile = expDates[i]+'_dataset_train_val_test_'+dataset_suffix+'.h5'
    fname_data_train_val_test_all.append(os.path.join(path_dataset_base,'datasets',name_datasetFile))

# Get the total num of samples and RGCs in each dataset
nsamps_alldsets = []
num_rgcs_all = []
for d in range(len(fname_data_train_val_test_all)):
    with h5py.File(fname_data_train_val_test_all[d]) as f:
        nsamps_alldsets.append(f['data_train']['X'].shape[0])
        num_rgcs_all.append(f['data_train']['y'].shape[1])
nsamps_alldsets = np.asarray(nsamps_alldsets)
num_rgcs_all = np.asarray(num_rgcs_all)

# Load datasets
idx_train_start = 0    # mins to chop off in the begining.
psf_params = dict(pixel_neigh=1,method='cross')
FRAC_U_TRTR = 1
d=0
dict_train = {};dict_trtr={};dict_trval={}
dict_val = {}
dict_test = {}
unames_allDsets = []
data_info = {}
dict_dinf = {}
nsamps_alldsets_loaded = []

exp = expDates[d]
print('Loading dataset %d of %d'%(d+1,len(fname_data_train_val_test_all)))
rgb = load_h5Dataset(fname_data_train_val_test_all[d],nsamps_val=validationSamps_dur,nsamps_train=trainingSamps_dur,nsamps_test=0,  # THIS NEEDS TO BE TIDIED UP
                     idx_train_start=idx_train_start)
data_train=rgb[0]
data_val = rgb[1]
data_test = rgb[2]
data_quality = rgb[3]
dataset_rr = rgb[4]
parameters = rgb[5]
if len(rgb)>7:
    data_info = rgb[7]
    
info_unitSplit = data_handler_ej.load_info_unitSplit(fname_data_train_val_test_all[d])
info_unitSplit['unames_train'] = parameters['unames'][info_unitSplit['idx_train']]
# info_unitSplit['idx_val'] = info_unitSplit['idx_train']
info_unitSplit['unames_val'] = parameters['unames'][info_unitSplit['idx_val']]
# info_unitSplit['unames_val'] = parameters['unames'][info_unitSplit['idx_train']]


t_frame = parameters['t_frame']     # time in ms of one frame/sample 

frac_train_units = parameters['frac_train_units']

data_train,data_val,dinf = handler_maps.arrange_data_formaps(exp,data_train,data_val,parameters,frac_train_units,psf_params=psf_params,info_unitSplit=None,
                                                             BUILD_MAPS=BUILD_MAPS,MODE='validation')
dinf['unit_locs_train'] = dinf['unit_locs'][dinf['idx_units_train']]
dinf['unit_types_train'] = dinf['unit_types'][dinf['idx_units_train']]

dinf['unit_locs_val'] = dinf['unit_locs'][dinf['idx_units_val']]
dinf['unit_types_val'] = dinf['unit_types'][dinf['idx_units_val']]


dinf['umaskcoords_trtr'],dinf['umaskcoords_trval'],dinf['umaskcoords_trtr_remap'],dinf['umaskcoords_trval_remap'] = handler_maps.umask_metal_split(dinf['umaskcoords_train'],
                                                                                                                                                   FRAC_U_TRTR=FRAC_U_TRTR)

data_trtr,data_trval = handler_maps.prepare_metaldataset(data_train,dinf['umaskcoords_trtr'],dinf['umaskcoords_trval'],bgr=0,frac_stim_train=0.5,BUILD_MAPS=BUILD_MAPS)

del data_train, data_test


# dict_train[fname_data_train_val_test_all[d]] = data_train
dict_trtr[fname_data_train_val_test_all[d]] = data_trtr
dict_trval[fname_data_train_val_test_all[d]] = data_trval

dict_val[fname_data_train_val_test_all[d]] = data_val
# dict_test[fname_data_train_val_test_all[d]] = data_test
dict_dinf[fname_data_train_val_test_all[d]] = dinf
unames_allDsets.append(parameters['unames'])
# nsamps_alldsets_loaded.append(data_trtr.X.shape[0])
nsamps_alldsets_loaded.append(len(data_trtr.X))
    
n_tasks = len(dict_val)
cell_types_unique = np.unique(dinf['umaskcoords_train'][:,1])
    
nrgcs_trtr=[];nrgcs_trval=[];nrgcs_val=[];
for d in range(len(fname_data_train_val_test_all)):
    nrgcs_trtr.append(dict_trtr[fname_data_train_val_test_all[d]].y.shape[-1])
    nrgcs_trval.append(dict_trval[fname_data_train_val_test_all[d]].y.shape[-1])
    nrgcs_val.append(dict_val[fname_data_train_val_test_all[d]].y.shape[-1])


# %
if BUILD_MAPS==False:   
    idx_unitsToTake_all_trtr=[];idx_unitsToTake_all_trval=[];idx_unitsToTake_all_val = [];
    mask_unitsToTake_all_trtr=[];mask_unitsToTake_all_trval=[];mask_unitsToTake_all_val = [];
    d=0
    for d in range(n_tasks):
        idx_unitsToTake,mask_unitsToTake = handler_maps.get_expandedRGClist(dict_trtr[fname_data_train_val_test_all[d]],MAX_RGCS)
        idx_unitsToTake_all_trtr.append(idx_unitsToTake)
        mask_unitsToTake_all_trtr.append(mask_unitsToTake)
        
        idx_unitsToTake,mask_unitsToTake = handler_maps.get_expandedRGClist(dict_val[fname_data_train_val_test_all[d]],MAX_RGCS)
        idx_unitsToTake_all_val.append(idx_unitsToTake)
        mask_unitsToTake_all_val.append(mask_unitsToTake)



# Get unit names
uname_train = [];uname_val = [];
c_tr = np.zeros(len(cell_types_unique),dtype='int'); c_val = np.zeros(len(cell_types_unique),dtype='int');
c_exp_tr=np.zeros((len(expDates),len(cell_types_unique)),dtype='int');c_exp_val=np.zeros((len(expDates),len(cell_types_unique)),dtype='int');

for d in range(n_tasks):
    dinf = dict_dinf[fname_data_train_val_test_all[d]]
    rgb = dinf['unames'][dinf['idx_units_train']]
    exp_uname = [expDates[d]+'_'+u for u in rgb ]
    uname_train.append(exp_uname)
    
    for t in range(len(cell_types_unique)):
        c_tr[t]=c_tr[t]+(dinf['unit_types'][dinf['idx_units_train']]==cell_types_unique[t]).sum()
        c_exp_tr[d,t] = (dinf['unit_types'][dinf['idx_units_train']]==cell_types_unique[t]).sum()
    
    rgb = dinf['unames'][dinf['idx_units_val']]
    exp_uname = [expDates[d]+'_'+u for u in rgb ]
    uname_val.append(exp_uname)
    
    for t in range(len(cell_types_unique)):
        c_val[t]=c_val[t]+(dinf['unit_types'][dinf['idx_units_val']]==cell_types_unique[t]).sum()
        c_exp_val[d,t] = (dinf['unit_types'][dinf['idx_units_val']]==cell_types_unique[t]).sum()

print('Total number of datasets: %d'%n_tasks)
for t in range(len(cell_types_unique)):
    print('Trainining set | Cell type %d: %d RGCs'%(cell_types_unique[t],c_tr[t]))
    print('Validation set | Cell type %d: %d RGCs'%(cell_types_unique[t],c_val[t]))


# Data will be rolled so that each sample has a temporal width. Like N frames of movie in one sample. The duration of each frame is in t_frame
# if the model has a photoreceptor layer, then the PR layer has a termporal width of pr_temporal_width, which before convs will be chopped off to temporal width
# this is done to get rid of boundary effects. pr_temporal_width > temporal width
if mdl_name[:2] == 'PR':    # in this case the rolling width should be that of PR
    temporal_width_prepData = pr_temporal_width
    temporal_width_eval = pr_temporal_width
    
else:   # in all other cases its same as temporal width
    temporal_width_prepData = temporal_width
    temporal_width_eval = temporal_width    # termporal width of each sample. Like how many frames of movie in one sample
    pr_temporal_width = 0


    modelNames_all = models_jax.model_definitions()    # get all model names
    modelNames_2D = modelNames_all[0]
    modelNames_3D = modelNames_all[1]

    # Expand dataset if needed for vectorization, and roll it for temporal dimension
    d=0
    for d in range(n_tasks):
        print(fname_data_train_val_test_all[d])
        # data_train = dict_train[fname_data_train_val_test_all[d]]
        data_trtr = dict_trtr[fname_data_train_val_test_all[d]]
        # data_trval = dict_trval[fname_data_train_val_test_all[d]]

        # data_test = dict_test[fname_data_train_val_test_all[d]]
        data_val = dict_val[fname_data_train_val_test_all[d]]
        
        if mdl_name in modelNames_2D:
            if BUILD_MAPS==True:
                idx_unitsToTake_trtr = []
                idx_unitsToTake_trval = []
                idx_unitsToTake_val = []
            else:
                idx_unitsToTake_trtr = idx_unitsToTake_all_trtr[d]
                # idx_unitsToTake_trval = idx_unitsToTake_all_trval[d]
                idx_unitsToTake_val = idx_unitsToTake_all_val[d]

            data_trtr = prepare_data_cnn2d_maps(data_trtr,temporal_width_prepData,MAKE_LISTS=True,idx_unitsToTake=idx_unitsToTake_trtr)     # [samples,temporal_width,rows,columns]
            # data_trval = prepare_data_cnn2d_maps(data_trval,temporal_width_prepData,MAKE_LISTS=True,idx_unitsToTake=idx_unitsToTake_trval)     # [samples,temporal_width,rows,columns]
            data_val = prepare_data_cnn2d_maps(data_val,temporal_width_prepData,MAKE_LISTS=True,idx_unitsToTake=idx_unitsToTake_val)   
                            
            filt1_3rdDim=0
            filt2_3rdDim=0
            filt3_3rdDim=0
    
            
        else:
            raise ValueError('model not found')

        dict_trtr[fname_data_train_val_test_all[d]] = data_trtr
        # dict_trval[fname_data_train_val_test_all[d]] = data_trval

        # dict_test[fname_data_train_val_test_all[d]] = data_test
        dict_val[fname_data_train_val_test_all[d]] = data_val
   
    # Shuffle just the training dataset
    dict_trtr = dataloaders.shuffle_dataset(dict_trtr)    
    # dict_trval = dataloaders.shuffle_dataset(dict_trval)    
   
maxLen_umaskcoords_tr_subtr = max([len(value['umaskcoords_trtr_remap']) for value in dict_dinf.values()])
# maxLen_umaskcoords_tr_subval = max([len(value['umaskcoords_trval_remap']) for value in dict_dinf.values()])
maxLen_umaskcoords_val = max([len(value['umaskcoords_val']) for value in dict_dinf.values()])


# ---- PACK MASKS AND COORDINATES FOR EACH RGC
b_umaskcoords_trtr=-1*np.ones((maxLen_umaskcoords_tr_subtr,dinf['umaskcoords_trtr_remap'].shape[1]),dtype='int32');
# b_umaskcoords_trval=-1*np.ones((n_tasks,maxLen_umaskcoords_tr_subval,dinf['umaskcoords_trval_remap'].shape[1]),dtype='int32')
b_umaskcoords_val=-1*np.ones((maxLen_umaskcoords_val,dinf['umaskcoords_val'].shape[1]),dtype='int32')

bool_trtr=np.zeros((maxLen_umaskcoords_tr_subtr),dtype='int32')
# bool_trval=np.zeros((n_tasks,maxLen_umaskcoords_tr_subval),dtype='int32')
bool_val=np.zeros((maxLen_umaskcoords_val),dtype='int32')


maskunits_trtr = np.zeros((MAX_RGCS),dtype='int')
# maskunits_trval = np.zeros((n_tasks,MAX_RGCS),dtype='int')
maskunits_val = np.zeros((MAX_RGCS),dtype='int')

d=0
rgb = dict_dinf[fname_data_train_val_test_all[d]]['umaskcoords_trtr_remap']
b_umaskcoords_trtr[:len(rgb),:] = rgb
N_trtr = len(np.unique(rgb[:,0]))
maskunits_trtr[:N_trtr] = 1

rgb = dict_dinf[fname_data_train_val_test_all[d]]['umaskcoords_val']
b_umaskcoords_val[:len(rgb),:] = rgb
N_val = len(np.unique(rgb[:,0]))
maskunits_val[:N_val] = 1

segment_size = dict_dinf[fname_data_train_val_test_all[d]]['segment_size']
    
    
dinf_tr = dict(umaskcoords_trtr=b_umaskcoords_trtr,
                  # umaskcoords_trval=b_umaskcoords_trval,
                  N_trtr=N_trtr,
                  # N_trval=N_trval,
                  maskunits_trtr=maskunits_trtr,
                  # maskunits_trval=maskunits_trval,
                  cell_types_unique=cell_types_unique,
                  segment_size=segment_size,
                  MAX_RGCS=MAX_RGCS)

dinf_val = dict(umaskcoords_val=b_umaskcoords_val,
                  N_val=N_val,
                  maskunits_val=maskunits_val,
                  segment_size=segment_size,
                  MAX_RGCS=MAX_RGCS)


# PRINT INFO
nsamps_train = 0;dset_names=[]
for d in range(n_tasks):
    dset = fname_data_train_val_test_all[d]
    rgb = re.split('_',os.path.split(dset)[-1])[0]
    dset_names.append(rgb)
    nsamps_train = nsamps_train+len(dict_val[dset].X)+len(dict_val[dset].X)

run_info = dict(
                nsamps_train=nsamps_train,
                nsamps_unique_alldsets=nsamps_alldsets.sum(),
                nunits_train=c_tr.sum(),
                nunits_val=c_val.sum(),
                nexps = n_tasks,
                exps = expDates
                )
  
# ---- Dataloaders  
assert MAX_RGCS > c_exp_tr.sum(axis=1).max(), 'MAX_RGCS limit lower than maximum RGCs in a dataset'
# MAX_RGCS=int(c_tr.sum())

n_tasks = len(fname_data_train_val_test_all)    


assert n_tasks==1, 'Finetuning pipeline only takes 1 dataset'
batch_size_train = bz

dset = fname_data_train_val_test_all[0]
data_trtr = dict_trtr[dset]
RetinaDataset_train = dataloaders.RetinaDataset(data_trtr.X,data_trtr.y,transform=None)
dataloader_train = DataLoader(RetinaDataset_train,batch_size=batch_size_train,collate_fn=dataloaders.jnp_collate);

# batch = next(iter(dataloader_train));a,b,c,d=batch

data_val = dict_val[dset]
RetinaDataset_val = dataloaders.RetinaDataset(data_val.X,data_val.y,transform=None)
dataloader_val = DataLoader(RetinaDataset_val,batch_size=256,collate_fn=dataloaders.jnp_collate);
# batch = next(iter(dataloader_val));a,b=batch
  
n_batches = len(dataloader_train)#np.ceil(len(data_train.X)/bz)

# %% Load Pre-Trained Model 
"""
 There are three ways of selecting/building a model
 1. Continue training an existing model whose training was interrupted
 2. Build a new model
 3. Build a new model but transfer some or all weights (In this case the weight transferring layers should be similar)
"""

dict_params['filt_temporal_width'] = temporal_width
dict_params['nout'] = len(cell_types_unique)


allSteps = glob.glob(path_pretrained+'/step*')
assert  len(allSteps)!=0, 'No checkpoints found'

step_numbers = np.sort(np.asarray([int(re.search(r'step-(\d+)', s).group(1)) for s in allSteps]))
    


nb_cps = len(allSteps)
lastcp = os.path.split(allSteps[-1])[-1]

with open(os.path.join(path_pretrained,'model_architecture.pkl'), 'rb') as f:
    mdl,config = cloudpickle.load(f)
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

bias_allSteps=[]; weights_allSteps=[]
# for i in tqdm(range(nb_cps)):
#     fname_latestWeights = os.path.join(path_pretrained,'step-%03d' % step_numbers[i])
#     # print(fname_latestWeights)
#     raw_restored = orbax_checkpointer.restore(fname_latestWeights)
#     pre_mdl_state = train_singleretunits.load(mdl,raw_restored['model'],lr_pretrained)
    
#     weights = pre_mdl_state.params
#     output_bias = np.array(weights['output']['bias'])
#     bias_allSteps.append(output_bias.sum())
#     weights_allSteps.append(weights)

# if sum(np.isnan(bias_allSteps))>0:
#     last_cp = np.where(np.isnan(bias_allSteps))[0][0]-1     # Where the weights are not nan
# else:
#     last_cp = nb_cps-1
# last_cp = step_numbers[last_cp]
last_cp = step_numbers[nb_cps-1]

fname_latestWeights = os.path.join(path_pretrained,'step-%03d' % last_cp)

raw_restored = orbax_checkpointer.restore(fname_latestWeights)
pre_mdl_state = train_singleretunits.load(mdl,raw_restored['model'],lr_pretrained)

print('Loaded pre-trained model')

# %% Create FT Model
ft_mdl_name = mdl_name

pretrained_params = getModelParams(path_pretrained)
ft_fname_model,ft_model_params = model.utils_si.modelFileName(U=len(cell_types_unique),P=0,T=temporal_width,CB_n=0,
                                                    C1_n=pretrained_params['C1_n'],C1_s=pretrained_params['C1_s'],C1_3d=pretrained_params['C1_3d'],
                                                    C2_n=pretrained_params['C2_n'],C2_s=pretrained_params['C2_s'],C2_3d=pretrained_params['C2_3d'],
                                                    C3_n=pretrained_params['C3_n'],C3_s=pretrained_params['C3_s'],C3_3d=pretrained_params['C3_3d'],
                                                    C4_n=pretrained_params['C4_n'],C4_s=pretrained_params['C4_s'],C4_3d=pretrained_params['C4_3d'],
                                                    BN=pretrained_params['BN'],MP=pretrained_params['MP'],LR=pretrained_params['LR'],
                                                    TR=pretrained_params['TR'],TRSAMPS=pretrained_params['TR'])

ft_model_params['filt_temporal_width'] = temporal_width
ft_model_params['nout'] = len(cell_types_unique)        

ft_path_model_save = os.path.join(path_pretrained,'finetuning',expDates[0])   # the model save directory is the fname_model appened to save path
if os.path.exists(ft_path_model_save):
    shutil.rmtree(ft_path_model_save)  # Remove any existing checkpoints from the last notebook run.
os.makedirs(ft_path_model_save)

ft_path_save_model_performance = os.path.join(ft_path_model_save,'performance')
if not os.path.exists(ft_path_save_model_performance):
    os.makedirs(ft_path_save_model_performance)
    

training_params = dict(LOSS_FUN=LOSS_FUN)


fname_excel = 'performance_'+fname_model+'.csv'

inp_shape = data_val.X[0].shape

lr = 1e-3
nb_epochs = 10
transition_steps = int(n_batches*10)# 20000
# lr_schedule = optax.exponential_decay(init_value=lr,transition_steps=transition_steps,decay_rate=0.5,staircase=True,transition_begin=0,end_value=1e-8)
lr_schedule = optax.constant_schedule(lr)

total_steps = n_batches*nb_epochs
# rgb_lrs = [lr_schedule(i) for i in range(total_steps)]
# rgb_lrs = np.array(rgb_lrs)
# plt.plot(rgb_lrs);plt.show()
# print(np.array(rgb_lrs))


model_func = getattr(models_jax,ft_mdl_name)
ft_mdl = model_func
ft_mdl_state,ft_mdl,ft_config = train_metalmaps.initialize_model(ft_mdl,ft_model_params,inp_shape,lr,save_model=True,lr_schedule=lr_schedule)
models_jax.model_summary(ft_mdl,inp_shape,console_kwargs={'width':180})


# % Select layers to finetune

# print(pre_mdl_state.params.keys())
layers_finetune = [key for key in pre_mdl_state.params]
# layers_finetune = ['LayerNorm_4','output','LayerNorm_5','TrainableAF_4']
ft_params_fixed,ft_params_trainable = train_metalmaps.split_dict(pre_mdl_state.params,layers_finetune)

print('Fixed layers:');print(ft_params_fixed.keys())
print('Trainable layers:');print(ft_params_trainable.keys())

      

# Copy all weights from pre-trained model
for key in ft_mdl_state.params:
    ft_mdl_state.params[key] = pre_mdl_state.params[key]


optimizer = optax.adam(learning_rate=lr_schedule) #,weight_decay=1e-4)
ft_mdl_state = train_metalmaps.TrainState.create(
            apply_fn=ft_mdl.apply,
            params=ft_params_trainable,
            tx=optimizer)


# %% Fine-tune model

val_loss,pred_rate,y,pred_rate_units,y_units = train_metalmaps.ft_eval_step(pre_mdl_state,ft_params_fixed,dataloader_val,dinf_val)
val_loss = np.mean(val_loss)
fev, fracExVar, predCorr, rrCorr = model_evaluate_new(y_units,pred_rate_units,temporal_width_eval,lag=0,obs_noise=0)
print('Pre-trained perf | loss: %f, fev: %0.2f, Corr: %0.2f'%(np.mean(val_loss),np.median(fev),np.median(predCorr)))

cp_interval = n_batches
training_params['cp_interval'] = cp_interval

t_elapsed = 0
t = time.time()
initial_epoch=0
if initial_epoch < nb_epochs:
    print('-----RUNNING MODEL-----')
    
    loss_currEpoch_master,loss_epoch_train,loss_epoch_val,ft_mdl_state,fev_epoch_train,fev_epoch_val = train_metalmaps.ft_train(ft_mdl_state,ft_params_fixed,config,training_params,\
                                                                                  dataloader_train,dataloader_val,dinf_tr,dinf_val,nb_epochs,ft_path_model_save,save=True,\
                                                                                  lr_schedule=lr_schedule,step_start=initial_epoch+1)
    _ = gc.collect()
        
t_elapsed = time.time()-t
print('time elapsed: '+str(t_elapsed)+' seconds')


# %% Evaluate performance

#  Load checkpoint infos
allSteps = glob.glob(ft_path_model_save+'/step*')
assert  len(allSteps)!=0, 'No checkpoints found'

step_numbers = np.sort(np.asarray([int(re.search(r'step-(\d+)', s).group(1)) for s in allSteps]))
nb_cps = len(allSteps)
last_cp = step_numbers[nb_cps-1]

# Select the testing dataset
d=0

n_cells = dinf_val['N_val']
cps_sel = np.arange(0,len(step_numbers)).astype('int')
nb_cps_sel = len(cps_sel)

idx_dset=0
val_loss_allEpochs = np.empty(nb_cps_sel+1)
val_loss_allEpochs[:] = np.nan
fev_medianUnits_allEpochs = np.empty(nb_cps_sel+1)
fev_medianUnits_allEpochs[:] = np.nan
fev_allUnits_allEpochs = np.zeros((nb_cps_sel+1,n_cells))
fev_allUnits_allEpochs[:] = np.nan
fracExVar_medianUnits_allEpochs = np.empty(nb_cps_sel+1)
fracExVar_medianUnits_allEpochs[:] = np.nan
fracExVar_allUnits_allEpochs = np.zeros((nb_cps_sel+1,n_cells))
fracExVar_allUnits_allEpochs[:] = np.nan

predCorr_medianUnits_allEpochs = np.empty(nb_cps_sel+1)
predCorr_medianUnits_allEpochs[:] = np.nan
predCorr_allUnits_allEpochs = np.zeros((nb_cps_sel+1,n_cells))
predCorr_allUnits_allEpochs[:] = np.nan
rrCorr_medianUnits_allEpochs = np.empty(nb_cps_sel)
rrCorr_medianUnits_allEpochs[:] = np.nan
rrCorr_allUnits_allEpochs = np.zeros((nb_cps_sel+1,n_cells))
rrCorr_allUnits_allEpochs[:] = np.nan

data_val = dict_val[fname_data_train_val_test_all[idx_dset]]

if isintuple(data_val,'y_trials'):
    rgb = np.squeeze(np.asarray(data_val.y_trials))
    obs_noise = 0#estimate_noise(rgb)
    # obs_noise = estimate_noise(data_test.y_trials)
    obs_rate_allStimTrials = np.asarray(data_val.y)
    num_iters = 1
    
elif 'stim_0' in dataset_rr and dataset_rr['stim_0']['val'][:,:,idx_unitsToTake].shape[0]>1:
    obs_rate_allStimTrials = dataset_rr['stim_0']['val'][:,:,idx_unitsToTake]
    obs_noise = None
    num_iters = 10
else:
    obs_rate_allStimTrials = np.asarray(data_val.y)
    if 'var_noise' in data_quality:
        obs_noise = data_quality['var_noise'][idx_unitsToTake]
    else:
        obs_noise = 0
    num_iters = 1

if isintuple(data_val,'dset_names'):
    rgb = data_val.dset_names
    idx_natstim = [i for i,n in enumerate(rgb) if re.search(r'NATSTIM',n)]
    idx_cb = [i for i,n in enumerate(rgb) if re.search(r'CB',n)]
    
samps_shift = 0 # number of samples to shift the response by. This was to correct some timestamp error in gregs data
  
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

RetinaDataset_val = dataloaders.RetinaDataset(data_val.X,data_val.y,transform=None)
dataloader_val = DataLoader(RetinaDataset_val,batch_size=128,collate_fn=dataloaders.jnp_collate)


# %
print('-----EVALUATING PERFORMANCE-----')
i=nb_cps_sel-1
for i in tqdm(range(0,nb_cps_sel+1)): #[nb_cps_sel-1]: #tqdm(range(0,nb_cps_sel-1)):
    
    if i==0:    # That is randomly initialized
        mdl_state_eval = pre_mdl_state
    else:
        mdl_state_eval = ft_mdl_state
        weight_fold = 'step-%03d' %(step_numbers[cps_sel[i-1]])  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
        weight_fold = os.path.join(ft_path_model_save,weight_fold)
    
        assert os.path.isdir(weight_fold)==True, 'Checkpoint %d does not exist'%step_numbers[cps_sel[i-1]]
        raw_restored = orbax_checkpointer.restore(weight_fold)
        mdl_state_eval = train_metalmaps.load(mdl,raw_restored['model'],lr)
            
    val_loss,pred_rate,y,pred_rate_units,y_units = train_metalmaps.ft_eval_step(mdl_state_eval,ft_params_fixed,dataloader_val,dinf_val)
    val_loss = np.mean(val_loss)

    val_loss_allEpochs[i] = val_loss
    

    fev, fracExVar, predCorr, rrCorr = model_evaluate_new(y_units,pred_rate_units,temporal_width_eval,lag=int(samps_shift),obs_noise=0)
            
    fev_allUnits_allEpochs[i,:] = fev
    fev_medianUnits_allEpochs[i] = np.nanmedian(fev)      
    # fracExVar_allUnits_allEpochs[i,:] = fracExVar
    # fracExVar_medianUnits_allEpochs[i] = np.nanmedian(fracExVar)
    
    predCorr_allUnits_allEpochs[i,:] = predCorr
    predCorr_medianUnits_allEpochs[i] = np.nanmedian(predCorr)
    # rrCorr_allUnits_allEpochs[i,:] = rrCorr
    # rrCorr_medianUnits_allEpochs[i] = np.nanmedian(rrCorr)
    
    _ = gc.collect()
    
fig,axs = plt.subplots(1,2,figsize=(14,5)); fig.suptitle(dset_names[idx_dset])
axs[0].plot(step_numbers[cps_sel],predCorr_medianUnits_allEpochs[:len(cps_sel)])
axs[0].set_xlabel('Training steps');axs[0].set_ylabel('Corr'); 
axs[1].plot(step_numbers[cps_sel],fev_medianUnits_allEpochs[:len(cps_sel)])
axs[1].set_xlabel('Training steps');axs[1].set_ylabel('FEV'); 

# %%

u=33;plt.plot(y_units[:500,u]);plt.plot(pred_rate_units[:500,u])


# %% Parameter changes

def compute_relative_changes(original_params, finetuned_params):
    param_changes = jax.tree_map(lambda a,b: jnp.linalg.norm(b-a)/jnp.linalg.norm(a), params_orig, params_final)
    
    return param_changes

def get_cpt_mdl(mdl_state,cpt=None):
    if cpt==None:
        allSteps = glob.glob(ft_path_model_save+'/step*')
        assert  len(allSteps)!=0, 'No checkpoints found'
        step_numbers = np.sort(np.asarray([int(re.search(r'step-(\d+)', s).group(1)) for s in allSteps]))
        nb_cps = len(allSteps)
        last_cp = step_numbers[nb_cps-1]
    else:
        last_cp = cpt
    
    mdl_state_eval = ft_mdl_state
    weight_fold = 'step-%03d' %(last_cp)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
    weight_fold = os.path.join(ft_path_model_save,weight_fold)
    
    assert os.path.isdir(weight_fold)==True, 'Checkpoint %d does not exist'%step_numbers[cps_sel[i-1]]
    raw_restored = orbax_checkpointer.restore(weight_fold)
    mdl_state_eval = train_metalmaps.load(ft_mdl,raw_restored['model'],lr)
    
    return mdl_state_eval

idx_bestcp = np.argmax(predCorr_medianUnits_allEpochs)
mdl_state_eval = get_cpt_mdl(ft_mdl_state,step_numbers[idx_bestcp])
params_final = mdl_state_eval.params
params_orig = pre_mdl_state.params


param_changes = compute_relative_changes(params_orig, params_final)

layer_order = ['Conv_0','LayerNorm_0','TrainableAF_0','Conv_1','LayerNorm_1','TrainableAF_1','Conv_2','LayerNorm_2','TrainableAF_2',
                   'Conv_3','LayerNorm_3','TrainableAF_3',
                   'LayerNorm_4','output','LayerNorm_5','TrainableAF_4']

# %%
def get_flattened_changes(param_changes,layer_order,param_name):
    flattened_changes=[]
    for layer_name in layer_order:
        for param in param_name:
            change = param_changes[layer_name][param]
            rgb = (f'{layer_name}/{param}', change)
            flattened_changes.append(rgb)
    param_names = [change[0] for change in flattened_changes]
    changes = [change[1] for change in flattened_changes]

    return param_names,changes

layer_order = ['TrainableAF_0','TrainableAF_1','TrainableAF_2','TrainableAF_3','TrainableAF_4']
pname_AF_sat,pval_AF_sat = get_flattened_changes(param_changes,layer_order,['sat',])
pname_AF_gain,pval_AF_gain = get_flattened_changes(param_changes,layer_order,['gain',])

layer_order = ['LayerNorm_0','LayerNorm_1','LayerNorm_2','LayerNorm_3','LayerNorm_4','LayerNorm_5']
pname_LN_scale,pval_LN_scale = get_flattened_changes(param_changes,layer_order,['scale',])
pname_LN_bias,pval_LN_bias = get_flattened_changes(param_changes,layer_order,['bias',])


layer_order = ['LayerNorm_0','LayerNorm_1','LayerNorm_2','LayerNorm_3','LayerNorm_4','LayerNorm_5']
param_name = ['bias',]

layer_order = ['Conv_0','Conv_1','Conv_2','Conv_3','output']
pname_conv_kernel,pval_conv_kernel = get_flattened_changes(param_changes,layer_order,['kernel',])
pname_conv_bias,pval_conv_bias = get_flattened_changes(param_changes,layer_order,['bias',])


fig,axs=plt.subplots(3,2,figsize=(12,10))
axs[0,0].plot(pval_conv_kernel,'-o')
axs[0,0].set_ylabel('relative change')
axs[0,0].set_title('Conv kernel')
axs[0,1].plot(pval_conv_bias,'-o')
axs[0,1].set_ylabel('relative change')
axs[0,1].set_title('Conv bias')


axs[1,0].plot(pval_LN_scale,'-o')
axs[1,0].set_ylabel('relative change')
axs[1,0].set_title('LayerNorm Scale')
axs[1,1].plot(pval_LN_bias,'-o')
axs[1,1].set_ylabel('relative change')
axs[1,1].set_title('LayerNorm Bias')


axs[2,0].plot(pval_AF_sat,'-o')
axs[2,0].set_ylabel('relative change')
axs[2,0].set_title('ActivationFunction Sat')
axs[2,1].plot(pval_AF_gain,'-o')
axs[2,1].set_ylabel('relative change')
axs[2,1].set_title('ActivationFunction Gain')


# %%
flattened_changes = [(f'{layer_name}/{p}', change) 
                     for layer_name in layer_order
                     for p, change in param_changes[layer_name]]


sorted_changes = sorted(flattened_changes, key=lambda x: x[1], reverse=True)
print(sorted_changes[:10])

param_names = [change[0] for change in sorted_changes]
param_changes = [change[1] for change in sorted_changes]

# Create a bar chart
N_top = len(param_names)
plt.figure(figsize=(10, 6))
plt.barh(param_names[:N_top], param_changes[:N_top])  # Display top 10 parameters
plt.xlabel('Relative Change')
plt.ylabel('Parameter')
plt.title('Parameter Changes After Fine-tuning')
plt.gca().invert_yaxis()  # To show the largest changes at the top
plt.show()




# %% Plot AF model

def TrainableAF(x,sat=0.01,gain=0.95):
    
    a = ((1-sat+1e-6)*jnp.log(1+jnp.exp(gain*x)))/(gain+1e-6)
    b = (sat*(jnp.exp(gain*x)))/(1+jnp.exp(gain*x)+1e-6)
    
    outputs = a+b
    return outputs

x = np.arange(-5,5,0.1)


chan=120
sel_layer = 'TrainableAF_4'
sat = params_orig[sel_layer]['sat'][chan]
gain = params_orig[sel_layer]['gain'][chan]
outputs_orig = TrainableAF(x,sat,gain)

sat = params_final[sel_layer]['sat'][chan]
gain = params_final[sel_layer]['gain'][chan]
outputs_final = TrainableAF(x,sat,gain)


plt.plot(x,outputs_orig,label='initial');plt.plot(x,outputs_final,label='final');
plt.legend();plt.title('Layer: %s | Chan: %d'%(sel_layer,chan))