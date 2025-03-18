#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:25:17 2025

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

from model.data_handler import prepare_data_cnn2d_maps,isintuple
from model import data_handler_ej
from model.data_handler_mike import load_h5Dataset
from model.performance import model_evaluate_new, estimate_noise
import model.paramsLogger
import model.utils_si

from model import models_jax, train_singleretunits, dataloaders,handler_maps
from model import train_metalmaps
from torch.utils.data import DataLoader

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
chan1_n=64; filt1_size=5
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

# expDates = expDates[-3:]


# %% load train val and test datasets from saved h5 file
"""
    load_h5dataset is a function to load training and validation data from h5 dataset. We can extract all data or a subset using the nsamps arguments.
    data_train, val and test are named tuples. data_train.X contains the stimulus with dimensions [samples,y pixels, x pixels]
    and data_train.y contains the spikerate normalized by median [samples,numOfCells]
"""
BUILD_MAPS = False
MAX_RGCS = model.train_metalmaps.MAX_RGCS


trainingSamps_dur = 0.5
    
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
d=17
dict_train = {};dict_trtr={};dict_trval={}
dict_val = {}
dict_test = {}
unames_allDsets = []
data_info = {}
dict_dinf = {}
nsamps_alldsets_loaded = []
for d in range(len(fname_data_train_val_test_all)):
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
    info_unitSplit['unames_val'] = parameters['unames'][info_unitSplit['idx_val']]


    t_frame = parameters['t_frame']     # time in ms of one frame/sample 

    frac_train_units = parameters['frac_train_units']

    data_train,data_val,dinf = handler_maps.arrange_data_formaps(exp,data_train,data_val,parameters,frac_train_units,psf_params=psf_params,info_unitSplit=None,
                                                                 BUILD_MAPS=BUILD_MAPS,MODE='validation')
    dinf['unit_locs_train'] = dinf['unit_locs'][dinf['idx_units_train']]
    dinf['unit_types_train'] = dinf['unit_types'][dinf['idx_units_train']]
    
    dinf['unit_locs_val'] = dinf['unit_locs'][dinf['idx_units_val']]
    dinf['unit_types_val'] = dinf['unit_types'][dinf['idx_units_val']]
    
    dict_val[fname_data_train_val_test_all[d]] = data_val
    dict_dinf[fname_data_train_val_test_all[d]] = dinf
    unames_allDsets.append(parameters['unames'])
    nsamps_alldsets_loaded.append(len(data_val.X))
    

cell_types_unique = np.unique(dinf['umaskcoords_train'][:,1])

nrgcs_val=[];
for d in range(len(fname_data_train_val_test_all)):
    nrgcs_val.append(dict_val[fname_data_train_val_test_all[d]].y.shape[-1])


# %
if BUILD_MAPS==False:   
    idx_unitsToTake_all_val = [];
    mask_unitsToTake_all_val = [];
    d=0
    for d in range(len(fname_data_train_val_test_all)):

        idx_unitsToTake,mask_unitsToTake = handler_maps.get_expandedRGClist(dict_val[fname_data_train_val_test_all[d]],MAX_RGCS)
        idx_unitsToTake_all_val.append(idx_unitsToTake)
        mask_unitsToTake_all_val.append(mask_unitsToTake)


# Get unit names
uname_val = [];
c_tr = np.zeros(len(cell_types_unique),dtype='int'); c_val = np.zeros(len(cell_types_unique),dtype='int');
c_exp_tr=np.zeros((len(expDates),len(cell_types_unique)),dtype='int');c_exp_val=np.zeros((len(expDates),len(cell_types_unique)),dtype='int');

d=1
for d in range(len(fname_data_train_val_test_all)):
    dinf = dict_dinf[fname_data_train_val_test_all[d]]
    for t in range(len(cell_types_unique)):
        c_tr[t]=c_tr[t]+(dinf['unit_types'][dinf['idx_units_train']]==cell_types_unique[t]).sum()
        c_exp_tr[d,t] = (dinf['unit_types'][dinf['idx_units_train']]==cell_types_unique[t]).sum()
    
    rgb = dinf['unames'][dinf['idx_units_val']]
    exp_uname = [expDates[d]+'_'+u for u in rgb ]
    uname_val.append(exp_uname)
    
    for t in range(len(cell_types_unique)):
        c_val[t]=c_val[t]+(dinf['unit_types'][dinf['idx_units_val']]==cell_types_unique[t]).sum()
        c_exp_val[d,t] = (dinf['unit_types'][dinf['idx_units_val']]==cell_types_unique[t]).sum()

print('Total number of datasets: %d'%len(fname_data_train_val_test_all))
for t in range(len(cell_types_unique)):
    print('Validation set | Cell type %d: %d RGCs'%(cell_types_unique[t],c_val[t]))


# Data will be rolled so that each sample has a temporal width. Like N frames of movie in one sample. The duration of each frame is in t_frame
# if the model has a photoreceptor layer, then the PR layer has a termporal width of pr_temporal_width, which before convs will be chopped off to temporal width
# this is done to get rid of boundary effects. pr_temporal_width > temporal width
temporal_width_prepData = temporal_width
temporal_width_eval = temporal_width    # termporal width of each sample. Like how many frames of movie in one sample
pr_temporal_width = 0


modelNames_all = models_jax.model_definitions()    # get all model names
modelNames_2D = modelNames_all[0]
modelNames_3D = modelNames_all[1]

# Expand dataset if needed for vectorization, and roll it for temporal dimension
d=0
for d in range(len(fname_data_train_val_test_all)):
    print(fname_data_train_val_test_all[d])
    data_val = dict_val[fname_data_train_val_test_all[d]]
    
    if mdl_name in modelNames_2D:

        data_val = prepare_data_cnn2d_maps(data_val,temporal_width_prepData,MAKE_LISTS=True,idx_unitsToTake=idx_unitsToTake_all_val[d])   
        
        filt1_3rdDim=0
        filt2_3rdDim=0
        filt3_3rdDim=0

        
    else:
        raise ValueError('model not found')

    dict_val[fname_data_train_val_test_all[d]] = data_val
   


# %
maxLen_umaskcoords_val = max([len(value['umaskcoords_val']) for value in dict_dinf.values()])

n_tasks = len(fname_data_train_val_test_all)
# ---- PACK MASKS AND COORDINATES FOR} EACH RGC
b_umaskcoords_val=-1*np.ones((n_tasks,maxLen_umaskcoords_val,dinf['umaskcoords_val'].shape[1]),dtype='int32')
bool_val=np.zeros((n_tasks,maxLen_umaskcoords_val),dtype='int32')
N_val = np.zeros((n_tasks),dtype='int')
maskunits_val = np.zeros((n_tasks,MAX_RGCS),dtype='int')

d=0
for d in range(len(fname_data_train_val_test_all)):
    rgb = dict_dinf[fname_data_train_val_test_all[d]]['umaskcoords_val']
    b_umaskcoords_val[d,:len(rgb),:] = rgb
    N_val[d] = len(np.unique(rgb[:,0]))
    maskunits_val[d,:N_val[d]] = 1

    segment_size = dict_dinf[fname_data_train_val_test_all[d]]['segment_size']
    
    
dinf_val = dict(umaskcoords_val=b_umaskcoords_val,
                  N_val=N_val,
                  maskunits_val=maskunits_val,
                  segment_size=segment_size,
                  MAX_RGCS=MAX_RGCS)

                

# PRINT INFO
nsamps_train = 0;dset_names=[]
for d in range(len(fname_data_train_val_test_all)):
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
    
# %% Select model 
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
for i in tqdm(range(nb_cps)):
    fname_latestWeights = os.path.join(path_pretrained,'step-%03d' % step_numbers[i])
    # print(fname_latestWeights)
    raw_restored = orbax_checkpointer.restore(fname_latestWeights)
    mdl_state = train_singleretunits.load(mdl,raw_restored['model'],lr_pretrained)
    
    weights = mdl_state.params
    output_bias = np.array(weights['output']['bias'])
    bias_allSteps.append(output_bias.sum())
    weights_allSteps.append(weights)

if sum(np.isnan(bias_allSteps))>0:
    last_cp = np.where(np.isnan(bias_allSteps))[0][0]-1     # Where the weights are not nan
else:
    last_cp = nb_cps-1
last_cp = step_numbers[last_cp]

fname_latestWeights = os.path.join(path_pretrained,'step-%03d' % last_cp)

raw_restored = orbax_checkpointer.restore(fname_latestWeights)
mdl_state = train_singleretunits.load(mdl,raw_restored['model'],lr_pretrained)

# Also load the dense layer weights
weights_output_file = os.path.join(path_pretrained,fname_latestWeights,'weights_output.h5')

with h5py.File(weights_output_file,'r') as f:
    kern_all = jnp.array(f['weights_output_kernel'])
    bias_all = jnp.array(f['weights_output_bias'])
    
weights_output = (kern_all,bias_all)

print('Loaded existing model')

path_save_model_performance = os.path.join(path_pretrained,'performance')
if not os.path.exists(path_save_model_performance):
    os.makedirs(path_save_model_performance)
            

fname_excel = 'performance_'+fname_model+'.csv'

inp_shape = data_val.X[0].shape
models_jax.model_summary(mdl,inp_shape,console_kwargs={'width':180})


# %% All Retinas Last Epoch

Nunits_total = dinf_val['N_val'].sum()

fev_allUnits_allExps = np.zeros((n_tasks,Nunits_total)); fev_allUnits_allExps[:] = np.nan
predCorr_allUnits_allExps = np.zeros((n_tasks,Nunits_total)); predCorr_allUnits_allExps[:] = np.nan
nunits_allExps = []
d=3
for d in range(n_tasks):
    idx_dset = d
    dinf_batch_val = jax.tree_map(lambda x: x[idx_dset] if isinstance(x, np.ndarray) else x, dinf_val)
    
    n_cells = dinf_val['N_val'][idx_dset]
    
    data_val = dict_val[fname_data_train_val_test_all[idx_dset]]
    
    if isintuple(data_val,'y_trials'):
        rgb = np.squeeze(np.asarray(data_val.y_trials))
        obs_noise = 0#estimate_noise(rgb)
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
      
    RetinaDataset_val = dataloaders.RetinaDataset(data_val.X,data_val.y,transform=None)
    dataloader_val = DataLoader(RetinaDataset_val,batch_size=512,collate_fn=dataloaders.jnp_collate)
    
    # %
    print('-----EVALUATING PERFORMANCE-----')
    
    if last_cp==0:    # That is randomly initialized
        model_func = getattr(models_jax,mdl_name)
        mdl = model_func
        mdl_state,mdl,config = model.train_metalmaps.initialize_model(mdl,dict_params,inp_shape,lr_pretrained,save_model=False,lr_schedule=None)
    else:
    
        weight_fold = 'step-%03d' %(last_cp)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
        weight_fold = os.path.join(path_pretrained,weight_fold)
        weights_output_file = os.path.join(path_pretrained,weight_fold,'weights_output.h5')
    
        if os.path.isdir(weight_fold):
            raw_restored = orbax_checkpointer.restore(weight_fold)
            mdl_state = train_metalmaps.load(mdl,raw_restored['model'],lr_pretrained)
            
            with h5py.File(weights_output_file,'r') as f:
                weights_kern = jnp.array(f['weights_output_kernel'][idx_dset])
                weights_bias = jnp.array(f['weights_output_bias'][idx_dset])
                
            # # Restore the correct dense weights for this dataset
            # mdl_state.params['output']['kernel'] = weights_kern
            # mdl_state.params['output']['bias'] = weights_bias
    
    val_loss,pred_rate,y,pred_rate_units,y_units = train_metalmaps.eval_step_dl(mdl_state,dataloader_val,dinf_batch_val)
    val_loss = np.mean(val_loss)
    
    fev, fracExVar, predCorr, rrCorr = model_evaluate_new(y_units,pred_rate_units,temporal_width_eval,lag=int(samps_shift),obs_noise=0)
            
    fev_allUnits_allExps[d,:len(fev)] = fev
    predCorr_allUnits_allExps[d,:len(predCorr)] = predCorr
    nunits_allExps.append(len(fev))
    
    # _ = gc.collect()

fev_medUnits_allExps = np.nanmedian(fev_allUnits_allExps,axis=-1)
predCorr_medUnits_allExps = np.nanmedian(predCorr_allUnits_allExps,axis=-1)


col_scheme = np.ones((3,n_tasks))*.5
lim_y = [-0.1,0.8]
lim_x = [-1,n_tasks+1]
fig,ax = plt.subplots(1,1,figsize=(10,5));fig.suptitle(testList + ' | '+APPROACH);ax=np.ravel(ax)
snb.boxplot(predCorr_allUnits_allExps.T,ax=ax[0],palette=col_scheme.T)
ax[0].plot([-5,n_tasks+5],[np.median(predCorr_medUnits_allExps),np.median(predCorr_medUnits_allExps)],'--r')
ax[0].set_ylim(lim_y);ax[0].set_ylabel('Corr')
ax[0].set_xlim(lim_x);ax[0].set_xlabel('Retina #')
ax[0].text(-.5,0.7,'N=%d RGCs'%Nunits_total)


fname_fig = os.path.join(path_save_model_performance,APPROACH+'_perf_lastEpoch_%s.png'%testList)
fig.savefig(fname_fig)

performance_lastEpoch = {
    'dset_names':dset_names,
       
    'fev_medUnits_allExps': fev_medUnits_allExps,
    'fev_allUnits_allExps': fev_allUnits_allExps,
    
    'predCorr_medUnits_allExps': predCorr_medUnits_allExps,
    'predCorr_allUnits_allExps': predCorr_allUnits_allExps,

    'weight_fold': weight_fold,
    'last_cp': last_cp,
    'nunits_allExps': nunits_allExps
    }
    

metaInfo = {
   'mdl_name': mdl_name,
   'path_model': path_pretrained,
   'uname_selectedUnits': unames_allDsets,#[idx_unitsToTake],dtype='bytes'),
   'thresh_rr': 0,
   'trial_num': 1,
   'Date': np.array(datetime.datetime.now(),dtype='bytes'),
   }
    

# fname_save_performance = os.path.join(path_save_model_performance,APPROACH+'_perf_allExps_lastEpoch.pkl')
# fname_save_performance = os.path.join(path_save_model_performance,APPROACH+'_perf_testExps_lastEpoch.pkl')
# fname_save_performance = os.path.join(path_save_model_performance,APPROACH+'_perf_trainingExps_lastEpoch.pkl')

# # fname_save_performance = os.path.join(path_save_model_performance,APPROACH+'_perf_trainingExps_valUnits_lastEpoch.pkl')

# with open(fname_save_performance, 'wb') as f:       # Save model architecture
#     cloudpickle.dump([performance_lastEpoch,metaInfo], f)

# %% Single retina all epochs

# Select the testing dataset
d=15

idx_dset = d
dinf_batch_val = jax.tree_map(lambda x: x[idx_dset] if isinstance(x, np.ndarray) else x, dinf_val)

n_cells = dinf_val['N_val'][idx_dset]

cps_sel = np.arange(0,len(step_numbers)).astype('int')
# cps_sel = np.arange(0,100).astype('int')

nb_cps_sel = len(cps_sel)


val_loss_allEpochs = np.empty(nb_cps_sel)
val_loss_allEpochs[:] = np.nan
fev_medianUnits_allEpochs = np.empty(nb_cps_sel)
fev_medianUnits_allEpochs[:] = np.nan
fev_allUnits_allEpochs = np.zeros((nb_cps_sel,n_cells))
fev_allUnits_allEpochs[:] = np.nan
fracExVar_medianUnits_allEpochs = np.empty(nb_cps_sel)
fracExVar_medianUnits_allEpochs[:] = np.nan
fracExVar_allUnits_allEpochs = np.zeros((nb_cps_sel,n_cells))
fracExVar_allUnits_allEpochs[:] = np.nan

predCorr_medianUnits_allEpochs = np.empty(nb_cps_sel)
predCorr_medianUnits_allEpochs[:] = np.nan
predCorr_allUnits_allEpochs = np.zeros((nb_cps_sel,n_cells))
predCorr_allUnits_allEpochs[:] = np.nan
rrCorr_medianUnits_allEpochs = np.empty(nb_cps_sel)
rrCorr_medianUnits_allEpochs[:] = np.nan
rrCorr_allUnits_allEpochs = np.zeros((nb_cps_sel,n_cells))
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
    
# if obs_noise.shape[0] == mask_unitsToTake_all[idx_dset].shape[0]:
# obs_noise = obs_noise[mask_unitsToTake_all[idx_dset]==1]

samps_shift = 0 # number of samples to shift the response by. This was to correct some timestamp error in gregs data
  
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

RetinaDataset_val = dataloaders.RetinaDataset(data_val.X,data_val.y,transform=None)
dataloader_val = DataLoader(RetinaDataset_val,batch_size=512,collate_fn=dataloaders.jnp_collate)

# mdl_state,mdl,config = model.jax.train_singleretunits.initialize_model(mdl,dict_params,inp_shape,lr,save_model=False)


# %
print('-----EVALUATING PERFORMANCE-----')
i=49
for i in [nb_cps_sel-1]: #[nb_cps_sel-1]: #tqdm(range(0,nb_cps_sel-1)):
    # print('evaluating cp %d of %d'%(i,nb_cps))
    # weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % (i+1)
    
    if i==0:    # That is randomly initialized
        model_func = getattr(models_jax,mdl_name)
        mdl = model_func
        mdl_state,mdl,config = model.train_metalmaps.initialize_model(mdl,dict_params,inp_shape,lr_pretrained,save_model=False,lr_schedule=None)
    else:

        weight_fold = 'step-%03d' %(step_numbers[cps_sel[i-1]])  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
        weight_fold = os.path.join(path_pretrained,weight_fold)
        weights_output_file = os.path.join(path_pretrained,weight_fold,'weights_output.h5')
    
        assert os.path.isdir(weight_fold)==True, 'Checkpoint %d does not exist'%step_numbers[cps_sel[i-1]]
        raw_restored = orbax_checkpointer.restore(weight_fold)
        mdl_state = train_metalmaps.load(mdl,raw_restored['model'],lr_pretrained)
            
            # with h5py.File(weights_output_file,'r') as f:
            #     weights_kern = jnp.array(f['weights_output_kernel'][idx_dset])
            #     weights_bias = jnp.array(f['weights_output_bias'][idx_dset])
                
            # # Restore the correct dense weights for this dataset
            # mdl_state.params['output']['kernel'] = weights_kern
            # mdl_state.params['output']['bias'] = weights_bias

    val_loss,pred_rate,y,pred_rate_units,y_units = train_metalmaps.eval_step_dl(mdl_state,dataloader_val,dinf_batch_val)
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
axs[1].plot(step_numbers[cps_sel],val_loss_allEpochs[:len(cps_sel)])
axs[1].set_xlabel('Training steps');axs[1].set_ylabel('Validation loss'); 

# %%

u=5;plt.plot(y_units[:500,u]);plt.plot(pred_rate_units[:500,u])


# %%

x = pred_rate_units.flatten()
y = y_units.flatten()
A = np.vstack([x,np.ones(len(x))]).T

params = np.linalg.lstsq(A,y,rcond=None)[0]
a,b = params

y_transormed = a*x + b
y_trans = y_transormed.reshape(y_units.shape)

u=0;plt.plot(y_units[:500,u]);plt.plot(pred_rate_units[:500,u]);plt.plot(-.5+y_trans[:500,u]*3.5)
u=0;plt.plot(y_units[:500,u]);plt.plot(-.5+y_trans[:500,u]*3.5)

pred_rate_trans[:,u] = y_transormed
