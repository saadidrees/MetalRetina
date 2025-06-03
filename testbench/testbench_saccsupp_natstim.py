#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:11:40 2025

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

# %% 
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

from model.data_handler import prepare_data_cnn2d_maps,isintuple
from model import data_handler_ej
from model.data_handler_mike import load_h5Dataset
from model.performance import model_evaluate_new, estimate_noise
import model.paramsLogger
import model.utils_si

from model import models_jax, train_singleretunits, dataloaders,handler_maps,prfr_params
from model import train_metalmaps
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


import seaborn as snb

Exptdata = namedtuple('Exptdata', ['X', 'y'])
Exptdata_spikes = namedtuple('Exptdata', ['X', 'y','spikes'])

# %

list_suffix = '20241115RstarM' #'8M'#'20241115M'   20241115RstarM
testList = 'natstim_sacc_supp'
trainList = 'trainlist_'+list_suffix

path_dataset_base = '/home/saad/data/Dropbox/postdoc/analyses/data_ej/'

with open(os.path.join(path_dataset_base,'datasets',trainList+'.txt'), 'r') as f:
    expDates_train = f.readlines()
expDates_train = [line.strip() for line in expDates_train]
dataset_suffix = expDates_train[0] #'CB_mesopic_f4_8ms_sig-4_MAPS'
expDates_train=expDates_train[1:]

path_base = '/home/saad/data/analyses/data_ej/models/cluster2/'
path_base_single =  '/home/saad/data/analyses/data_ej/models/cluster/'

# dataset_suffix = 'CB_mesopic_f4_8ms_sig-4_MAPS'
validationSamps_dur = 0.5

# Pretrained model params
APPROACH='metalzero'
mdl_name = 'PRFR_CNN2D_MAP' #'CNN2D_MAP3'  PRFR_CNN2D_MAP
pr_params_name = 'fr_cones_gammalarge'

LOSS_FUN = 'mad'
U = len(expDates_train)#32#32
lr_pretrained = 0.0001
P = 100
temporal_width=80
chan1_n=32; filt1_size=7
chan2_n=64; filt2_size=7
chan3_n=128; filt3_size=7
chan4_n=0; filt4_size=0
MaxPool=1
trainingSamps_dur = 15 #1#20 #-1 #0.05 # minutes per dataset


fname_model,dict_params = model.utils_si.modelFileName(U=U,P=P,T=temporal_width,CB_n=0,
                                                    C1_n=chan1_n,C1_s=filt1_size,C1_3d=0,
                                                    C2_n=chan2_n,C2_s=filt2_size,C2_3d=0,
                                                    C3_n=chan3_n,C3_s=filt3_size,C3_3d=0,
                                                    C4_n=chan4_n,C4_s=filt4_size,C4_3d=0,
                                                    BN=1,MP=MaxPool,LR=lr_pretrained,TR=1,TRSAMPS=trainingSamps_dur)


path_pretrained = os.path.join(path_base,APPROACH,trainList,LOSS_FUN,mdl_name,pr_params_name,fname_model+'/')
fname_pretrained = os.path.split(path_pretrained[:-1])[-1]
rgb = model.models_jax.getModelParams(fname_pretrained)
dict_params = dict_params | rgb



assert os.path.exists(path_pretrained), 'Model does not exist'

# expDates = expDates[-3:]
ft_mdl_name = 'CNN2D_MAP3_FT' #  PRFR_CNN2D_MAP_FT   CNN2D_MAP3_FT

lr_pretrained = dict_params['LR']
temporal_width=dict_params['T']
if 'PR' in mdl_name:
    pr_temporal_width=dict_params['P']
    temporal_width_prep = pr_temporal_width
else:
    pr_temporal_width=0
    temporal_width = dict_params['T']
    temporal_width_prep = temporal_width




# %% Load Model

# dict_params = model.models_jax.getModelParams(fname_pretrained)

lr_pretrained = dict_params['LR']
# temporal_width=dict_params['T']


assert os.path.exists(path_pretrained), 'Model does not exist'

dict_params['filt_temporal_width'] = temporal_width
dict_params['nout'] = 2

allCps = glob.glob(path_pretrained+'/step*')
assert  len(allCps)!=0, 'No checkpoints found'

allCps = [cp for cp in allCps if ".orbax-checkpoint" not in cp]

step_numbers = np.sort(np.asarray([int(re.search(r'step-(\d+)', s).group(1)) for s in allCps]))
    
nb_cps = len(allCps)
lastcp = os.path.split(allCps[-1])[-1]

with open(os.path.join(path_pretrained,'model_architecture.pkl'), 'rb') as f:
    mdl,config = cloudpickle.load(f)
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


bias_allSteps=[]; weights_allSteps=[]
for i in range(nb_cps):
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
nb_cps = last_cp

# for C in range(len(step_numbers)):
last_cp =  step_numbers[7] #

fname_latestWeights = os.path.join(path_pretrained,'step-%03d' % last_cp)

raw_restored = orbax_checkpointer.restore(fname_latestWeights)
mdl_state = train_singleretunits.load(mdl,raw_restored['model'],lr_pretrained)

inp_shape = (temporal_width_prep,40,80)
lr_schedule = optax.constant_schedule(lr_pretrained)

ft_pr_params_name = pr_params_name  #'fr_rods_fixed'
if 'PR' in mdl_name:
    pr_params_fun = getattr(model.prfr_params,ft_pr_params_name)
    ft_pr_params = pr_params_fun()
    # ft_pr_params['sigma_trainable']=True
    # ft_pr_params['phi_trainable']=True
    # ft_pr_params['eta_trainable']=True
    # ft_pr_params['beta_trainable']=True
    # ft_pr_params['gamma_trainable']=True
    ft_pr_params['gamma']=0.01
    # ft_pr_params['sigma']=0.707#0.707
    # ft_pr_params['phi']=0.707#0.707
    # ft_pr_params['eta']=0.00253*10
    ft_pr_params['beta']= 1#2.5

    dict_params['pr_params'] = ft_pr_params
    model_func = getattr(models_jax,mdl_name) #ft_mdl_name #CNN2D_MAP3_FT

    ft_mdl = model_func
    ft_mdl_state,ft_mdl,ft_config = train_metalmaps.initialize_model(ft_mdl,dict_params,inp_shape,lr_pretrained,save_model=False,lr_schedule=lr_schedule)
    # models_jax.model_summary(ft_mdl,inp_shape,console_kwargs={'width':180})
    layers_finetune = []#'LayerNorm_0']#,'output','LayerNorm_4','TrainableAF_3']
    ft_params_fixed,ft_params_trainable = train_metalmaps.split_dict(mdl_state.params,[])   
    ft_mdl_state = ft_mdl_state.replace(params=mdl_state.params)
    mdl_state = ft_mdl_state
    
# models_jax.model_summary(mdl,inp_shape,console_kwargs={'width':180})



# %   
# models_jax.model_summary(mdl,inp_shape,console_kwargs={'width':180})

path_save_model_performance = os.path.join(path_pretrained,'testbench_saccsupp')
if not os.path.exists(path_save_model_performance):
    os.makedirs(path_save_model_performance)

# %% Load NATSTIM MOVS
def normalize_stim(data_t,data_v,scale_lims):
    stim_frames1 = data_t.X
    stim_frames2 = data_v.X

    if len(scale_lims)>0:
        min_val = np.min(stim_frames1,axis=(0,1,2),keepdims=True)
        max_val = np.max(stim_frames1,axis=(0,1,2),keepdims=True)
        stim_frames1 = (stim_frames1-min_val) * ((scale_lims[1]-scale_lims[0])/(max_val-min_val)) + scale_lims[0]
        
        min_val = np.min(stim_frames2,axis=(0,1,2),keepdims=True)
        max_val = np.max(stim_frames2,axis=(0,1,2),keepdims=True)
        stim_frames2 = (stim_frames2-min_val) * ((scale_lims[1]-scale_lims[0])/(max_val-min_val)) + scale_lims[0]


    # plt.hist(stim_frames1[:,:,:,5,5].flatten(),20);plt.xlim([-10,10])
    data_t=data_t._replace(X=stim_frames1)
    data_v=data_v._replace(X=stim_frames2)
    
    return data_t,data_v

def normalize_resp(data_t,data_v):
    y1 = data_t.y
    y2 = data_v.y
    
    y1_a = np.moveaxis(y1,1,0)
    y2_a = np.moveaxis(y2,1,0)
    
    y1_b = y1_a.reshape(y1_a.shape[0],-1)
    y2_b = y2_a.reshape(y2_a.shape[0],-1)
    
    y1_c = y1_b/y1_b.max(axis=-1,keepdims=True)
    y2_c = y2_b/y2_b.max(axis=-1,keepdims=True)
    
    y1_d = y1_c.reshape(y1_a.shape)
    y1_d = np.moveaxis(y1_d,0,1)
    y2_d = y2_c.reshape(y2_a.shape)
    y2_d = np.moveaxis(y2_d,0,1)
    
    data_t=data_t._replace(y=y1_d)
    data_v=data_v._replace(y=y2_d)
    
    return data_t,data_v

# def applyLightIntensities(meanIntensity,X,t_frame):

#     # X = X.astype('float32')
#     X_norm = X
#     X_min = np.min(X)
#     if X_min<0:
#         X_norm = X+np.abs(X_min)
    
#     new_max = 2*meanIntensity
#     new_min = 1e-3 #meanIntensity/300
#     X_rstar = X_norm * (new_max - new_min) + new_min
#     X_rstar = X_rstar * 1e-3 * t_frame  # photons per time bin 
#     return X_rstar

def applyLightIntensities(meanIntensity,X,t_frame):

    # X = X.astype('float32')
    X_norm = X
    X_min = np.min(X)
    if X_min<0:
        X_norm = X+np.abs(X_min)
    old_min = X_norm.min()
    old_max = X_norm.max()
    new_max = 2*meanIntensity
    new_min = 1e-3 #meanIntensity/300
    X_rstar = ((X_norm - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    X_rstar = X_rstar * 1e-3 * t_frame  # photons per time bin 
    return X_rstar





expDate = '20240229C'   # 20240229C   20230725C
natstim_idx_val_all = [4]#,3,4,6,7,8]#6,7,8]
# natstim_idx_val_all = [0,1,2,3,4,5,7,8] #[0,1,2,3,5,7,8] #[0,1,2,3,4,5,6,7,8]
# natstim_idx_val_all = np.arange(0,161)
dataset= '200k'
file_suffix = '_spatResamp_sig-12'
D_TYPE = 'f4'

expDates = expDate #'20240229C'    #  20230725C
# path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_mike/',expDate,'datasets_maps/padded/')            
path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_mike/',expDate,'datasets_maps/ContraM/')   
# path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_mike/',expDate,'datasets_maps/stretched/')   
# path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_mike/',expDate,'datasets_maps/stretched_midgets/')   

# path_dataset = os.path.join('/home/saad/postdoc_db/analyses/data_mike/',expDate,'datasets_maps/raw/')   

         

t_frame = 8

# fname_data_train_val_test = os.path.join(path_dataset,(expDate+'_dataset_train_val_test_'+STIM+'_'+dataset+'-'+file_suffix+'_'+D_TYPE+'_'+str(t_frame)+'ms'))

BUILD_MAPS = False
MAX_RGCS = 500


trainingSamps_dur = 0.5
    
# Check whether the filename has multiple datasets that need to be merged
    
fname_data_train_val_test_all = []
i=0
for i in range(len(natstim_idx_val_all)):
    natstim_idx_val = natstim_idx_val_all[i]

    STIM = 'NATSTIM'+str(natstim_idx_val)+'_CORR' # 'CB'  'NATSTIM'

    name_datasetFile = expDate+'_dataset_train_val_test_'+STIM+'_'+dataset+file_suffix+'_'+D_TYPE+'_'+str(t_frame)+'ms.h5'
    fname_data_train_val_test_all.append(os.path.join(path_dataset,name_datasetFile))

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
psf_params = dict(pixel_neigh=3,method='square')
FRAC_U_TRTR = 1
d=0
dict_train = {};dict_trtr={};dict_trval={}
dict_val = {}
dict_test = {}
unames_allDsets = []
data_info = {}
dict_dinf = {}
nsamps_alldsets_loaded = []
for d in range(len(fname_data_train_val_test_all)):
    
    exp = expDates#[d]
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

    data_val = Exptdata_spikes(data_val.X,np.squeeze(data_val.y),data_val.spikes)
    
    info_unitSplit = data_handler_ej.load_info_unitSplit(fname_data_train_val_test_all[d])
    info_unitSplit['unames_train'] = parameters['unames'][info_unitSplit['idx_train']]
    info_unitSplit['unames_val'] = parameters['unames'][info_unitSplit['idx_val']]

    # parameters['unit_locs'][:,0] = parameters['unit_locs'][:,0]+1

    t_frame = parameters['t_frame']     # time in ms of one frame/sample 

    frac_train_units = parameters['frac_train_units']
    
    if 'PR' in mdl_name:
        meanLightLevel = 200000 #50 #3000   #50
        X = applyLightIntensities(meanLightLevel,data_train.X,parameters['t_frame'])
        data_train = data_train._replace(X=X)
        
        X = applyLightIntensities(meanLightLevel,data_val.X,parameters['t_frame'])
        data_val = data_val._replace(X=X)
        
    else:
        # scale_lims = [-6.90,6.9]
        # scale_lims = [-6.0,6.0]
        scale_lims = [0.2,0.8]
        data_train,data_val=normalize_stim(data_train,data_val,scale_lims=scale_lims)
        # data_train,data_val=scale_intensity(data_train,data_val)


    # data_train,data_val=normalize_resp(data_train,data_val)

    # data_train,data_val=scale_intensity(data_train,data_val)

    data_train,data_val,dinf = handler_maps.arrange_data_formaps(exp,data_train,data_val,parameters,frac_train_units,
                                                                 psf_params=psf_params,info_unitSplit=None,BUILD_MAPS=BUILD_MAPS,MODE='validation',NORMALIZE_RESP=0)
    dinf['unit_locs_train'] = dinf['unit_locs'][dinf['idx_units_train']]
    dinf['unit_types_train'] = dinf['unit_types'][dinf['idx_units_train']]
    
    dinf['unit_locs_val'] = dinf['unit_locs'][dinf['idx_units_val']]
    dinf['unit_types_val'] = dinf['unit_types'][dinf['idx_units_val']]
    
    dict_val[fname_data_train_val_test_all[d]] = data_val
    dict_dinf[fname_data_train_val_test_all[d]] = dinf
    unames_allDsets.append(parameters['unames'])
    nsamps_alldsets_loaded.append(len(data_val.X))
    

cell_types_unique = np.unique(dinf['umaskcoords_train'][:,1])



# %% Sample units


X = data_val.X.copy()[:,:,:]
u=29
idx_start = 320#120 #320
idx_sacc = 519 #319 #519

X_chunk = data_val.X.copy()[idx_start:idx_sacc,:,:]
y_chunk = data_val.y.copy()[idx_start:idx_sacc,:]

# X_chunk = np.concatenate((np.ones((100,40,80))*X_chunk[0],X_chunk),axis=0)
# y_chunk = np.concatenate((np.ones((100,y_chunk.shape[-1]))*y_chunk[0],y_chunk),axis=0)

a = np.where(dinf['umaskcoords_train'][:,0]==u)[0]
ucoords = dinf['umaskcoords_train'][a,2:4]
X_rf = X_chunk[:,ucoords[:,1],ucoords[:,0]]
X_rf = np.mean(X_rf,axis=-1)

plt.plot(X_rf[temporal_width_prep:]);plt.show()
# plt.plot(y_chunk[temporal_width_prep:,u]);plt.show()

# %

# % Generate stimuli
def normalize_stim(stim,scale_lims):
    if len(scale_lims)>0:
        min_val = np.min(stim)
        max_val = np.max(stim)
        stim = (stim-min_val) * ((scale_lims[1]-scale_lims[0])/(max_val-min_val)) + scale_lims[0]
    return stim


#[17,50,100,250,500,2000] 
#[2,6,12,30,60,240]
t_frame=8
flash_dur_base = 240
flash_durs = np.array([2,4,6,12,30,60,flash_dur_base]) # 6,12,30,60
flash_contr = [-0.05]
flash_frames = 10


start=X_chunk.shape[0]
end = 50
bgr_dur = 60

scale_lims = [0,1]         # ON: [-6,6] | OFF: [-1,0]

total_dur = start+bgr_dur+flash_dur_base+flash_frames-10

t_contr = start+bgr_dur-temporal_width_prep

n_flashes = len(flash_durs)
stim_all = []
stimbase_all = []
y_all = []
ybase_all=[]

len_stim = []
ctr=-1
cs=0;cd=0;fd=0;fc=0;
for fd in range(len(flash_durs)):
    stim = np.ones((total_dur,40,80))
    stim[:len(X_chunk),:,:] = X_chunk
    stim[len(X_chunk):,:,:] = X_chunk[-1]

    stim_base = stim.copy()
    y_base_orig = np.zeros((len(stim_base),y_chunk.shape[-1]))
    y_base_orig[:len(y_chunk)]=y_chunk
    y_base_orig[len(y_chunk):]=1e-4
    y_base_orig = y_base_orig/(np.max(y_base_orig,axis=0)+1e-6)

    for fc in range(len(flash_contr)):
        bgr = X_chunk[-1]
        flash_val = bgr*((1+flash_contr[fc])/(1-flash_contr[fc]))
        flash_onset = len(X_chunk)+flash_durs[fd]
        flash_offset = flash_onset+flash_frames
        
        stim[flash_onset:flash_offset] = flash_val
            
    # stim = normalize_stim(stim,scale_lims)
    stim_frames = stim.copy() #np.tile(stim[:,None,None],(1,40,80))
    stim_frames = model.data_handler.rolling_window(stim_frames,temporal_width_prep)
    
    # stim_base = normalize_stim(stim_base,scale_lims)
    stim_frames_base = stim_base.copy() #np.tile(stim_base[:,None,None],(1,40,80))
    stim_frames_base = model.data_handler.rolling_window(stim_frames_base,temporal_width_prep)
    
    # % All Retinas Last Epoch
    
    batch_size = 32
    batch_idx = np.ceil(np.linspace(0,len(stim_frames),int(len(stim_frames)/batch_size))).astype('int')
    y_pred_map = []
    i=0
    for i in range(len(batch_idx)-1):
        batch_val = stim_frames[batch_idx[i]:batch_idx[i+1]]
        y_pred_batch = train_metalmaps.eval_step(mdl_state,batch_val,None)
        y_pred_map.append(y_pred_batch)
        
    y_pred_map = np.concatenate(y_pred_map)
    y_pred = np.array(model.train_metalmaps.pred_psfavg(y_pred_map,dinf['umaskcoords_train'],dinf['segment_size']))
    y_pred = y_pred[:,:y_chunk.shape[-1]]
    
    y_pred_base_map = []
    for i in range(len(batch_idx)-1):
        batch_val = stim_frames_base[batch_idx[i]:batch_idx[i+1]]
        y_pred_batch = train_metalmaps.eval_step(mdl_state,batch_val,None)
        y_pred_base_map.append(y_pred_batch)
        
    y_pred_base_map = np.concatenate(y_pred_base_map)
    y_base = np.array(model.train_metalmaps.pred_psfavg(y_pred_base_map,dinf['umaskcoords_val'],dinf['segment_size']))
    y_base = y_base[:,:y_chunk.shape[-1]]
    
    # y_sub = y-y_base
    
    stim_all.append(stim_frames)
    stimbase_all.append(stim_frames_base)
    
    y_all.append(y_pred)
    ybase_all.append(y_base)


stim_all = np.stack(stim_all)
stimbase_all = np.stack(stimbase_all)
y_all = np.stack(y_all)
ybase_all = np.stack(ybase_all)

y_all = y_all*3
y_all = y_all-np.min(y_all,axis=(0,1))[None,None,:]

ybase_all = ybase_all*3
ybase_all = ybase_all-np.min(ybase_all,axis=(0,1))[None,None,:]


# ysub_all = y_all-ybase_all


# ybase_all = y_all[-1,-60]*np.ones(y_all[-1].shape)
# rgb = y_all[-1,:-60,:]
# ybase_all[:len(rgb),:] = rgb

ysub_all = y_all-ybase_all
# y_sub_all

# %%
unit_names = ['OFF','ON']
sacc_offset = int((X_chunk.shape[0]-temporal_width_prep)*t_frame)

# %
for u in range(16,32):
    # u=18;  #[12,15,33,44
# %
# u = 29  #[d0: 6,15 | d7: 6,10,11,12,18]
    unit_loc = dinf['unit_locs'][u]
    unit_type = dinf['unit_types'][u]
    a = np.where(dinf['umaskcoords_train'][:,0]==u)[0]
    ucoords = dinf['umaskcoords_train'][a,2:4]
    
    # %
    stim_rf = stim_all[:,:,-1,ucoords[:,1],ucoords[:,0]]
    stim_rf = np.mean(stim_rf,axis=-1)
    stimbase_rf = stimbase_all[:,:,-1,ucoords[:,1],ucoords[:,0]]
    stimbase_rf = np.mean(stimbase_rf[-1],axis=-1)
    
    peaks = np.max(ysub_all,axis=1)
    peaks_vals = np.argmax(ysub_all,axis=1)
    
    x_vals=(np.arange(len(stim_frames)))
    
    # x_vals=x_vals-(X_chunk.shape[0]-temporal_width_prep)
    # x_vals=x_vals*t_frame
    # peaks_vals=peaks_vals-(X_chunk.shape[0]-temporal_width_prep)
    # peaks_vals=peaks_vals*t_frame
    
    
    # lim_y = [-0.1,0.9]
    fig,ax = plt.subplots(3,1,figsize=(13,10));ax=np.ravel(ax);plt.suptitle(u)
    for i in range(len(stim_all)):
        # ax[0].step(x_vals,stim_all[i,:,-1,unit_loc[1],unit_loc[0]]);
        ax[0].step(x_vals,stim_rf[i]);
        ax[1].plot(x_vals,y_all[i,:,u])
        ax[2].plot(x_vals,ysub_all[i,:,u])
    
    ax[0].step(x_vals,stimbase_rf,'--k');
    ax[1].plot(x_vals,ybase_all[-1,:,u],'--k')
    ax[2].plot(peaks_vals[:,u],peaks[:,u],'o')
    ax[2].plot([x_vals[0],x_vals[-1]],[peaks[-1,u],peaks[-1,u]],'--b')
        # ax[2].set_ylim([0,ysub_all[:,:,u].max()+(ysub_all[:,:,u].max()*.25)])
    ax[0].set_title('%s RGC'%unit_names[unit_type-1])
    ax[0].set_ylabel('Stimulus intensity')
    ax[1].set_ylabel('Model predicted RGC response')
    ax[2].set_ylabel('Baseline subtracted response')
    plt.setp(ax,xlabel='Time from movement offset (ms)')
    # plt.setp(ax,xticks=np.arange(-sacc_offset,2400, 200))
    # plt.setp(ax,xlim=[-600,2300])
    plt.show()



# %% Modulation indices
peaks = np.max(ysub_all,axis=1)

peaks_avg = np.mean(peaks,axis=-1)
plt.plot(peaks[:,5],'-o')