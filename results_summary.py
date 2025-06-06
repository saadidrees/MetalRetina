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
from scipy.stats import sem, t


# % Single retina models - Test retinas

list_suffix = '20241115M' #'8M'#'20241115M'
testList = 'testlist_'+list_suffix
trainList = 'trainlist_'+list_suffix

path_dataset_base = '/home/saad/postdoc_db/analyses/data_ej/'


with open(os.path.join(path_dataset_base,'datasets',testList+'.txt'), 'r') as f:
    expDates_test = f.readlines()
expDates_test = [line.strip() for line in expDates_test][1:]

with open(os.path.join(path_dataset_base,'datasets',trainList+'.txt'), 'r') as f:
    expDates_train = f.readlines()
expDates_train = [line.strip() for line in expDates_train][1:]


durs = np.array([3,5,8,10,15,20,25,30])


fname_ft = '/mnt/devices/nvme-2tb/Dropbox/postdoc/projects/MetalRetina/results/finetuning/ft_testlist.pkl'
fname_single = '/mnt/devices/nvme-2tb/Dropbox/postdoc/projects/MetalRetina/results/singleretina/singleretina_testlist.pkl'

with open(fname_single, 'rb') as f:
    fev_grand,corr_grand,unames_val_all,idx_val_all,expDates_sel,durs = cloudpickle.load(f)

rgb = np.moveaxis(corr_grand,0,-1)
rgb = rgb.reshape(len(rgb),-1)
idxkeep = ~np.isnan(rgb[-1])
corr_single = rgb[:,idxkeep]


with open(fname_ft, 'rb') as f:
    ft_fev_grand,ft_corr_grand,pre_fev_grand,pre_corr_grand,expDates_sel,durs,lrs,idx_bestlr = cloudpickle.load(f)

idx_bestlr = idx_bestlr.astype('int')
corr_ft_allExps_allUnits = ft_corr_grand[np.arange(len(idx_bestlr)),idx_bestlr,:,-1]
corr_ft_allExps_valUnits = np.zeros_like(corr_ft_allExps_allUnits);corr_ft_allExps_valUnits[:]=np.nan

for i in range(len(corr_ft_allExps_allUnits)):
    idx_val = idx_val_all[i]
    corr_ft_allExps_valUnits[i,:,:len(idx_val)] = corr_ft_allExps_allUnits[i,:,idx_val].T

rgb = np.moveaxis(corr_ft_allExps_allUnits,0,-1)
rgb = rgb.reshape(len(rgb),-1)
idxkeep = ~np.isnan(rgb[-1])
corr_ft = rgb[:,idxkeep]

rgb = np.moveaxis(corr_ft_allExps_valUnits,0,-1)
rgb = rgb.reshape(len(rgb),-1)
idxkeep = ~np.isnan(rgb[-1])
corr_ft_valUnits = rgb[:,idxkeep]

    
# corr_ft = np.nanmedian(corr_ft,axis=-1)
# corr_ft_valUnits = np.nanmedian(corr_ft_valUnits,axis=-1)



rgb = np.moveaxis(pre_corr_grand,0,-1)
rgb = rgb.reshape(len(rgb),-1)
idxkeep = ~np.isnan(rgb[-1])
corr_pre = rgb[:,idxkeep]

corr_pre_allExps_valUnits = np.zeros_like(pre_corr_grand);corr_pre_allExps_valUnits[:]=np.nan
for i in range(len(pre_corr_grand)):
    idx_val = idx_val_all[i]
    corr_pre_allExps_valUnits[i,:len(idx_val)] = pre_corr_grand[i,idx_val]


rgb = corr_pre_allExps_valUnits.flatten()
idxkeep = ~np.isnan(rgb)
corr_pre_valUnits = rgb[idxkeep]


corr_single_med = np.nanmedian(corr_single,axis=-1)
se = np.nanstd(corr_single,axis=-1)/corr_single.shape[-1]
corr_single_ci = se*t.ppf(0.975,df=corr_single.shape[-1]-1)

corr_ft_med = np.nanmedian(corr_ft_valUnits,axis=-1)
se = np.nanstd(corr_ft_valUnits,axis=-1)/corr_ft_valUnits.shape[-1]
corr_ft_ci = se*t.ppf(0.975,df=corr_ft_valUnits.shape[-1]-1)

corr_pre_med = np.nanmedian(corr_pre_valUnits,axis=0)
se = np.nanstd(corr_pre_valUnits,axis=-1)/corr_pre_valUnits.shape[-1]
corr_ft_ci = se*t.ppf(0.975,df=corr_pre_valUnits.shape[-1]-1)


# %% Line plots

lim_y = [0.0,0.5]
# lim_x = lim_y
fig,ax = plt.subplots(1,1,figsize=(4,4));fig.suptitle('Exp: %s'%(list_suffix))

ax.plot(trainingDurs,corr_singlevec_med,'-o',label='SingleRet-Vec')
ax.plot(durs,corr_single_med,'-o',label='SingleRet')
# ax.errorbar(durs,corr_single_med,yerr=corr_single_ci,label='SingleRet')
ax.plot(durs,corr_ft_med,'-ro',label='MetalRet (fine-tuned)')
ax.plot([durs[0],durs[-1]],[corr_pre_med,corr_pre_med],'--k',label='MetalRet(zero-shot)')
ax.legend(loc='lower right')
ax.set_ylim(lim_y)
ax.text(15,0.2,'N=%d Test RGCs\nfrom %d Retinas'%(corr_single.shape[-1],len(expDates_sel)))
ax.set_xlabel('Training data (minutes)')
ax.set_ylabel('Prediction correlation median across\nvalidation RGCs and experiments')

# %%
lim_y = [0.0,0.7]
lim_x = lim_y

corr_single_exps = np.nanmedian(corr_grand,axis=-1)
corr_ft_exps = np.nanmedian(corr_ft_allExps_valUnits,axis=-1)
corr_pre_exps = np.nanmedian(corr_pre_allExps_valUnits,axis=-1)

dur_sel = -4

fig,ax = plt.subplots(1,3,figsize=(12,4));fig.suptitle('Exp: %s | Training data: %d mins'%(list_suffix,durs[dur_sel]))
ax[0].scatter(corr_single_exps[:,dur_sel],corr_pre_exps)
ax[0].plot([0,1],[0,1],'--k')
ax[0].legend(loc='lower right')
ax[0].set_ylim(lim_y)
ax[0].set_xlim(lim_x)
ax[0].set_xlabel('SingleRetina')
ax[0].set_ylabel('MetalRetina (zero-shot)')

ax[1].scatter(corr_single_exps[:,dur_sel],corr_ft_exps[:,dur_sel])
ax[1].plot([0,1],[0,1],'--k')
ax[1].legend(loc='lower right')
ax[1].set_ylim(lim_y)
ax[1].set_xlim(lim_x)
ax[1].set_xlabel('SingleRetina')
ax[1].set_ylabel('MetalRetina (Fine-tuned)')

ax[2].scatter(corr_pre_exps,corr_ft_exps[:,dur_sel])
ax[2].plot([0,1],[0,1],'--k')
ax[2].legend(loc='lower right')
ax[2].set_ylim(lim_y)
ax[2].set_xlim(lim_x)
ax[2].set_xlabel('MetalRetina (Zero-shot)')
ax[2].set_ylabel('MetalRetina (Fine-tuned)')


# %% Single retina models - Test retinas | VEC




path_base_single =  '/home/saad/data_hdd/analyses/data_ej/ss/models_correct/cluster/'
# trainingDurs = np.array([0.03,0.04,0.15,1,2.5,5,10,15,20,25,30])
trainingDurs = np.array([2.5,5,10,15,20,25,30])
select_rgctype = np.array([1,2])


U=762
fev_test_lastEpoch_AllExps = np.zeros((len(expDates_sel),len(trainingDurs),U));fev_test_lastEpoch_AllExps[:]=np.nan
corr_test_lastEpoch_AllExps = np.zeros((len(expDates_sel),len(trainingDurs),U));corr_test_lastEpoch_AllExps[:]=np.nan
num_rgcs_allExps = np.zeros((len(expDates_sel)));num_rgcs_allExps[:]=np.nan




path_mdlFold = os.path.join(path_base_single,'testing_retinas')
i=0;j=0
fname_missing_scratch = []
for i in range(len(expDates_sel)):
    exp = expDates_sel[i]


    path_single = os.path.join(path_mdlFold,exp,'CNN2D_FT')
    rgb = os.listdir(path_single)
    params = model.performance.getModelParams(rgb[0])
    nrgcs = params['U']




    for j in range(len(trainingDurs)):


        fname_model = 'U-%d_T-060_C1-20-03_C2-30-03_C3-30-03_C4-30-03_BN-1_MP-2_LR-0.001_TRSAMPS-%03d_TR-01'%(nrgcs,trainingDurs[j])    
        fname_perf = os.path.join(path_single,fname_model,'performance','%s_%s.pkl'%(exp,fname_model))
        if os.path.exists(fname_perf):


            with open(fname_perf,'rb') as f:
                rgb = cloudpickle.load(f)
            perf_data = rgb[3]
            
            fev_test_allEpochs = perf_data['fev_test_allUnits_allEpochs']
            corr_test_allEpochs = perf_data['corr_test_allUnits_allEpochs']
            
            
            fev_test_lastEpoch = perf_data['fev_test_allUnits_bestEpoch'][idx_val_all[i]]
            corr_test_lastEpoch = perf_data['corr_test_allUnits_bestEpoch'][idx_val_all[i]]
                
            n_epochs_exp = 45#len(fev_val_allEpochs)


            fev_test_lastEpoch_AllExps[i,j,:len(fev_test_lastEpoch)] = fev_test_lastEpoch
            corr_test_lastEpoch_AllExps[i,j,:len(corr_test_lastEpoch)] = corr_test_lastEpoch


        else:
            fname_missing_scratch.append(fname_perf)



rgb = np.moveaxis(corr_test_lastEpoch_AllExps,0,-1)
rgb = rgb.reshape(len(rgb),-1)
idxkeep = ~np.isnan(rgb[-1])
corr_singlevec = rgb[:,idxkeep]

corr_singlevec_med = np.nanmedian(corr_singlevec,axis=-1)


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


f_mamlmaps = os.path.join(path_base,'metalzero1step',trainList,'mad',
                          'CNN2D_MAP3/U-71_T-080_C1-64-07_C2-128-07_C3-256-07_BN-1_MP-0_LR-0.0001_TRSAMPS-005_TR-01/performance/metalzero1step_perf_testExps_lastEpoch.pkl')


with open(os.path.join(f_mamlmaps), 'rb') as f:
    perf_mamlmaps = cloudpickle.load(f)


med_mamlmaps = perf_mamlmaps[0]['predCorr_medUnits_allExps']
# idx_train = np.arange(24,95)


# med_mamlmaps = med_mamlmaps[idx_train]
# %%
med_singleRetVec_test = np.nanmedian(corr_test_lastEpoch_AllExps,axis=-1)
# med_singleRetVec_train = np.nanmedian(corr_train_lastEpoch_AllExps,axis=-1)


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


# path_figsave = '/home/saad/data/Dropbox/postdoc/projects/MetalRetina/results/progress'
# fname_fig = 'scat_singlerr'
# fig.savefig(os.path.join(path_figsave,fname_fig+'.png'),dpi=600)
# fig.savefig(os.path.join(path_figsave,fname_fig+'.svg'),dpi=600)




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


