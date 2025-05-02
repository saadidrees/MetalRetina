#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 10:11:24 2021

@author: saad
"""

import numpy as np
import os
import h5py
import math
from scipy.stats import zscore
from model import utils_si
from model.performance import model_evaluate
import re
import json
# from tqdm import tqdm
import gc

from collections import namedtuple


def rolling_window(array, window, time_axis=0):
    """
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    array : array_like
        Array to add rolling window to

    window : int
        Size of rolling window

    time_axis : int, optional
        The axis of the temporal dimension, either 0 or -1 (Default: 0)
 
    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:

    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])
    """
    if window > 0:
        if time_axis == 0:
            array = array.T
    
        elif time_axis == -1:
            pass
    
        else:
            raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')
    
        assert window < array.shape[-1], "`window` is too long."
    
        # with strides
        shape = array.shape[:-1] + (array.shape[-1] - window, window)
        strides = array.strides + (array.strides[-1],)
        arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    
        if time_axis == 0:
            return np.rollaxis(arr.T, 1, 0)
        else:
            return arr
    else:
        # arr = arr[:,np.newaxis,:,:]
        return array
                 
def unroll_data(data,time_axis=0,rolled_axis=1):
    rgb = data[0]
    rgb = np.concatenate((rgb,data[1:,data.shape[1]-1,:]),axis=0)
    # rgb = np.concatenate((rgb,data[-1:,0,:]),axis=0)
    return rgb
        
def isintuple(x,name):
    t = type(x)
    attrs = getattr(t, '_fields', None)
    if name in attrs:
        return True
    else:
        return False

def load_data(fname_dataFile,frac_val=0.2,frac_test=0.05,filt_temporal_width=40,idx_cells=None,thresh_rr=0.45):
    
    # Data
    t_start = 500    # of frames to skip in the begining of stim file (each frame is 16 or 17 ms)   
    
    f = h5py.File(fname_dataFile,'r')
    data_key = 'data'
    stims_keys = list(f[data_key].keys())
    t_frame = np.array(f['data'][stims_keys[0]]['stim_frames'].attrs['t_frame'])
    
    #units
    units_all = np.array(f['/units'])
    units_all = utils_si.h5_tostring(units_all)
    
    if idx_cells is None:
        idx_cells_temp = np.array([np.arange(len(units_all))])
        idx_cells = idx_cells_temp[0,:]
    else:
        idx_cells = idx_cells

    
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    datasets = np.empty(len(stims_keys),dtype='object')
    stim_id = -1*np.ones(len(stims_keys),dtype='int32')
    # total_spikeCounts = np.zeros(len(idx_cells))
    
    # same_stims = ((0,2),(1,3))
    
    for s in range(len(stims_keys)):
        
        # stim info
        code_stim = '/'+data_key+'/'+stims_keys[s]
        stim_len = np.array(f[code_stim+'/spikeRate']).shape[0]
        
        idx_time = np.arange(t_start,stim_len)
         
        # firing rates
        resp = np.array(f[code_stim+'/spikeRate'])[idx_time,idx_cells_temp.T]
        resp = resp.T
        
        # spikeCounts = np.array(f[code_stim+'/spikeCounts'])[idx_time,idx_cells.T]
        # spikeCounts = spikeCounts.T    
       
        #  stim
        stim = np.array(f[code_stim+'/stim_frames'][idx_time,:])
    
        num_CB_x = f[code_stim+'/stim_frames'].attrs['num_checkers_x']   # cb_info['steps_x'][0][0]
        num_CB_y = f[code_stim+'/stim_frames'].attrs['num_checkers_y']   # cb_info['steps_y'][0][0]
        stim_id[s] = f[code_stim+'/stim_frames'].attrs['stim_id']
        
        # # map values to -1 and 1
        # stim = np.round(stim - stim.mean())
        # stim = stim/stim.max()       
        # stim = zscore(stim)
        # Roll for temporal dimension
        stim = np.reshape(stim,(stim.shape[0],num_CB_y,num_CB_x),order='F')       
        stim = rolling_window(stim,filt_temporal_width,time_axis=0) 
        
        resp = resp[filt_temporal_width:,:]
        # spikeCounts = spikeCounts[filt_temporal_width:,:]
        # total_spikeCounts = total_spikeCounts + np.sum(spikeCounts,axis=0)
     
        
        resp_norm = np.empty(resp.shape)
        resp_median = np.empty(resp.shape[1])
        for i in range(0,resp.shape[1]):
            rgb = resp[:,i]
            rgb[rgb==0] = np.nan
            resp_median[i] = np.nanmedian(rgb)
            temp = resp[:,i]/resp_median[i]
            temp[np.isnan(temp)] = 0
            resp_norm[:,i] = temp
        
    
        # idx_train = np.arange(resp_norm.shape[0])
     
        train_stim = stim #[idx_train]
        train_resp_norm = resp_norm #[idx_train,:]
        datasets[s] = Exptdata(train_stim, train_resp_norm)     # array of Exptdata datasets
        
    f.close()
    
    
    del train_stim
    del train_resp_norm
    del resp
    del resp_norm
    del stim
    
    # Check which stimuli files are same i.e. repeats - the same_stims contains index of the datasets that are repeats
    stim_unique = np.unique(stim_id)
    same_stims = np.empty(len(stim_unique),dtype='object')
    
    for i in range(len(stim_unique)):
        a = stim_id == stim_unique[i]
        a = np.where(a)[0]
        same_stims[i] = a
    
    idx_stim_noRepeat = list()
    for i in range(len(same_stims)):
        if same_stims[i].shape[0] < 2:
            idx_stim_noRepeat.append(i)
    
    same_stims = np.delete(same_stims,idx_stim_noRepeat) # the same_stims contains index of the datasets that are repeats
    

# Compute retinal reliability - old [correlation based]
    idx_start = 100
    maxLen = int((60*1000)/t_frame) # just take 60 s #resp.shape[0] - idx_start-1 #15000
    
    dist_cc = np.empty((len(same_stims),len(idx_cells)))
    
    for s in range(len(same_stims)):
        obs_rate = datasets[same_stims[s][0]].y[idx_start:maxLen,:]
        est_rate = datasets[same_stims[s][1]].y[idx_start:maxLen,:]
        x_mu = obs_rate - np.mean(obs_rate, axis=0)
        x_std = np.std(obs_rate, axis=0)
        y_mu = est_rate - np.mean(est_rate, axis=0)
        y_std = np.std(est_rate, axis=0)
        dist_cc[s,:] = np.squeeze(np.mean(x_mu * y_mu,axis=0) / (x_std * y_std))
        
    dist_cc_meanAcrossStims = np.mean(dist_cc,axis=0)       
    
    
# NEW - FEV method
    
    idx_unitsToTake = np.arange(units_all.shape[-1])
    numCells = len(idx_unitsToTake)
    dataset_rr = {}
    # For repeated datasets, put their responses in a single matrix so we can later easily compute variances
    for s in range(len(same_stims)):
        temp = np.empty((len(same_stims[s]),maxLen,numCells))    
        for t in range(len(same_stims[s])):
            temp[t,:,:] = datasets[same_stims[s][t]].y[idx_start:idx_start+maxLen,idx_unitsToTake]
            
            dict_vars = {
                 'subset_train': temp,
                 # 'dataset_name': np.atleast_1d(np.array(dset_name,dtype='bytes'))
                 }
         
            dataset_rr['stim_'+str(s)] = dict_vars

           
    dataset_rr_stims = list(dataset_rr.keys())  # name of unique datasets
    var_rate_uniqueTrials = np.array([]).reshape(0,numCells)    # rows correspond to different repeats; unique stimuli has been concatenated below  
    for s in dataset_rr_stims:
        rate_sameStim_trials = dataset_rr[s]['subset_train']   # [trials,frames,units]
        rate_sameStim_avgTrials = np.nanmean(rate_sameStim_trials,axis=0)
        
        rate_avgTrials_sub = rate_sameStim_trials - rate_sameStim_avgTrials[None,:,:]   # trial mean subtracted from rate [trials,frames,units] --> trial-trial noise in the signal
        var_sameStims = np.mean(rate_avgTrials_sub**2,axis=0)   # variance across trials
        
        var_rate_uniqueTrials = np.concatenate((var_rate_uniqueTrials,var_sameStims),axis=0)    # [(frames from stim_0, frames from stim_1,...), units]
    
    var_noise = np.nanmean(var_rate_uniqueTrials,axis=0)    # estimate of noise
       
    rate_all = np.array([]).reshape(0,numCells) 
    for s in dataset_rr_stims:      # concatenate rate from all trials of all stimuli
        for t in range(dataset_rr[s]['subset_train'].shape[0]):
            rgb = dataset_rr[s]['subset_train'][t,:,:]
            rate_all = np.vstack((rate_all,rgb))
    
    var_rate_all = np.var(rate_all,axis=0)  # total variance in the signal
    fractionExplainableVariance_allUnits = (var_rate_all - var_noise)/var_rate_all
    retinalReliability = np.nanmedian(fractionExplainableVariance_allUnits)

    data_quality = {
        'retinalReliability': retinalReliability,
        'dist_cc': dist_cc_meanAcrossStims,        
        'idx_unitsToTake': idx_unitsToTake,         # old metric for compatibility
        'fev_allUnits': fractionExplainableVariance_allUnits,   # This is for all the units!
        'retinalReliability': retinalReliability,
        'var_noise': var_noise,
        'uname_selectedUnits': units_all,  
        }
    print('Retinal Reliability: '+str(np.round(retinalReliability,2)))
    print('Number of selected cells: ',str(numCells))

    
    # Form validation dataset by taking a chunk out of every unique stimulus file
    stim_unique
    n_total = 0
    for s in range(len(datasets)):
        n_total = n_total + datasets[s].y.shape[0]
    n_val = int(np.ceil(frac_val*n_total))
    
    n_perUniqueStim = int(n_val/len(stim_unique))
    train_resp = np.array([]).reshape(0,numCells)
    train_stim = np.array([]).reshape(0,datasets[0].X.shape[1],datasets[0].X.shape[2])
    val_resp = np.array([]).reshape(0,numCells)
    val_stim = np.array([]).reshape(0,datasets[0].X.shape[1],datasets[0].X.shape[2])
    
    for s in range(len(datasets)):
        rgb_train_resp = datasets[s].y[:-n_perUniqueStim]
        rgb_train_stim = datasets[s].X[:-n_perUniqueStim]

        if s in stim_unique:
            rgb_val_resp = datasets[s].y[-n_perUniqueStim:]
            rgb_val_stim = datasets[s].X[-n_perUniqueStim:]
            
        else:
            rgb_val_resp = np.array([]).reshape(0,numCells)
            rgb_val_stim = np.array([]).reshape(0,datasets[0].X.shape[1],datasets[0].X.shape[2])
 
            
        train_resp = np.concatenate((train_resp,rgb_train_resp),axis=0)
        train_stim = np.concatenate((train_stim,rgb_train_stim),axis=0)
        
        val_resp = np.concatenate((val_resp,rgb_val_resp),axis=0)
        val_stim = np.concatenate((val_stim,rgb_val_stim),axis=0)
            
    data_train = Exptdata(train_stim,train_resp)
    data_val = Exptdata(val_stim,val_resp)
    data_test = data_val;
            
    # make sure no validation or test samples are in training
    if filt_temporal_width > 0:
        a = unroll_data(data_train.X)
        a = a.reshape(a.shape[0],a.shape[1]*a.shape[2])
    else:
        a = data_train.X
    a = np.unique(a,axis=0)
    
    if filt_temporal_width > 0:
        b = unroll_data(data_val.X)
        b = b.reshape(b.shape[0],b.shape[1]*b.shape[2])
        b = b[filt_temporal_width-1:,:]
    else:
        b = data_val.X
    b = np.unique(b,axis=0)
    
    c = np.concatenate((a,b),axis=0)
    train_val_unique = np.unique(c,axis=0)
    train_val_unique = train_val_unique.shape[0]==c.shape[0]
    
    del a, b, c
    
    if train_val_unique is False:
        raise ValueError('Training and Validation sets not unique!')
    else:
        return data_train,data_val,data_test,data_quality,dataset_rr


def load_data_kr_allLightLevels(fname_dataFile,dataset,frac_val=0.2,frac_test=0.05,filt_temporal_width=60,idx_cells_orig=None,thresh_rr=0.15,N_split=0,NORM_RESP=1):
    
    # valSets = ['scotopic','photopic']   # embed this in h5 file
    
    # Data
    t_start = 0    # of frames to skip in the begining of stim file (each frame is 16 or 17 ms)   
    
    f = h5py.File(fname_dataFile,'r')
    data_key = dataset
    stims_keys = list(f[data_key].keys())
    
    #units
    units_all = np.array(f['/units'])
    units_all = utils_si.h5_tostring(units_all)
    
    if idx_cells_orig is None:
        idx_cells_temp = np.array([np.arange(len(units_all))])
        idx_cells = idx_cells_temp[0,:]
    else:
        idx_cells = idx_cells_orig

    
    Exptdata = namedtuple('Exptdata', ['X', 'y','spikes'])
    datasets = {}
    resp_non_norm = {}
    # total_spikeCounts = np.zeros(len(idx_cells))
    
    # same_stims = ((0,2),(1,3))
    
    
    for s in stims_keys:
        
        # stim info
        code_stim = '/'+data_key+'/'+s
        stim_len = np.array(f[code_stim+'/spikeRate']).shape[0]        
        idx_time = np.arange(t_start,stim_len)
          
        # firing rates
        resp = np.array(f[code_stim+'/spikeRate'])[idx_time]
        spikes = np.array(f[code_stim+'/spikeCounts'])[idx_time]
        # resp = resp.T
        if resp.ndim == 2:
            resp = resp[:,idx_cells]
            resp = resp[filt_temporal_width:]
            spikes = spikes[:,idx_cells]
            spikes = spikes[filt_temporal_width:]

        if resp.ndim > 2:
            resp = resp[filt_temporal_width:]
            # resp = np.moveaxis(resp,-1,0)
            resp = resp[:,idx_cells,:]
            spikes = spikes[filt_temporal_width:]
            spikes = spikes[:,idx_cells,:]
        
        # spikeCounts = np.array(f[code_stim+'/spikeCounts'])[idx_time,idx_cells.T]
        # spikeCounts = spikeCounts.T    
       
        #  stim
        stim = np.array(f[code_stim+'/stim_frames'][idx_time,:])
        t_frame = f[code_stim+'/stim_frames'].attrs['t_frame']   
        num_CB_x = f[code_stim+'/stim_frames'].attrs['num_checkers_x']   # cb_info['steps_x'][0][0]
        num_CB_y = f[code_stim+'/stim_frames'].attrs['num_checkers_y']   # cb_info['steps_y'][0][0]
        
        if stim.ndim==3:    # only if stim space is flattened
            stim = np.reshape(stim,(stim.shape[0],num_CB_y,num_CB_x),order='F')       
        # stim = zscore(stim)
        stim = rolling_window(stim,filt_temporal_width,time_axis=0) 
        
        
        # spikeCounts = spikeCounts[filt_temporal_width:,:]
        # total_spikeCounts = total_spikeCounts + np.sum(spikeCounts,axis=0)
    
        
        resp_norm = np.empty(resp.shape)
        
        if s == 'train': #resp_norm.ndim < 3:       Calculate the median based on training data and later apply that to test data
            resp_median = np.empty(resp.shape[1])
            for i in range(0,resp.shape[1]):
                rgb = resp[:,i]
                rgb[rgb==0] = np.nan
                resp_median[i] = np.nanmedian(rgb)
                
                temp = resp[:,i]/resp_median[i]
                temp[np.isnan(temp)] = 0
                resp_norm[:,i] = temp
                resp_orig = resp
            
        else:
            # resp_median = np.empty((resp.shape[1],resp.shape[2]))
            # for j in range(0, resp.shape[2]):
            #     for i in range(0,resp.shape[1]):
            #         rgb = resp[:,i,j]
            #         rgb[rgb==0] = np.nan
            #         resp_median[i,j] = np.nanmedian(rgb)
            if resp.ndim == 3:  # i.e. when the validation set contains of repetitions
                temp = resp/resp_median[None,:,None]
            else:
                temp = resp/resp_median[None,:]
                temp = temp[:,:,None]
                spikes = spikes[:,:,None]
            temp[np.isnan(temp)] = 0
            resp_norm = temp
            resp_orig = resp
        
        resp_orig[np.isnan(resp_orig)] = 0
        if NORM_RESP==1:
            datasets[s] = Exptdata(stim, resp_norm, spikes)
        else:
            datasets[s] = Exptdata(stim, resp_orig, spikes)
        resp_non_norm[s] = resp_orig
        
    f.close()
    
    if N_split>0:
        idx_half = int(np.floor(datasets['train'].X.shape[0])/2)
        N_samps = int(N_split*idx_half)
        
        X = datasets['train'].X[:N_samps]
        X = np.concatenate((X,datasets['train'].X[idx_half+100:idx_half+100+N_samps]),axis=0)
        
        y = datasets['train'].y[:N_samps]
        y = np.concatenate((y,datasets['train'].y[idx_half+100:idx_half+100+N_samps]),axis=0)
 
        spikes = datasets['train'].spikes[:N_samps]
        spikes = np.concatenate((spikes,datasets['train'].spikes[idx_half+100:idx_half+100+N_samps]),axis=0)

        datasets['train'] = Exptdata(X,y,spikes)

    
    
    del resp
    del resp_norm
    del stim
    
    # Check which stimuli are same i.e. repeats
    # stim_unique = np.unique(datasets['train'].X,axis=0)
    
    # select only cells where we wont have so many zero spikerate in a batch   
    if idx_cells_orig is None:
        rgb = np.sum(datasets['train'].y,axis=0)
        thresh = thresh_rr*np.median(rgb)           # This was 0.75 for the big rat dataset
        idx_unitsToTake = np.arange(len(units_all))
        idx_unitsToTake = idx_unitsToTake[rgb>thresh]
        units_all = units_all[idx_unitsToTake]
        resp_median_allUnits = resp_median[idx_unitsToTake]
    else:
        idx_unitsToTake = idx_cells
        units_all = units_all[idx_unitsToTake]
        resp_median_allUnits = resp_median
        
# dataset for retinal reliability

    try:
        temp_val = np.moveaxis(datasets['val'].y,-1,0)
        temp_val_spikes = np.moveaxis(datasets['val'].spikes,-1,0)
        dset_name = np.array(dataset)
     
    except:
        select_val = valSets[-1]
        temp_val = np.moveaxis(datasets['val_'+select_val].y,-1,0)
        temp_val_spikes = np.moveaxis(datasets['val_'+select_val].spikes,-1,0)
        dset_name = select_val
    
    dataset_rr = {}    
    if idx_cells_orig is None:
        temp_val = temp_val[:,:,idx_unitsToTake]
        temp_val_spikes = temp_val_spikes[:,:,idx_unitsToTake]

    dict_vars = {
         'val': temp_val,
         'val_spikes': temp_val_spikes,
         'dataset_name': np.atleast_1d(np.array(dset_name,dtype='bytes'))
         }
     
    dataset_rr['stim_0'] = dict_vars
     
    
# Retinal reliability method 1 - only take one validation set at this stage


    numCells = len(idx_unitsToTake)
    rate_sameStim_trials = dataset_rr['stim_0']['val']
    # if rate_sameStim_trials.shape[0]>1: # use this method if we have multiple trials
    _, fracExVar_allUnits, _, corr_allUnits = model_evaluate(rate_sameStim_trials,None,filt_temporal_width,RR_ONLY=True)
    retinalReliability_fev = np.round(np.median(fracExVar_allUnits),2)
    retinalReliability_corr = np.round(np.median(corr_allUnits),2)
                

    data_quality = {
        'retinalReliability_fev': retinalReliability_fev,
        'retinalReliability_corr': retinalReliability_corr,
        'uname_selectedUnits': units_all,  
        'idx_unitsToTake': idx_unitsToTake,
        'fracExVar_allUnits': fracExVar_allUnits,
        'corr_allUnits': corr_allUnits,
        'resp_median_allUnits' : resp_median_allUnits,
        }
    print('Retinal Reliability - FEV: '+str(np.round(retinalReliability_fev,2)))
    print('Retinal Reliability - Corr: '+str(np.round(retinalReliability_corr,2)))
    print('Number of selected cells: ',str(len(idx_unitsToTake)))
    

    
    # Split dataset into train, validation and test 
    # For validation at model running stage, only take one dataset. 
    try:
        datasets['val'].y.shape
        val_dset_name = 'val'
    except:
        val_dset_name = 'val_'+dset_name
        
    rgb = np.nanmean(datasets[val_dset_name].y,-1)
    rgb_spikes = np.nanmean(datasets[val_dset_name].spikes,-1)
    if idx_cells_orig is None:
        data_val = Exptdata(datasets[val_dset_name].X,rgb[:,idx_unitsToTake],rgb_spikes[:,idx_unitsToTake])      # for validation i take the mean rate across all trials
    else:
        data_val = Exptdata(datasets[val_dset_name].X,rgb,rgb_spikes)
    
    if frac_test>0:
        nsamples_test = int(np.floor(datasets['train'].X.shape[0]*frac_test))       
        idx_test = np.floor(np.arange((datasets['train'].X.shape[0]/2)-(nsamples_test/2),(datasets['train'].X.shape[0]/2)))
        idx_test = np.concatenate((idx_test,np.arange(datasets['train'].X.shape[0]-(nsamples_test/2),datasets['train'].X.shape[0]-1)),axis=0)
        idx_test = idx_test.astype('int')
        if idx_test.shape[0] % 2 != 0:
            idx_test = idx_test[2:]
            # idx_test = np.insert(idx_test,0,idx_test[0]-1)
            

        stim_test = datasets['train'].X[idx_test]
        resp_test = datasets['train'].y[idx_test,:]
        spikes_test = datasets['train'].spikes[idx_test,:]
        if idx_cells_orig is None:
            resp_test = resp_test[:,idx_unitsToTake]
            spikes_test = spikes_test[:,idx_unitsToTake]
        data_test = Exptdata(stim_test,resp_test,spikes_test)
        
        idx_train = np.setdiff1d(np.arange(0,datasets['train'].X.shape[0]),idx_test)
        stim_train = datasets['train'].X[idx_train]
        resp_train = datasets['train'].y[idx_train,:]   
        spikes_train = datasets['train'].spikes[idx_train,:]  
        if idx_cells_orig is None:
            resp_train = resp_train[:,idx_unitsToTake] 
            spikes_train = spikes_train[:,idx_unitsToTake] 
        
        data_train = Exptdata(stim_train,resp_train,spikes_train)
        
    else:
        data_test = data_val
        
        if idx_cells_orig is None:
            rgb = datasets['train'].y[:,idx_unitsToTake]
            rgb_spikes = datasets['train'].spikes[:,idx_unitsToTake]
            data_train = Exptdata(datasets['train'].X,rgb,rgb_spikes)
        else:
            data_train = Exptdata(datasets['train'].X,datasets['train'].y,datasets['train'].spikes)
    
    resp_orig = {}
    for i in resp_non_norm.keys():
        resp_orig[i] = resp_non_norm[i][:,idx_unitsToTake]
        
    # check_trainVal_contamination(data_train.X,data_val.X)
    # check_trainVal_contamination(data_train.X,data_test.X)
        
                     
    return data_train,data_val,data_test,data_quality,dataset_rr,resp_orig


def prepare_data_cnn3d(data,filt_temporal_width,idx_unitsToTake):
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    
    X = rolling_window(data.X,filt_temporal_width,time_axis=0)
    X = np.moveaxis(X,1,-1)
    X = X[:,np.newaxis,:,:,:]
    
    y = data.y[:,idx_unitsToTake]
    y = y[filt_temporal_width:]
    
    data = Exptdata(X,y)
    del X, y
    
    return data

def prepare_data_cnn2d(data,filt_temporal_width,idx_unitsToTake,num_chunks=1,MAKE_LISTS=False):  # resp_format = INDIVIDUAL, MAP
    
    if data.X.ndim==5:       # if the data has multiple stims and trials
        X = data.X
        y = data.y


        X_rgb = X.reshape(X.shape[0],X.shape[1],X.shape[2],-1,order='A')
        y_rgb = y.reshape(y.shape[0],y.shape[1],-1,order='A')

        
        if filt_temporal_width==1: # the case when filt_temporal_width=0
            X_rgb = X_rgb[:,None,:,:,:]
        else:
            X_rgb = rolling_window(X_rgb,filt_temporal_width,time_axis=0)
            y_rgb = y_rgb[filt_temporal_width:]
        
        
        X_list = []
        y_list = []
        i=0
        for i in range(X_rgb.shape[-1]):
            rgb = list(X_rgb[:,:,:,:,i])
            X_list = X_list + rgb
            
            rgb = list(y_rgb[:,:,i])
            y_list = y_list + rgb
        
        X = X_list
        y = y_list

        del X_rgb, y_rgb
    
    else:
        
        if filt_temporal_width>0:
            X = rolling_window(data.X,filt_temporal_width,time_axis=0)
            y = data.y[:,idx_unitsToTake]
            y = y[filt_temporal_width:]
            if isintuple(data,'spikes')==True:
                spikes = data.spikes[:,idx_unitsToTake]
                spikes = spikes[filt_temporal_width:]
                
            if isintuple(data,'y_trials')==True:
                y_trials = data.y_trials[:,idx_unitsToTake]
                y_trials = y_trials[filt_temporal_width:]

        else:
            X = np.expand_dims(data.X,axis=1)
            y = data.y[:,idx_unitsToTake]
            if isintuple(data,'y_trials')==True:
                y_trials = data.y_trials[:,idx_unitsToTake]

        if X.ndim==5:       # if the data has multiple stims
            X_rgb = np.moveaxis(X,0,-1)
            X_rgb =  X_rgb.reshape(X_rgb.shape[0],X_rgb.shape[1],X_rgb.shape[2],-1)
            X_rgb = np.moveaxis(X_rgb,-1,0)
            
            y_rgb = np.moveaxis(y,0,-1)
            y_rgb = y_rgb.reshape(y_rgb.shape[0],-1)
            y_rgb = np.moveaxis(y_rgb,-1,0)
            
            X = X_rgb
            y = y_rgb
            
            del X_rgb, y_rgb
            
        if MAKE_LISTS==True:    # if we want to arrange samples as lists rather than numpy arrays
            X = list(X)
            y = list(y)
            if isintuple(data,'y_trials')==True:
                y_trials = list(y_trials)

            if isintuple(data,'spikes')==True:
                spikes = list(spikes)

    
    data_vars = ['X','y','spikes','y_trials']
    dataDict = {}
    for var in data_vars:
        if var in locals():
            dataDict[var]=eval(var)
            
    data = namedtuple('Exptdata',dataDict)
    data=data(**dataDict)
    
    del X, y
    return data

def prepare_data_cnn2d_maps(data,filt_temporal_width,num_chunks=1,MAKE_LISTS=False,idx_unitsToTake=[]):  # resp_format = INDIVIDUAL, MAP

    if filt_temporal_width>0:
        X = rolling_window(data.X,filt_temporal_width,time_axis=0)
        y = data.y
        if len(idx_unitsToTake)==0:       # if data is in map format
            y = y[filt_temporal_width:]
        else:
            y = y[filt_temporal_width:][:,idx_unitsToTake]         # if data is in units format then repeat the last unit for vectorizatrion
        if isintuple(data,'spikes')==True:
            spikes = data.spikes[filt_temporal_width:]
            
        if isintuple(data,'y_trials')==True:
            y_trials = data.y_trials[filt_temporal_width:]

    else:
        X = np.expand_dims(data.X,axis=1)
        if len(idx_unitsToTake)==0:
            y = data.y
        else:
            y = data.y[:,idx_unitsToTake]

    if X.ndim>=5:       # if the data has multiple stims
        X_rgb = np.moveaxis(X,0,-1)
        X_rgb =  X_rgb.reshape(X_rgb.shape[0],X_rgb.shape[1],X_rgb.shape[2],-1)
        X_rgb = np.moveaxis(X_rgb,-1,0)
        
        y_rgb = np.moveaxis(y,0,-1)
        y_rgb = y_rgb.reshape(y_rgb.shape[0],-1)
        y_rgb = np.moveaxis(y_rgb,-1,0)
        
        X = X_rgb
        y = y_rgb
        
        del X_rgb, y_rgb
        
    if MAKE_LISTS==True:    # if we want to arrange samples as lists rather than numpy arrays
        X = list(X)
        y = list(y)
        if isintuple(data,'y_trials')==True:
            y_trials = list(y_trials)

        if isintuple(data,'spikes')==True:
            spikes = list(spikes)

    
    data_vars = ['X','y','spikes','y_trials']
    dataDict = {}
    for var in data_vars:
        if var in locals():
            dataDict[var]=eval(var)
            
    data = namedtuple('Exptdata',dataDict)
    data=data(**dataDict)
    
    del X, y
    return data


"""

    if X.ndim==6:       # if the data has multiple stims and trials
        # X_rgb = np.moveaxis(X,0,-1)
        # X_rgb = X_rgb.reshape(X_rgb.shape[0],X_rgb.shape[1],X_rgb.shape[2],X_rgb.shape[3],-1)
        # X_rgb =  X_rgb.reshape(X_rgb.shape[0],X_rgb.shape[1],X_rgb.shape[2],-1)
        # X_rgb = np.moveaxis(X_rgb,-1,0)
        
        # y_rgb = np.moveaxis(y,0,-1)
        # y_rgb = y_rgb.reshape(y_rgb.shape[0],y_rgb.shape[1],-1)
        # y_rgb = y_rgb.reshape(y_rgb.shape[0],-1)
        # y_rgb = np.moveaxis(y_rgb,-1,0)
        
        # X = X_rgb
        # y = y_rgb
        
        # del X_rgb, y_rgb
        
        
        # num_chunks = int(np.ceil(X.shape[0]/chunk_size))
        
        X_rgb = X.reshape(X.shape[0],X.shape[1],X.shape[2],-1,order='A')
        X_rgb = rolling_window(X_rgb,filt_temporal_width,time_axis=0)
        # rgb_X = np.moveaxis(rgb_X,0)
        
        y_rgb = y.reshape(y.shape[0],y.shape[1],-1,order='A')
        y_rgb = y_rgb[filt_temporal_width:]
        # rgb_y = np.moveaxis(rgb_y,-1,0)
        
        X_list = []
        y_list = []
        i=0
        for i in range(X_rgb.shape[-1]):
            rgb = list(X_rgb[:,:,:,:,i])
            X_list = X_list + rgb
            
            rgb = list(y_rgb[:,:,i])
            y_list = y_list + rgb
        
        X = X_list
        y = y_list
        
        # # chunk_size = 50
        # chunks_idx = np.linspace(0,rgb.shape[0],num_chunks+1,dtype='int')
        # data_idx = rgb.shape[1]*chunks_idx
        
        # X_rgb = np.ones((rgb.shape[0]*rgb.shape[1],rgb.shape[2],rgb.shape[3],rgb.shape[4]),dtype=X.dtype)
        # y_rgb = np.empty((0,y.shape[1]),dtype=y.dtype)
        # i=0
        # for i in tqdm(range(num_chunks)):
        #     rgb = np.moveaxis(X[chunks_idx[i]:chunks_idx[i+1]],0,-1)
        #     rgb = rgb.reshape(rgb.shape[0],rgb.shape[1],rgb.shape[2],rgb.shape[3],-1)
        #     rgb =  rgb.reshape(rgb.shape[0],rgb.shape[1],rgb.shape[2],-1)
        #     rgb = np.moveaxis(rgb,-1,0)
        #     X_rgb = np.concatenate((X_rgb,rgb),axis=0)
            
        #     rgb = np.moveaxis(y[chunks_idx[i]:chunks_idx[i+1]],0,-1)
        #     rgb = rgb.reshape(rgb.shape[0],rgb.shape[1],-1)
        #     rgb = rgb.reshape(rgb.shape[0],-1)
        #     rgb = np.moveaxis(rgb,-1,0)
        #     y_rgb = np.concatenate((y_rgb,rgb),axis=0)
            
        #     _=gc.collect()

        # X = X_rgb
        # y = y_rgb
        
        del X_rgb, y_rgb

"""

def prepare_data_pr_cnn2d(data,pr_temporal_width,idx_unitsToTake):
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    
    X = rolling_window(data.X,pr_temporal_width,time_axis=0)
    y = data.y[:,idx_unitsToTake]
    y = y[pr_temporal_width:]
        
    data = Exptdata(X,y)
    del X, y
    
    return data


def prepare_data_convLSTM(data,filt_temporal_width,idx_unitsToTake):
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    
    X = rolling_window(data.X,filt_temporal_width,time_axis=0)
    X = X[:,:,np.newaxis,:,:]
    
    y = data.y[:,idx_unitsToTake]
    y = y[filt_temporal_width:]
    
    data = Exptdata(X,y)
    del X, y
    
    return data

    

def save_h5Dataset(fname,data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig=None,data_train_info=None,data_val_info=None,dtype='float16'):
    
    if dtype=='float16':
        h5dtype='f2'
    elif dtype=='float16':
        h5dtype='f4'
    
    f = h5py.File(fname,'a')
       
    grp = f.create_group('/data_quality')
    keys = list(data_quality.keys())
    for i in range(len(data_quality)):
        if data_quality[keys[i]].dtype == 'O':
            grp.create_dataset(keys[i], data=np.array(data_quality[keys[i]],dtype='bytes'))        
        else:
            grp.create_dataset(keys[i], data=data_quality[keys[i]])
            
            
    if type(data_train) is dict:    # if the training set is divided into multiple datasets then create a group for each
        data_train_keys = list(data_train.keys())
        grp = f.create_group('/data_train')
        grp.create_dataset('X',data=data_train[data_train_keys[0]].X.astype(dtype),
                           compression='gzip',chunks=True,dtype=h5dtype,
                           maxshape=(None,data_train[data_train_keys[0]].X.shape[-2],data_train[data_train_keys[0]].X.shape[-1]))
        grp.create_dataset('y',data=data_train[data_train_keys[0]].y.astype(dtype),
                           compression='gzip',chunks=True,dtype=h5dtype,
                           maxshape=(None,data_train[data_train_keys[0]].y.shape[-1]))
        grp.create_dataset('spikes',data=data_train[data_train_keys[0]].spikes,compression='gzip',chunks=True,
                           maxshape=(None,data_train[data_train_keys[0]].spikes.shape[-1]))

        for i in range(1,len(data_train_keys)):
            f['data_train']['X'].resize((f['data_train']['X'].shape[0] + data_train[data_train_keys[i]].X.shape[0]),axis=0)
            f['data_train']['X'][-data_train[data_train_keys[i]].X.shape[0]:] = data_train[data_train_keys[i]].X.astype(dtype)
            
            f['data_train']['y'].resize((f['data_train']['y'].shape[0] + data_train[data_train_keys[i]].y.shape[0]),axis=0)
            f['data_train']['y'][-data_train[data_train_keys[i]].y.shape[0]:] = data_train[data_train_keys[i]].y.astype(dtype)

            f['data_train']['spikes'].resize((f['data_train']['spikes'].shape[0] + data_train[data_train_keys[i]].spikes.shape[0]),axis=0)
            f['data_train']['spikes'][-data_train[data_train_keys[i]].spikes.shape[0]:] = data_train[data_train_keys[i]].spikes

    # if type(data_train) is dict:    # if the training set is divided into multiple datasets then create a group for each
    #     for i in data_train.keys():
    #         grp = f.create_group('/'+i)
    #         grp.create_dataset('X',data=data_train[i].X,compression='gzip')
    #         grp.create_dataset('y',data=data_train[i].y,compression='gzip')
            
    else:
        grp = f.create_group('/data_train')
        grp.create_dataset('X',data=data_train.X.astype(dtype),compression='gzip',dtype=h5dtype)
        grp.create_dataset('y',data=data_train.y.astype(dtype),compression='gzip',dtype=h5dtype)
        grp.create_dataset('spikes',data=data_train.spikes,compression='gzip')
    
    
    if type(data_val)==dict: # if the training set is divided into multiple datasets
        data_val_keys = list(data_val.keys())
        grp = f.create_group('/data_val')
        grp.create_dataset('X',data=data_val[data_val_keys[0]].X.astype(dtype),
                           compression='gzip',chunks=True,dtype=h5dtype,
                           maxshape=(None,data_val[data_val_keys[0]].X.shape[-2],data_val[data_val_keys[0]].X.shape[-1]))
        grp.create_dataset('y',data=data_val[data_val_keys[0]].y.astype(dtype),
                           compression='gzip',chunks=True,dtype=h5dtype,
                           maxshape=(None,data_val[data_val_keys[0]].y.shape[-1]))
        grp.create_dataset('spikes',data=data_val[data_val_keys[0]].spikes,
                           compression='gzip',chunks=True,
                           maxshape=(None,data_val[data_val_keys[0]].spikes.shape[-1]))

        for i in range(1,len(data_val_keys)):
            f['data_val']['X'].resize((f['data_val']['X'].shape[0] + data_val[data_val_keys[i]].X.shape[0]),axis=0)
            f['data_val']['X'][-data_val[data_val_keys[i]].X.shape[0]:] = data_val[data_val_keys[i]].X.astype(dtype)
            
            f['data_val']['y'].resize((f['data_val']['y'].shape[0] + data_val[data_val_keys[i]].y.shape[0]),axis=0)
            f['data_val']['y'][-data_val[data_val_keys[i]].y.shape[0]:] = data_val[data_val_keys[i]].y.astype(dtype)
            
            f['data_val']['spikes'].resize((f['data_val']['spikes'].shape[0] + data_val[data_val_keys[i]].spikes.shape[0]),axis=0)
            f['data_val']['spikes'][-data_val[data_val_keys[i]].spikes.shape[0]:] = data_val[data_val_keys[i]].spikes
    else:
        grp = f.create_group('/data_val')
        grp.create_dataset('X',data=data_val.X.astype(dtype),compression='gzip',dtype=h5dtype)
        grp.create_dataset('y',data=data_val.y.astype(dtype),compression='gzip',dtype=hfdtype)
        grp.create_dataset('spikes',data=data_val.spikes,compression='gzip')
    
    if data_test != None:  # data_test is None if it does not exist. So if it doesn't exist, don't save it.
        grp = f.create_group('/data_test')
        grp.create_dataset('X',data=data_test.X.astype(dtype),compression='gzip',dtype=h5dtype)
        grp.create_dataset('y',data=data_test.y.astype(dtype),compression='gzip',dtype=h5dtype)
        # grp.create_dataset('spikes',data=data_test.spikes,compression='gzip')
        
    # Training data info
    if type(data_train_info)==dict: # if the training set is divided into multiple datasets
        for i in data_train_info.keys():
            grp = f.create_group('/'+i)
            info_keys = data_train_info[i].keys()
            for j in info_keys:
                grp.create_dataset(j,data=data_train_info[i][j])
    elif data_train_info!=None:
        grp = f.create_group('/data_train_info')
        info_keys = data_train_info[i].keys()
        for j in info_keys:
            grp.create_dataset(j,data=data_train_info[i][j])
    
    
    # Validation data info
    if type(data_val_info)==dict: # if the training set is divided into multiple datasets
        for i in data_val_info.keys():
            grp = f.create_group('/'+i)
            info_keys = data_val_info[i].keys()
            for j in info_keys:
                grp.create_dataset(j,data=data_val_info[i][j])
    elif data_val_info!=None:
        grp = f.create_group('/data_val_info')
        info_keys = data_val_info[i].keys()
        for j in info_keys:
            grp.create_dataset(j,data=data_val_info[i][j])

    
    if dataset_rr != None:
        grp = f.create_group('/dataset_rr')
        keys = list(dataset_rr.keys())
        for j in keys:
            grp = f.create_group('/dataset_rr/'+j)
            keys_2 = list(dataset_rr[j].keys())
            for i in range(len(keys_2)):
                if 'bytes' in dataset_rr[j][keys_2[i]].dtype.name:
                    grp.create_dataset(keys_2[i], data=dataset_rr[j][keys_2[i]])
                elif dataset_rr[j][keys_2[i]].dtype == 'O':
                    grp.create_dataset(keys_2[i], data=np.array(dataset_rr[j][keys_2[i]],dtype='bytes'))
                else:
                    grp.create_dataset(keys_2[i], data=dataset_rr[j][keys_2[i]],compression='gzip')
                
            
    grp = f.create_group('/parameters')   
    keys = list(parameters.keys())
    for i in range(len(parameters)):
        grp.create_dataset(keys[i], data=parameters[keys[i]]) 
        
    if resp_orig != None:
        grp = f.create_group('/resp_orig')
        keys = list(resp_orig.keys())
        for i in keys:
            grp.create_dataset(i, data=resp_orig[i],compression='gzip') 
            
    f.close()


def compute_samp_ranges(nsamps_alldsets, nsamps_train,thresh):
    starts = np.zeros_like(nsamps_alldsets)
    stops = np.zeros_like(nsamps_alldsets)
    
    stops[0] = nsamps_train
    
    i=4
    for i in range(1,len(nsamps_alldsets)):
        n = nsamps_alldsets[i]
        
        rgb = stops[i-1]+nsamps_train
        
        if rgb > n-1:
            stops[i] = n-1
            starts[i] = n-nsamps_train-1
        elif rgb>thresh-1:
            starts[i]=0
            stops[i] = nsamps_train
        else:
            stops[i] = rgb
            starts[i] = stops[i-1]
    
    idx_samps_ranges = np.stack([starts,stops]).T
    
            
    assert np.all(idx_samps_ranges[:,0]<nsamps_alldsets),'idx_samp start out of range'
    assert np.all(idx_samps_ranges[:,1]<nsamps_alldsets),'idx_samp end out of range'
    assert len(np.unique(np.diff(idx_samps_ranges,axis=1)))==1,'num samps not uniform across datasets'
    

    return idx_samps_ranges


    
# NEED TO TIDY THIS UP
def load_h5Dataset(fname_data_train_val_test,LOAD_TR=True,LOAD_VAL=True,LOAD_ALL_TR=False,nsamps_val=-1,nsamps_train=-1,nsamps_test=-1,RETURN_VALINFO=False,
                   idx_train_start=0,VALFROMTRAIN=False,LOADFROMBOOL=False,dtype='float16'):     # LOAD_TR determines whether to load training data or not. In some cases only validation data is required
    FLAG_VALFROMTRAIN=False
    f = h5py.File(fname_data_train_val_test,'r')
    t_frame = np.array(f['parameters']['t_frame'])
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    Exptdata_spikes = namedtuple('Exptdata', ['X', 'y','spikes'])
    f_keys = list(f.keys())

    if LOADFROMBOOL == True:
        if nsamps_train.dtype=='bool':
            idx = np.where(nsamps_train)
        else:
            idx = nsamps_train
        X = np.array(f['data_train']['X'][idx],dtype='float32')
        y = np.array(f['data_train']['y'][idx],dtype='float32')
        spikes = np.array(f['data_train']['spikes'][idx],dtype='float32')
        data_train = Exptdata_spikes(X,y,spikes)
        return data_train

        
    
    # some loading parameters
    if nsamps_val==-1 or nsamps_val==0:
        idx_val_start = 0
        idx_val_end = -1
    else:
        nsamps_val = int((nsamps_val*60*1000)/t_frame)      # nsamps arg is in minutes so convert to samples
        idx_val_start = 0
        idx_val_end = idx_val_start+nsamps_val
        
    if nsamps_test==-1 or nsamps_test==0:
        idx_test_start = 0
        idx_test_end = -1
    else:
        nsamps_test = int((nsamps_test*60*1000)/t_frame)      # nsamps arg is in minutes so convert to samples
        idx_test_start = f['data_test']['X'].shape[0]-nsamps_test
        idx_test_end = idx_test_start+nsamps_test

        
    idx_train_start = int((idx_train_start*60*1000)/(t_frame))    # mins to frames
    if nsamps_train==-1 or nsamps_train==0 :
        # idx_train_start = 0
        idx_train_end = -1
        # idx_data = np.arange(idx_train_start,np.array(f['data_train']['y'].shape[0]))
    else:
        LOAD_ALL_TR = False
        if nsamps_train <1000:  # i.e. if this is in time, else it is in samples
            nsamps_train = int((nsamps_train*60*1000)/t_frame)
        # idx_train_start = 0
        idx_train_end = idx_train_start+nsamps_train
        # idx_data = np.arange(idx_train_start,idx_train_end)
    
    
    # Training data
    if LOAD_TR==True:   # only if it is requested to load the training data
        regex = re.compile(r'data_train_(\d+)')
        dsets = [i for i in f_keys if regex.search(i)]
        if len(dsets)>0:    # if the dataset is split into multiple datasets
            if LOAD_ALL_TR==True:   # concatenate all datasets into one
                X = np.array([]).reshape(0,f[dsets[0]]['X'].shape[1],f[dsets[0]]['X'].shape[2])
                y = np.array([]).reshape(0,f[dsets[0]]['y'].shape[1])
                spikes = np.array([]).reshape(0,f[dsets[0]]['spikes'].shape[1])
                for i in dsets:
                    rgb = np.array(f[i]['X'])
                    X = np.concatenate((X,rgb),axis=0)
                    
                    rgb = np.array(f[i]['y'])
                    y = np.concatenate((y,rgb),axis=0)
                    
                    rgb = np.array(f[i]['spikes'])
                    spikes = np.concatenate((spikes,rgb),axis=0)

                X = X.astype('float32')
                y = y.astype('float32')
                spikes = spikes.astype('float32')
                
                
                                
                data_train = Exptdata_spikes(X,y,spikes)
            else:            # just pick the first dataset
                X = np.array(f[dsets[0]]['X'][idx_train_start:idx_train_end],dtype='float32')
                y = np.array(f[dsets[0]]['y'][idx_train_start:idx_train_end],dtype='float32')
                spikes = np.array(f[dsets[0]]['spikes'][idx_train_start:idx_train_end],dtype='float32')
                data_train = Exptdata_spikes(X,y,spikes)
                # data_train = Exptdata(X,y)
            
        else:   # if there is only one dataset
            if idx_train_end!=-1:
                if nsamps_val==0:   # for backwards compat
                    nsamps_val = int((0.3*60*1000)/t_frame)
                
                # Take data offset by start time. Take validation and test data from center of training data
                bool_idx_train = np.zeros(f['data_train']['X'].shape[0],dtype='bool')
                bool_idx_val = np.zeros(f['data_train']['X'].shape[0],dtype='bool')
                bool_idx_test = np.zeros(f['data_train']['X'].shape[0],dtype='bool')
                bool_idx_val_test = np.zeros(f['data_train']['X'].shape[0],dtype='bool')
                
                nsamps_test = int(nsamps_val/4)
                nsamps_val_test = nsamps_val+nsamps_test
                nsamps_train_val_test = nsamps_train+nsamps_val+nsamps_test

                mid =  int(nsamps_train_val_test/2) + idx_train_start       # mid point of train_val_test
                bool_idx_val_test[mid-int(nsamps_val_test/2):mid] = True
                bool_idx_val_test[mid:mid+int(nsamps_val_test/2)] = True
                bool_idx_train[idx_train_start:idx_train_start+nsamps_train_val_test] = True
                bool_idx_train[bool_idx_val_test] = False
                if VALFROMTRAIN == True:
                    idx_val_start = np.where(bool_idx_val_test)[0][0]
                    bool_idx_val[idx_val_start:idx_val_start+nsamps_val] = True
                    bool_idx_test[idx_val_start+nsamps_val:idx_val_start+nsamps_val+nsamps_test]  = True
                    
                    assert(sum(bool_idx_train&bool_idx_val)<2)
                    assert(sum(bool_idx_train&bool_idx_test)<2)
                    assert(sum(bool_idx_val&bool_idx_test)<2)
                    
                    data_val_info = dict(nsamps_train=nsamps_train,nsamps_val=nsamps_val,nsamps_test=nsamps_test,
                                         bool_idx_train=bool_idx_train,bool_idx_val=bool_idx_val,bool_idx_test=bool_idx_test)
                
                    # plt.plot(bool_idx_train);plt.plot(bool_idx_val);plt.plot(bool_idx_test);plt.xlim([260000,270000]);plt.show();
                
            else:
                bool_idx_train = np.ones(f['data_train']['X'].shape[0],dtype='bool')
            
            idx = np.where(bool_idx_train)
            X = np.array(f['data_train']['X'][idx],dtype='float32')
            y = np.array(f['data_train']['y'][idx],dtype='float32')
            if 'spikes' in f['data_train']:
                spikes = np.array(f['data_train']['spikes'][idx],dtype='float32')
                data_train = Exptdata_spikes(X,y,spikes)
            else:
                data_train = Exptdata(X,y)
            

    else:
        data_train = None
        
    # Validation data
    if VALFROMTRAIN==False:
        if LOAD_VAL==True:
            regex = re.compile(r'data_val_(\d+)')
            dsets = [i for i in f_keys if regex.search(i)]
            if len(dsets)>0:
                d = 0
                X = np.array(f[dsets[d]]['X'][idx_val_start:idx_val_end],dtype='float32') # only extract n_samples. if nsamps = -1 then extract all.
                y = np.array(f[dsets[d]]['y'][idx_val_start:idx_val_end],dtype='float32')
                spikes = np.array(f[dsets[d]]['spikes'][idx_val_start:idx_val_end],dtype='float32')
                data_val = Exptdata_spikes(X,y,spikes)
    
                      
                # dataset info
                if ('data_val_info_'+str(d)) in f:
                    data_val_info = {}
                    for i in f['data_val_info_'+str(d)].keys():
                        data_val_info[i] = np.array(f['data_val_info_'+str(d)][i])
                        
                    if 'triggers' in data_val_info:
                        data_val_info['triggers'] = data_val_info['triggers'][idx_val_start:idx_val_end]
                        
                else:
                    data_val_info = None
            else:
                X = np.array(f['data_val']['X'][idx_val_start:idx_val_end],dtype='float32') # only extract n_samples. if nsamps = -1 then extract all.
                y = np.array(f['data_val']['y'][idx_val_start:idx_val_end],dtype='float32')
                if 'spikes' in f['data_val']:
                    spikes = np.array(f['data_val']['spikes'][idx_val_start:idx_val_end],dtype='float32')
                    data_val = Exptdata_spikes(X,y,spikes)
                else:
                    data_val = Exptdata(X,y)
    
                # dataset info
                if 'data_val_info' in f:
                    data_val_info = {}
                    for i in f['data_val_info'].keys():
                        data_val_info[i] = np.array(f['data_val_info'][i])
                else:
                    data_val_info = None
        else:
            data_val = None
    else:
        idx = np.where(bool_idx_val)
        X = np.array(f['data_train']['X'][idx],dtype='float32')
        y = np.array(f['data_train']['y'][idx],dtype='float32')
        spikes = np.array(f['data_train']['spikes'][idx],dtype='float32')
        data_val = Exptdata_spikes(X,y,spikes)

       
    
    # Testing data
    if VALFROMTRAIN==False:
        if 'data_test' in f.keys():
            data_test = Exptdata(np.array(f['data_test']['X']),np.array(f['data_test']['y']))
        else:       
            data_test = None
    else:
        idx = np.where(bool_idx_test)
        X = np.array(f['data_train']['X'][idx],dtype='float32')
        y = np.array(f['data_train']['y'][idx],dtype='float32')
        spikes = np.array(f['data_train']['spikes'][idx],dtype='float32')
        data_test = Exptdata_spikes(X,y,spikes)
    
    # Quality data
    select_groups = ('data_quality')
    level_keys = list(f[select_groups].keys())
    data_quality = {}
    for i in level_keys:
        data_key = '/'+select_groups+'/'+i
        rgb = np.array(f[data_key])
        rgb_type = rgb.dtype.name
           
        if 'bytes' in rgb_type:
            data_quality[i] = utils_si.h5_tostring(rgb)
        else:
            data_quality[i] = rgb
            
    # Retinal reliability data
    select_groups = ('dataset_rr')
    level_keys = list(f[select_groups].keys())
    dataset_rr = {}
    for i in level_keys:
        level4_keys = list(f[select_groups][i].keys())
        temp_2 = {}

        for d in level4_keys:
            data_key ='/'+select_groups+'/'+i+'/'+d
        
            rgb = np.array(f[data_key])
            try:
                rgb_type = rgb.dtype.name
                if 'bytes' in rgb_type:
                    temp_2[d] = utils_si.h5_tostring(rgb)
                else:
                    temp_2[d] = rgb
            except:
                temp_2[d] = rgb
        dataset_rr[i] = temp_2
        
    # Parameters
    select_groups = ('parameters')
    level_keys = list(f[select_groups].keys())
    parameters = {}
    for i in level_keys:
        data_key = '/'+select_groups+'/'+i
        rgb = np.array(f[data_key])
        rgb_type = rgb.dtype.name
           
        if 'bytes' in rgb_type:
            parameters[i] = utils_si.h5_tostring(rgb)
        else:
            parameters[i] = rgb
    
    # Orig response (non normalized)
    try:
        select_groups = ('resp_orig')
        level_keys = list(f[select_groups].keys())
        resp_orig = {}
        for i in level_keys:
            data_key = '/'+select_groups+'/'+i
            rgb = np.array(f[data_key])
            rgb_type = rgb.dtype.name
               
            if 'bytes' in rgb_type:
                resp_orig[i] = utils_si.h5_tostring(rgb)
            else:
                resp_orig[i] = rgb
    except:
        resp_orig = None

            
    f.close()

    if RETURN_VALINFO==False:
        return data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig
    else:
        return data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig,data_val_info
    
"""
def load_h5Dataset(fname_data_train_val_test,LOAD_TR=True,LOAD_VAL=True,LOAD_ALL_TR=False,nsamps_val=-1,nsamps_train=-1,RETURN_VALINFO=False,idx_train_start=0,VALFROMTRAIN=False):     # LOAD_TR determines whether to load training data or not. In some cases only validation data is required
    FLAG_VALFROMTRAIN=False
    f = h5py.File(fname_data_train_val_test,'r')
    t_frame = np.array(f['parameters']['t_frame'])
    # some loading parameters
    if nsamps_val==-1 or nsamps_val==0:
        idx_val_start = 0
        idx_val_end = -1
    else:
        nsamps_val = int((nsamps_val*60*1000)/t_frame)      # nsamps arg is in minutes so convert to samples
        idx_val_start = 1000
        idx_val_end = idx_val_start+nsamps_val
        
    idx_train_start = int((idx_train_start*60*1000)/(t_frame))    # mins to frames
    if nsamps_train==-1 or nsamps_train==0 :
        # idx_train_start = 0
        idx_train_end = -1
        # idx_data = np.arange(idx_train_start,np.array(f['data_train']['y'].shape[0]))
    else:
        LOAD_ALL_TR = False
        if nsamps_train <1000:  # i.e. if this is in time, else it is in samples
            nsamps_train = int((nsamps_train*60*1000)/t_frame)
        # idx_train_start = 0
        idx_train_end = idx_train_start+nsamps_train
        # idx_data = np.arange(idx_train_start,idx_train_end)
    
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    f_keys = list(f.keys())
    
    # Training data
    if LOAD_TR==True:   # only if it is requested to load the training data
        regex = re.compile(r'data_train_(\d+)')
        dsets = [i for i in f_keys if regex.search(i)]
        if len(dsets)>0:    # if the dataset is split into multiple datasets
            if LOAD_ALL_TR==True:   # concatenate all datasets into one
                X = np.array([]).reshape(0,f[dsets[0]]['X'].shape[1],f[dsets[0]]['X'].shape[2])
                y = np.array([]).reshape(0,f[dsets[0]]['y'].shape[1])
                for i in dsets:
                    rgb = np.array(f[i]['X'])
                    X = np.concatenate((X,rgb),axis=0)
                    
                    rgb = np.array(f[i]['y'])
                    y = np.concatenate((y,rgb),axis=0)
                X = X.astype('float32')
                y = y.astype('float32')
                
                
                                
                data_train = Exptdata(X,y)
            else:            # just pick the first dataset
                X = np.array(f[dsets[0]]['X'][idx_train_start:idx_train_end],dtype='float32')
                y = np.array(f[dsets[0]]['y'][idx_train_start:idx_train_end],dtype='float32')
                data_train = Exptdata(X,y)
            
        else:   # if there is only one dataset
            if idx_train_end!=-1:
                
                # take a chunk from start and a chunk from the end
                # bool_idx = np.zeros(f['data_train']['X'].shape[0],dtype='bool')
                # bool_idx[:int(nsamps_train/2)] = True
                # bool_idx[-int(nsamps_train/2):] = True
                
                # take a chunk from middle
                bool_idx = np.zeros(f['data_train']['X'].shape[0],dtype='bool')
                mid = int(bool_idx.shape[0]/2) + idx_train_start
                assert(mid+nsamps_train < f['data_train']['X'].shape[0])
                bool_idx[mid-int(nsamps_train/2):mid] = True
                bool_idx[mid:mid+int(nsamps_train/2)] = True
                assert(sum(bool_idx)==nsamps_train)
                
                if VALFROMTRAIN==True:
                    bool_idx_val = np.zeros(f['data_train']['X'].shape[0],dtype='bool')
                    val_start = np.where(bool_idx)[-1][-1]+100
                    bool_idx_val[val_start:val_start+int(nsamps_val)] = True
                    assert(sum(bool_idx_val)==nsamps_val)
                    
                    bool_idx_test = np.zeros(f['data_train']['X'].shape[0],dtype='bool')
                    test_start = np.where(bool_idx|bool_idx_val)[-1][-1]+100
                    bool_idx_test[test_start:test_start+int(nsamps_val/2)] = True
                    assert(sum(bool_idx_test)==nsamps_val/2)
                    
                    FLAG_VALFROMTRAIN=True
            else:
                bool_idx = np.ones(f['data_train']['X'].shape[0],dtype='bool')
                
            int_idx = np.where(bool_idx)
            X = np.array(f['data_train']['X'][int_idx],dtype='float32')
            y = np.array(f['data_train']['y'][int_idx],dtype='float32')
            data_train = Exptdata(X,y)
            

    else:
        data_train = None
        
    # Validation data
    if FLAG_VALFROMTRAIN==False:
        if LOAD_VAL==True:
            regex = re.compile(r'data_val_(\d+)')
            dsets = [i for i in f_keys if regex.search(i)]
            if len(dsets)>0:
                d = 0
                X = np.array(f[dsets[d]]['X'][idx_val_start:idx_val_end],dtype='float32') # only extract n_samples. if nsamps = -1 then extract all.
                y = np.array(f[dsets[d]]['y'][idx_val_start:idx_val_end],dtype='float32')
                data_val = Exptdata(X,y)
    
                      
                # dataset info
                if ('data_val_info_'+str(d)) in f:
                    data_val_info = {}
                    for i in f['data_val_info_'+str(d)].keys():
                        data_val_info[i] = np.array(f['data_val_info_'+str(d)][i])
                        
                    if 'triggers' in data_val_info:
                        data_val_info['triggers'] = data_val_info['triggers'][idx_val_start:idx_val_end]
                        
                else:
                    data_val_info = None
            else:
                X = np.array(f['data_val']['X'][idx_val_start:idx_val_end],dtype='float32') # only extract n_samples. if nsamps = -1 then extract all.
                y = np.array(f['data_val']['y'][idx_val_start:idx_val_end],dtype='float32')
                data_val = Exptdata(X,y)
    
                # dataset info
                if 'data_val_info' in f:
                    data_val_info = {}
                    for i in f['data_val_info'].keys():
                        data_val_info[i] = np.array(f['data_val_info'][i])
                else:
                    data_val_info = None
        else:
            data_val = None
    else:
        int_idx = np.where(bool_idx_val)
        X = np.array(f['data_train']['X'][int_idx],dtype='float32')
        y = np.array(f['data_train']['y'][int_idx],dtype='float32')
        data_val = Exptdata(X,y)

       
    
    # Testing data
    if FLAG_VALFROMTRAIN==False:
        if 'data_test' in f.keys():
            data_test = Exptdata(np.array(f['data_test']['X']),np.array(f['data_test']['y']))
        else:       
            data_test = None
    else:
        int_idx = np.where(bool_idx_test)
        X = np.array(f['data_train']['X'][int_idx],dtype='float32')
        y = np.array(f['data_train']['y'][int_idx],dtype='float32')
        data_test = Exptdata(X,y)
    
    # Quality data
    select_groups = ('data_quality')
    level_keys = list(f[select_groups].keys())
    data_quality = {}
    for i in level_keys:
        data_key = '/'+select_groups+'/'+i
        rgb = np.array(f[data_key])
        rgb_type = rgb.dtype.name
           
        if 'bytes' in rgb_type:
            data_quality[i] = utils_si.h5_tostring(rgb)
        else:
            data_quality[i] = rgb
            
    # Retinal reliability data
    select_groups = ('dataset_rr')
    level_keys = list(f[select_groups].keys())
    dataset_rr = {}
    for i in level_keys:
        level4_keys = list(f[select_groups][i].keys())
        temp_2 = {}

        for d in level4_keys:
            data_key ='/'+select_groups+'/'+i+'/'+d
        
            rgb = np.array(f[data_key])
            try:
                rgb_type = rgb.dtype.name
                if 'bytes' in rgb_type:
                    temp_2[d] = utils_si.h5_tostring(rgb)
                else:
                    temp_2[d] = rgb
            except:
                temp_2[d] = rgb
        dataset_rr[i] = temp_2
        
    # Parameters
    select_groups = ('parameters')
    level_keys = list(f[select_groups].keys())
    parameters = {}
    for i in level_keys:
        data_key = '/'+select_groups+'/'+i
        rgb = np.array(f[data_key])
        rgb_type = rgb.dtype.name
           
        if 'bytes' in rgb_type:
            parameters[i] = utils_si.h5_tostring(rgb)
        else:
            parameters[i] = rgb
    
    # Orig response (non normalized)
    try:
        select_groups = ('resp_orig')
        level_keys = list(f[select_groups].keys())
        resp_orig = {}
        for i in level_keys:
            data_key = '/'+select_groups+'/'+i
            rgb = np.array(f[data_key])
            rgb_type = rgb.dtype.name
               
            if 'bytes' in rgb_type:
                resp_orig[i] = utils_si.h5_tostring(rgb)
            else:
                resp_orig[i] = rgb
    except:
        resp_orig = None

            
    f.close()

    if RETURN_VALINFO==False:
        return data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig
    else:
        return data_train,data_val,data_test,data_quality,dataset_rr,parameters,resp_orig,data_val_info
    
"""
def check_trainVal_contamination(stimFrames_train,stimFrames_val,filt_temporal_width=0):
    
    if filt_temporal_width>0:
        rgb = unroll_data(stimFrames_train)
        stimFrames_train_flattened = np.reshape(rgb,(rgb.shape[0],np.prod(rgb.shape[1:])))

        rgb = unroll_data(stimFrames_val)
        stimFrames_val_flattened = np.reshape(rgb,(rgb.shape[0],np.prod(rgb.shape[1:])))
        
    else:
        stimFrames_train_flattened = np.reshape(stimFrames_train,(stimFrames_train.shape[0],np.prod(stimFrames_train.shape[1:])))
        stimFrames_val_flattened = np.reshape(stimFrames_val,(stimFrames_val.shape[0],np.prod(stimFrames_val.shape[1:])))
    
    stimFrames_train_flattened_unique = np.unique(stimFrames_train_flattened,axis=0)
    
    if stimFrames_train_flattened_unique.shape[0] != stimFrames_train_flattened.shape[0]:
        Warning('training dataset contains repeated stimulus frames')
    
    if np.unique(stimFrames_val_flattened,axis=0).shape[0] != stimFrames_val_flattened.shape[0]:
        Warning('validation dataset contains repeated stimulus frames')
    
    a = stimFrames_train_flattened_unique #np.unique(stimFrames_train_flattened,axis=0)
    b = np.unique(stimFrames_val_flattened,axis=0)   
    c = np.concatenate((b,a),axis=0)
    d,d_idx = np.unique(c,axis=0,return_index=True)
    # d_idx = d_idx[d_idx>b.shape[0]] - b.shape[0]
    # idx_discard = np.setdiff1d(np.arange(0,a.shape[0]),d_idx)
    idx_discard = np.atleast_1d(np.empty(0))
    if np.abs(np.unique(c,axis=0).shape[0] - c.shape[0]) > 2:
        print('Training samples contains validation samples. Finding training indices to remove')
        
        idx_discard = get_index_contamination(stimFrames_train_flattened,stimFrames_val_flattened)
        return idx_discard
    
    else:
        print('No contamination found')
        return idx_discard

        
def get_index_contamination(stimFrames_train_flattened,stimFrames_val_flattened):
        
    a = stimFrames_train_flattened
    b = stimFrames_val_flattened
    
    progress_vals = np.arange(0,1,0.1)
    progress = progress_vals*a.shape[0]
    idx_discard = np.atleast_1d([])
    for i in range(a.shape[0]):
        train_frame = a[i]
        for j in range(b.shape[0]):
            val_frame = b[j]
            
            if np.all(train_frame == val_frame):
                idx_discard = np.append(idx_discard,i)
                break
            
        if any(progress == i):
            rgb = progress_vals[progress == i]
            print('progresss: '+str(rgb*100)+'%')
            
    return idx_discard


def merge_datasets(dict_data):
    
    dset_name = {}

    keys = list(dict_data.keys())
    key = keys[0]
    
    islist = 0
    has_y_trials = 0
    has_spikes = 0
    for key in keys:
        rgb = key
        rgb = os.path.split(rgb)[-1]
        # patt = r'_dataset_train_val_test_'
        # a = re.search(patt,rgb)
        dset_name[key] = rgb#[:a.start()+1]+rgb[a.end():]
        
        if isinstance(dict_data[key].X,list):
            islist+=1
        if isintuple(dict_data[key],'y_trials'):
            has_y_trials+=1
            if isinstance(dict_data[key].X,list):
                n_trials = len(dict_data[key].y_trials)
                y_trials_shape = dict_data[key].y_trials[0].shape
            else:
                n_trials = dict_data[key].y_trials.shape[-1]
                y_trials_shape = dict_data[key].y_trials.shape[1:]
        if isintuple(dict_data[key],'spikes'):
            has_spikes+=1
            

    dset_names = []
    if islist>0:
        X = []
        y = []
        if has_y_trials>0:
            y_trials = []
        if has_spikes>0:
            spikes = []

        for key in keys:
            if isinstance(dict_data[key].X,list)==False:
                X = X + list(dict_data[key].X)
                y = y + list(dict_data[key].y)
            else:
                X = X + dict_data[key].X
                y = y + dict_data[key].y
            try:
                spikes = spikes + dict_data[key].spikes
            except:
                print('skipping spikes dataset')
            if isintuple(dict_data[key],'y_trials'):
                y_trials = y_trials + dict_data[key].y_trials
            elif isintuple(dict_data[key],'y_trials')==False and has_y_trials>0:
                rgb = [None]*len(dict_data[key].X)
                y_trials = y_trials + rgb
            
            dset_names = dset_names + [dset_name[key]]*len(dict_data[key].X)
            
    else:
        X = np.zeros(([0,*dict_data[key].X.shape[1:]]),dtype=dict_data[key].X.dtype)
        y = np.zeros(([0,*dict_data[key].y.shape[1:]]),dtype=dict_data[key].X.dtype)
        spikes = np.zeros(([0,*dict_data[key].y.shape[1:]]),dtype=dict_data[key].X.dtype)
        if has_y_trials>0:
            y_trials = np.zeros(([0,*y_trials_shape]),dtype=dict_data[key].X.dtype)
        
        for key in keys:
            X = np.concatenate((X,dict_data[key].X),axis=0)
            y = np.concatenate((y,dict_data[key].y),axis=0)
            try:
                spikes = np.concatenate((spikes,dict_data[key].spikes),axis=0)
            except:
                print('skipping spikes dataset')
            if isintuple(dict_data[key],'y_trials'):
                y_trials = np.concatenate((y_trials,dict_data[key].y_trials),axis=0)
                
            elif isintuple(dict_data[key],'y_trials')==False and has_y_trials>0:
                rgb = np.zeros((dict_data[key].y.shape[0],*y_trials_shape),dtype=dict_data[key].y.dtype)
                rgb[:]=np.nan
                y_trials = np.concatenate((y_trials,rgb),axis=0)

            dset_names = dset_names + [dset_name[key]]*dict_data[key].X.shape[0]
    
    data_vars = ['X','y','spikes','y_trials','dset_names']

    dataDict = {}
    for var in data_vars:
        if var in locals():
            dataDict[var]=eval(var)
            
    data_tuple = namedtuple('Exptdata',dataDict)
    data=data_tuple(**dataDict)

    return data


def merge_datasets_maps(dict_data):
    
    dset_name = {}

    keys = list(dict_data.keys())
    key = keys[0]
    
    islist = 0
    has_y_trials = 0
    has_spikes = 0
    for key in keys:
        rgb = key
        rgb = os.path.split(rgb)[-1]
        # patt = r'_dataset_train_val_test_'
        # a = re.search(patt,rgb)
        dset_name[key] = rgb#[:a.start()+1]+rgb[a.end():]
        
        if isinstance(dict_data[key].X,list):
            islist+=1
        if isintuple(dict_data[key],'y_trials'):
            has_y_trials+=1
            if isinstance(dict_data[key].X,list):
                n_trials = len(dict_data[key].y_trials)
                y_trials_shape = dict_data[key].y_trials[0].shape
            else:
                n_trials = dict_data[key].y_trials.shape[-1]
                y_trials_shape = dict_data[key].y_trials.shape[1:]
        if isintuple(dict_data[key],'spikes'):
            has_spikes+=1
            

    dset_names = []
    if islist>0:
        X = []
        y = []
        if has_y_trials>0:
            y_trials = []
        if has_spikes>0:
            spikes = []

        for key in keys:
            if isinstance(dict_data[key].X,list)==False:
                X = X + list(dict_data[key].X)
                y = y + list(dict_data[key].y)
            else:
                X = X + dict_data[key].X
                y = y + dict_data[key].y
            try:
                spikes = spikes + dict_data[key].spikes
            except:
                print('skipping spikes dataset')
            if isintuple(dict_data[key],'y_trials'):
                y_trials = y_trials + dict_data[key].y_trials
            elif isintuple(dict_data[key],'y_trials')==False and has_y_trials>0:
                rgb = [None]*len(dict_data[key].X)
                y_trials = y_trials + rgb
            
            dset_names = dset_names + [dset_name[key]]*len(dict_data[key].X)
            
    else:
        X = np.zeros(([0,*dict_data[key].X.shape[1:]]),dtype=dict_data[key].X.dtype)
        y = np.zeros(([0,*dict_data[key].y.shape[1:]]),dtype=dict_data[key].X.dtype)
        spikes = np.zeros(([0,*dict_data[key].y.shape[1:]]),dtype=dict_data[key].X.dtype)
        if has_y_trials>0:
            y_trials = np.zeros(([0,*y_trials_shape]),dtype=dict_data[key].X.dtype)
        
        for key in keys:
            X = np.concatenate((X,dict_data[key].X),axis=0)
            y = np.concatenate((y,dict_data[key].y),axis=0)
            try:
                spikes = np.concatenate((spikes,dict_data[key].spikes),axis=0)
            except:
                print('skipping spikes dataset')
            if isintuple(dict_data[key],'y_trials'):
                y_trials = np.concatenate((y_trials,dict_data[key].y_trials),axis=0)
                
            elif isintuple(dict_data[key],'y_trials')==False and has_y_trials>0:
                rgb = np.zeros((dict_data[key].y.shape[0],*y_trials_shape),dtype=dict_data[key].y.dtype)
                rgb[:]=np.nan
                y_trials = np.concatenate((y_trials,rgb),axis=0)

            dset_names = dset_names + [dset_name[key]]*dict_data[key].X.shape[0]
    
    data_vars = ['X','y','spikes','y_trials','dset_names']

    dataDict = {}
    for var in data_vars:
        if var in locals():
            dataDict[var]=eval(var)
            
    data_tuple = namedtuple('Exptdata',dataDict)
    data=data_tuple(**dataDict)

    return data


def dataset_shuffle(data,n_train):
    import random
    
    X = []
    y = []
    dset_names = []
    idx_shuffle = list(range(n_train))
    random.shuffle(idx_shuffle)
    for i in idx_shuffle:
        X.append(data.X[i])
        y.append(data.y[i])
        if isintuple(data,'dset_names'):
            dset_names.append(data.dset_names[i])
    
        
    data_vars = ['X','y','spikes','y_trials','dset_names']

    dataDict = {}
    for var in data_vars:
        if var in locals():
            dataDict[var]=eval(var)
            
    data_tuple = namedtuple('Exptdata',dataDict)
    data=data_tuple(**dataDict)

    return data

def augment_datatuple(data,dict_fields):
    data_fields = list(data._fields)
    
    dataDict = {}
    for name in data_fields:
        dataDict[name]=getattr(data,name)
        
    for var in dict_fields:
        dataDict[var]=dict_fields[var]
            
    data_tuple = namedtuple('Exptdata',dataDict)
    data=data_tuple(**dataDict)

    return data


