#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 09:54:56 2021
 
@author: saad
"""
import h5py
import numpy as np
import re
import random
import math
from matplotlib import pyplot as plt
import multiprocessing as mp
from functools import partial
import os

def get_weightsDict(mdl):
    names = [weight.name for layer in mdl.layers for weight in layer.weights]
    weights = mdl.get_weights()
    weights_dict = {}
    for i in range(len(names)):
        weight_name = names[i][:-2]        
        weights_dict[weight_name] = np.atleast_1d(np.squeeze(weights[i]))
        
    return weights_dict

def get_weightsOfLayer(weights_dict,layer_name):
    weights_keys = list(weights_dict.keys())
    rgb = re.compile(layer_name+'*')
    layer_weight_names = list(filter(rgb.match, weights_keys))
    weights_layer = {}
    for l_name in layer_weight_names:
        param_name_full = os.path.basename(l_name)
        rgb = re.findall(r'[^0-9]',param_name_full)
        rgb = ''.join(rgb)
        if rgb[-1] == '_':
            rgb = rgb[:-1]
        
        param_name = rgb
        weights_layer[param_name] = weights_dict[l_name]
    
    return weights_layer
        
def get_layerIdx(mdl,layer_name):
    # layer_name = 'batch_normalization'
    layer_idx = []
    for i in range(len(mdl.layers)):
        if layer_name in mdl.layers[i].name:
            layer_idx.append(i)
    return layer_idx
        

  
def save_modelPerformance(fname_save_performance,fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dataset_rr,datasets_val,dataset_pred):

    f = h5py.File(fname_save_performance,'w')
    
    # grpName_mdl = fname_model
    # grp_exist = '/'+grpName_mdl in f
    # if grp_exist:
    #     del f[grpName_mdl]
        
    # grp_model = f.create_group(grpName_mdl)
     
    keys = list(metaInfo.keys())
    for i in range(len(metaInfo)):
        f.create_dataset(keys[i], data=metaInfo[keys[i]])
    
    grp = f.create_group('/data_quality')
    keys = list(data_quality.keys())
    for i in range(len(data_quality)):
        if data_quality[keys[i]].dtype == 'O':
            grp.create_dataset(keys[i], data=np.array(data_quality[keys[i]],dtype='bytes'))        
        else:
            grp.create_dataset(keys[i], data=data_quality[keys[i]])
    
    grp = f.create_group('/model_performance')
    keys = list(model_performance.keys())
    for i in range(len(model_performance)):
        if model_performance[keys[i]].dtype == 'O':
            grp.create_dataset(keys[i], data=np.array(model_performance[keys[i]],dtype='bytes'))        
        else:
            grp.create_dataset(keys[i], data=model_performance[keys[i]])
    
    
    grp = f.create_group('/model_params')
    keys = list(model_params.keys())
    for i in range(len(model_params)):
        grp.create_dataset(keys[i], data=model_params[keys[i]])
    
    
    grp = f.create_group('/stim_info')
    keys = list(stim_info.keys())
    for i in range(len(stim_info)):
        grp.create_dataset(keys[i], data=stim_info[keys[i]])
        
    if dataset_rr is not None:
        grp_exist = '/dataset_rr' in f
        if not grp_exist:
            grp = f.create_group('/dataset_rr')
            keys = list(dataset_rr.keys())
            for j in keys:
                grp = f.create_group('/dataset_rr/'+j)
                keys_2 = list(dataset_rr[j].keys())
                for i in range(len(keys_2)):
                    if 'bytes' in dataset_rr[j][keys_2[i]].dtype.name:
                        grp.create_dataset(keys_2[i], data=dataset_rr[j][keys_2[i]])
                    else:
                        grp.create_dataset(keys_2[i], data=dataset_rr[j][keys_2[i]],compression='gzip')
            
            
    # grp_exist = '/val_test_data' in f
    # if not grp_exist:
        
    #     keys = list(datasets_val.keys())
    #     grp = f.create_group('/val_test_data')
        
    #     for i in range(len(datasets_val)):
    #         grp.create_dataset(keys[i], data=datasets_val[keys[i]],compression='gzip')
            
            
    grp = f.create_group('/dataset_pred')
    keys = list(dataset_pred.keys())
    for i in range(len(dataset_pred)):
        if dataset_pred[keys[i]].dtype == 'O':
            grp.create_dataset(keys[i],data=np.array(dataset_pred[keys[i]],dtype='bytes'))
        else:
            grp.create_dataset(keys[i],data=dataset_pred[keys[i]],compression='gzip')
    
     
    f.close()
 
def getModelParams(fname_modelFolder):
    params = {}
    
    p_regex = re.compile(r'U-(\d+\.\d+)')
    rgb = p_regex.search(fname_modelFolder)
    if rgb != None:
        params['U'] = float(rgb.group(1))
    else:
        p_regex = re.compile(r'U-(\d+)')
        rgb = p_regex.search(fname_modelFolder)
        params['U'] = int(rgb.group(1))
    
    try:
        rgb = re.compile(r'P-(\d+)')
        rgb = rgb.search(fname_modelFolder)
        params['P'] = int(rgb.group(1))
    except:
        pass
    
    rgb = re.compile(r'T-(\d+)')
    rgb = rgb.search(fname_modelFolder)
    params['T'] = int(rgb.group(1))

    try:
        rgb = re.compile(r'C1-(\d+)-(\d+)-(\d+)')
        rgb = rgb.search(fname_modelFolder)
        params['C1_3d'] = int(rgb.group(2))
    except:
        rgb = re.compile(r'C1-(\d+)-(\d+)')
        rgb = rgb.search(fname_modelFolder)
        params['C1_3d'] = int(0)
    params['C1_n'] = int(rgb.group(1))
    params['C1_s'] = int(rgb.group(2))
    
    try:
        rgb = re.compile(r'C2-(\d+)-(\d+)-(\d+)')
        rgb = rgb.search(fname_modelFolder)
        params['C2_3d'] = int(rgb.group(2))
    except:
        rgb = re.compile(r'C2-(\d+)-(\d+)')
        rgb = rgb.search(fname_modelFolder)
        params['C2_3d'] = int(0)
    params['C2_n'] = int(rgb.group(1))
    params['C2_s'] = int(rgb.group(2))
    
    try:
        rgb = re.compile(r'C3-(\d+)-(\d+)-(\d+)')
        rgb = rgb.search(fname_modelFolder)
        params['C3_3d'] = int(rgb.group(2))
    except:
        rgb = re.compile(r'C3-(\d+)-(\d+)')
        rgb = rgb.search(fname_modelFolder)
        params['C3_3d'] = int(0)
    params['C3_n'] = int(rgb.group(1))
    params['C3_s'] = int(rgb.group(2))
    
    try:
        rgb = re.compile(r'C4-(\d+)-(\d+)-(\d+)')
        rgb = rgb.search(fname_modelFolder)
        params['C4_3d'] = int(rgb.group(2))
    except:
        rgb = re.compile(r'C4-(\d+)-(\d+)')
        rgb = rgb.search(fname_modelFolder)
        params['C4_3d'] = int(0)
    if rgb==None:
        params['C4_n']=0
        params['C4_s'] = 0
        params['C4_3d'] = int(0)

    else:
        params['C4_n'] = int(rgb.group(1))
        params['C4_s'] = int(rgb.group(2))
        params['C4_3d'] = int(0)


    rgb = re.compile(r'BN-(\d+)')
    rgb = rgb.search(fname_modelFolder)
    params['BN'] = int(rgb.group(1))

    rgb = re.compile(r'MP-(\d+)')
    rgb = rgb.search(fname_modelFolder)
    params['MP'] = int(rgb.group(1))

    rgb = re.compile(r'TR-(\d+)')
    rgb = rgb.search(fname_modelFolder)
    params['TR'] = int(rgb.group(1))
    
    rgb = re.compile(r'TRSAMPS')
    rgb = rgb.search(fname_modelFolder)
    if rgb!=None:
        try:
            rgb = re.compile(r'TRSAMPS-(-?\d+)')
            rgb = rgb.search(fname_modelFolder)
            params['TRSAMPS'] = int(rgb.group(1))
        except:
            pass
    
    lr_token = re.compile(r'(LR)-')
    lr_token = lr_token.search(fname_modelFolder)
    if lr_token != None:
        rgb = re.compile(r'LR-(\d+\.\d+)')
        rgb = rgb.search(fname_modelFolder)
        if rgb == None:     # if it is in scientific notation
            rgb = re.compile(r'LR-([\d.eE+-]+)')
            rgb = rgb.search(fname_modelFolder)
        params['LR'] = float(rgb.group(1))
        
    return params

def getModelParams_old(fname_modelFolder):
    
    rgb = re.split('_',fname_modelFolder)
    
    p_U = float(re.findall("U-(\d+\.\d+)",rgb[0])[0])
    p_T = int(re.findall("T-(\d+)",rgb[1])[0])
    
    c1 = re.split('-',rgb[2])[1:]
    p_C1_n = int(c1[0])
    p_C1_s = int(c1[1])
    if len(c1)>2:
        p_C1_3d = int(c1[2])
    else:
        p_C1_3d = 0
    
    c2 = re.split('-',rgb[3])[1:]
    p_C2_n = int(c2[0])
    p_C2_s = int(c2[1])
    if len(c2)>2:
        p_C2_3d = int(c2[2])
    else:
        p_C2_3d = 0
    
    c3 = re.split('-',rgb[4])[1:]
    p_C3_n = int(c3[0])
    p_C3_s = int(c3[1])
    if len(c3)>2:
        p_C3_3d = int(c3[2])
    else:
        p_C3_3d = 0
        
    p_BN = int(re.findall("BN-(\d+)",rgb[5])[0])
    p_MP = int(re.findall("MP-(\d+)",rgb[6])[0])
    
    if len(rgb)>7:
        p_TR = int(re.findall("TR-(\d+)",rgb[7])[0])
    else:
        p_TR = 1
        
    
    params = {
        'U': p_U,
        'T': p_T,
        'C1_n': p_C1_n,
        'C1_s': p_C1_s,
        'C1_3d': p_C1_3d,
        'C2_n': p_C2_n,
        'C2_s': p_C2_s,
        'C2_3d': p_C2_3d,
        'C3_n': p_C3_n,
        'C3_s': p_C3_s,
        'C3_3d': p_C3_3d,  
        'BN': p_BN,
        'MP': p_MP,
        'TR': p_TR
        }
    
    if len(rgb)>8:
        p_TR = int(re.findall("TRSAMPS-(\d+)",rgb[8])[0])
        params['TRSAMPS'] = p_TR

    
    return params

def paramsToName(mdl_name,LR=None,U=0,P=0,T=60,C1_n=1,C1_s=1,C1_3d=0,C2_n=0,C2_s=0,C2_3d=0,C3_n=0,C3_s=0,C3_3d=0,BN=1,MP=0,TR=0):
    if mdl_name=='CNN_2D' or mdl_name=='replaceDense_2D':
        if LR==None:    # backwards compatibility
            paramFileName = 'U-%0.2f_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d' %(U,T,C1_n,C1_s,
                                                                                                         C2_n,C2_s,
                                                                                                         C3_n,C3_s,
                                                                                                         BN,MP)
        else:
            paramFileName = 'U-%0.2f_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_LR-%0.4f' %(U,T,C1_n,C1_s,
                                                                                                         C2_n,C2_s,
                                                                                                         C3_n,C3_s,
                                                                                                         BN,MP,LR)
            
        
    elif mdl_name[:8]=='PR_CNN2D' or mdl_name[:8]=='PR_CNN3D' or mdl_name[:10]=='PRFR_CNN2D' or mdl_name[:8]=='BP_CNN2D':
        if LR==None:    # backwards compatibility
            paramFileName = 'U-%0.2f_P-%03d_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d' %(U,P,T,C1_n,C1_s,
                                                                                                             C2_n,C2_s,
                                                                                                             C3_n,C3_s,
                                                                                                             BN,MP)   
        else:
            paramFileName = 'U-%0.2f_P-%03d_T-%03d_C1-%02d-%02d_C2-%02d-%02d_C3-%02d-%02d_BN-%d_MP-%d_LR-%0.4f' %(U,P,T,C1_n,C1_s,
                                                                                                                 C2_n,C2_s,
                                                                                                                 C3_n,C3_s,
                                                                                                                 BN,MP,LR)       
    else:
        paramFileName = 'U-%0.2f_T-%03d_C1-%02d-%02d-%02d_C2-%02d-%02d-%02d_C3-%02d-%02d-%02d_BN-%d_MP-%d' %(U,T,C1_n,C1_s,C1_3d,
                                                                                                             C2_n,C2_s,C2_3d,
                                                                                                             C3_n,C3_s,C3_3d,
                                                                                                             BN,MP)
    
    return paramFileName
    
    

def model_evaluate(obs_rate_allStimTrials,pred_rate,filt_temporal_width,RR_ONLY=False,lag = 0):
    numCells = obs_rate_allStimTrials.shape[-1]
    num_trials = obs_rate_allStimTrials.shape[0]
    idx_allTrials = np.arange(num_trials)
    
    if lag != 0:
        obs_rate_allStimTrials_corrected = obs_rate_allStimTrials[:,:-lag,:]
        if RR_ONLY is False:
            pred_rate_corrected = pred_rate[lag:,:]
    else:
        obs_rate_allStimTrials_corrected = obs_rate_allStimTrials
        if RR_ONLY is False:
            pred_rate_corrected = pred_rate
        
    

# for predicting trial averaged responses

    idx_trials_r1 = np.array(random.sample(range(0,len(idx_allTrials)),int(np.ceil(len(idx_allTrials)/2))))
    assert(np.unique(idx_trials_r1).shape[0] == idx_trials_r1.shape[0])
    idx_trials_r2 = np.setdiff1d(idx_allTrials,idx_trials_r1)

    r1 = np.mean(obs_rate_allStimTrials_corrected[idx_trials_r1,filt_temporal_width:,:],axis=0)
    r2 = np.mean(obs_rate_allStimTrials_corrected[idx_trials_r2,filt_temporal_width:,:],axis=0)
    
    noise_trialAveraged = np.mean((r1-r2)**2,axis=0)
    fracExplainableVar = (np.var(r2,axis=0) - noise_trialAveraged)/np.var(r2,axis=0)
    
    if RR_ONLY is True:
        fev = None
    else:
        r_pred = pred_rate_corrected
        mse_resid = np.mean((r_pred-r2)**2,axis=0)
        fev = 1 - ((mse_resid-noise_trialAveraged)/(np.var(r2,axis=0)-noise_trialAveraged))
    
    
# Pearson correlation
    rr_corr = correlation_coefficient_distribution(r1,r2)
    if RR_ONLY is True:
        pred_corr = None
    else:
        pred_corr = correlation_coefficient_distribution(r2,r_pred)
    

    return fev, fracExplainableVar, pred_corr, rr_corr


def estimate_noise(resp,n_loops=100):
    if resp.ndim<3:
        obs_noise=0
    else:
        n_trials = resp.shape[-1]
        idx_valid = ~np.isnan(resp[:,0,0])
        if idx_valid.sum()==0:
            obs_noise=0
        else:
            resp_valid = resp[idx_valid]
            resp_valid = np.moveaxis(resp_valid,-1,0)
            
            noise_trialAveraged = np.zeros((n_loops,resp_valid.shape[-1]))
            for i in range(0,n_loops):
                idx_allTrials = np.arange(n_trials)
                idx_trials_r1 = np.array(random.sample(range(0,n_trials),int(np.ceil(n_trials/2))))
                assert(np.unique(idx_trials_r1).shape[0] == idx_trials_r1.shape[0])
                idx_trials_r2 = np.setdiff1d(idx_allTrials,idx_trials_r1)
    
                r1 = np.mean(resp_valid[idx_trials_r1,:,:],axis=0)
                r2 = np.mean(resp_valid[idx_trials_r2,:,:],axis=0)
                
                noise_trialAveraged[i] = np.mean((r1-r2)**2,axis=0)
                
            obs_noise = np.mean(noise_trialAveraged,axis=0)
    return obs_noise
                
        

def model_evaluate_new(obs_rate_allStimTrials,pred_rate,filt_width,RR_ONLY=False,lag = 0,obs_noise=0,resp_format='INDIVIDUAL'):
    numCells = obs_rate_allStimTrials.shape[-1]
    if resp_format=='INDIVIDUAL':
        if obs_rate_allStimTrials.ndim>2:
            num_trials = obs_rate_allStimTrials.shape[0]
            idx_allTrials = np.arange(num_trials)
        else:
            num_trials = 1
    else:
        num_trials = 1
    
    if num_trials > 1:  # for kierstens data or where we have multiple trials of the validation data
    
        t_start = 20
        
        # obs_rate_allStimTrials_corrected = obs_rate_allStimTrials[:,filt_width:,:]
        obs_rate_allStimTrials_corrected = obs_rate_allStimTrials
        t_end = obs_rate_allStimTrials_corrected.shape[1]-t_start-20
        obs_rate_allStimTrials_corrected = obs_rate_allStimTrials_corrected[:,t_start:t_end-lag,:]
        
        # if RR_ONLY is False:
        pred_rate_corrected = pred_rate[t_start+lag:t_end,:]
        
        
        # for predicting trial averaged responses
        idx_trials_r1 = np.array(random.sample(range(0,len(idx_allTrials)),int(np.ceil(len(idx_allTrials)/2))))
        assert(np.unique(idx_trials_r1).shape[0] == idx_trials_r1.shape[0])
        idx_trials_r2 = np.setdiff1d(idx_allTrials,idx_trials_r1)
    
        r1 = np.mean(obs_rate_allStimTrials_corrected[idx_trials_r1,:,:],axis=0)
        r2 = np.mean(obs_rate_allStimTrials_corrected[idx_trials_r2,:,:],axis=0)
        
        if obs_noise!=None:
            noise_trialAveraged = np.mean((r1-r2)**2,axis=0)
            
        fracExplainableVar = (np.var(r2,axis=0) - noise_trialAveraged)/(np.var(r2,axis=0)+1e-5)
        
        if RR_ONLY is True:
            fev = None
        else:
            r_pred = pred_rate_corrected
            mse_resid = np.mean((r_pred-r2)**2,axis=0)
            fev = 1 - ((mse_resid-noise_trialAveraged)/(np.var(r2,axis=0)-noise_trialAveraged))
            # fev = 1 - ((mse_resid-noise_trialAveraged)/(np.var(obs_rate_allStimTrials_corrected[idx_trials_r2,:,:],axis=(0,1))-noise_trialAveraged))
            # fev = 1 - ((mse_resid)/(np.var(r2,axis=0)-noise_trialAveraged))
        
        
        # Pearson correlation
        rr_corr = correlation_coefficient_distribution(r1,r2)
        if RR_ONLY is True:
            pred_corr = None
        else:
            pred_corr = correlation_coefficient_distribution(r2,r_pred)
            
    else:
        t_start = 0
        
        obs_rate_allStimTrials_corrected = obs_rate_allStimTrials
        t_end = obs_rate_allStimTrials_corrected.shape[0]
        obs_rate_allStimTrials_corrected = obs_rate_allStimTrials_corrected[t_start:t_end-lag,:]
        
        # if RR_ONLY is False:
        pred_rate_corrected = pred_rate[t_start+lag:t_end,:]

        resid = obs_rate_allStimTrials_corrected - pred_rate_corrected
        mse_resid = np.mean(resid**2,axis=0)
        var_test = np.var(obs_rate_allStimTrials_corrected,axis=0)
        fev = 1 - ((mse_resid - obs_noise)/(var_test+1e-5-obs_noise))
        fracExplainableVar = None #(var_test-obs_noise)/var_test
        
        pred_corr = correlation_coefficient_distribution(obs_rate_allStimTrials_corrected,pred_rate_corrected)
        rr_corr = None
   

    return fev, fracExplainableVar, pred_corr, rr_corr
"""
FUEV can be >1 if the variance is too small like in case of sparse activitiy
FUEV can be <0 if noise is higher than the variance which can again happen in case of sparse activity
"""

# fig,axs = plt.subplots(1,1)
# axs.plot(r2[:,0])
# axs.plot(r_pred[:,0])

def correlation_coefficient_distribution(obs_rate,est_rate):
    x_mu = obs_rate - np.mean(obs_rate, axis=0)
    x_std = np.std(obs_rate, axis=0)
    y_mu = est_rate - np.mean(est_rate, axis=0)
    y_std = np.std(est_rate, axis=0)
    cc_allUnits = np.mean(x_mu * y_mu,axis=0) / (x_std * y_std+1e-6)
    
    return cc_allUnits

# # for predicting single trial responses
   
#     with mp.Pool(processes=3) as pool:
#         result = pool.starmap(partial(parallel_singleTrialEstimates),[(obs_rate_allStimTrials,pred_rate,filt_temporal_width,i) for i in idx_allTrials])
    
                              
#     noise_trialSingle_all_avg = np.mean(np.array([item[0] for item in result]),axis=0)
#     var_trialSingle_all_avg = np.mean(np.array([item[1] for item in result]),axis=0)
#     mse_trialSingle_all_avg = np.mean(np.array([item[2] for item in result]),axis=0)
        
#     fractExplainableVar_2 = (var_trialSingle_all_avg - noise_trialSingle_all_avg)/var_trialSingle_all_avg
#     fev_2 = 1 - ((mse_trialSingle_all_avg - noise_trialSingle_all_avg)/(var_trialSingle_all_avg-noise_trialSingle_all_avg))
       
    
#     return fev_1a,fev_1b,fev_2



def parallel_singleTrialEstimates(obs_rate_allStimTrials,pred_rate,filt_temporal_width,idx_trialToIsolate):
    num_trials = obs_rate_allStimTrials.shape[0]
    idx_allTrials = np.arange(num_trials)
    
    idx_trialsToAvg = np.setdiff1d(idx_allTrials,idx_trialToIsolate)
    r1 = obs_rate_allStimTrials[idx_trialToIsolate,filt_temporal_width:,:]
    r2 = np.nanmean(obs_rate_allStimTrials[idx_trialsToAvg,filt_temporal_width:,:],axis=0)
    
    noise_trialSingle = np.mean((r1-r2)**2,axis=0)
    
    var_trialSingle = np.mean((r1 - np.mean(r1,axis=0))**2,axis=0)
    
    mse_trialSingle = np.mean((pred_rate - r1)**2,axis=0)
        
    return noise_trialSingle,var_trialSingle,mse_trialSingle



    # noise_trialSingle_all = np.zeros((num_trials,numCells))
    # var_trialSingle_all = np.zeros((num_trials,numCells))
    # mse_trialSingle_all = np.zeros((num_trials,numCells))
    
    # # first estimate the average noise, MSE and variances

    # for idx_trialToIsolate in idx_allTrials:
    #     idx_trialsToAvg = np.setdiff1d(idx_allTrials,idx_trialToIsolate)
    #     r1 = obs_rate_allStimTrials[idx_trialToIsolate,filt_temporal_width:,:]
    #     r2 = np.nanmean(obs_rate_allStimTrials[idx_trialsToAvg,filt_temporal_width:,:],axis=0)
        
    #     noise_trialSingle = np.mean((r1-r2)**2,axis=0)
    #     noise_trialSingle_all[idx_trialToIsolate,:] = noise_trialSingle
        
    #     var_trialSingle = np.mean((r1 - np.mean(r1,axis=0))**2,axis=0)
    #     var_trialSingle_all[idx_trialToIsolate,:] = var_trialSingle
        
    #     mse_trialSingle = np.mean((pred_rate - r1)**2,axis=0)
    #     mse_trialSingle_all[idx_trialToIsolate,:] = mse_trialSingle
        
               
    # noise_trialSingle_all_avg = np.mean(noise_trialSingle_all,axis=0)
    # var_trialSingle_all_avg = np.mean(var_trialSingle_all,axis=0)
    # mse_trialSingle_all_avg = np.mean(mse_trialSingle_all,axis=0)
    
    # fractExplainableVar_2 = (var_trialSingle_all_avg - noise_trialSingle_all_avg)/var_trialSingle_all_avg
    # fev_2 = 1 - ((mse_trialSingle_all_avg - noise_trialSingle_all_avg)/(var_trialSingle_all_avg-noise_trialSingle_all_avg))


# def newFEV(noise_allStimTrials,fracExplainableVar_allStimTrials,obs_rate_allStimTrials,pred_rate,filt_temporal_width):
#     # Retinal reliability method 2
#     num_trials = obs_rate_allStimTrials.shape[0]
#     idx_allTrials = np.arange(num_trials)

#     numCells = pred_rate.shape[1]
#     fev_allUnits_allStimTrials_m2 = np.zeros((num_trials,numCells))
    
#     for idx_trialToIsolate in range(num_trials):
#         idx_trialsToAvg = np.setdiff1d(idx_allTrials,idx_trialToIsolate)
#         rgb1 = np.nanmean(obs_rate_allStimTrials[idx_trialsToAvg,:,:],axis=0)
#         obs_rate = rgb1[filt_temporal_width:,:]
        
#         resid = obs_rate - pred_rate
#         mse_resid = np.mean(resid**2,axis=0)
        
#         fev = 1 - ((mse_resid - noise_allStimTrials[idx_trialToIsolate,:]) / fracExplainableVar_allStimTrials[idx_trialToIsolate,:])
        
#         fev_allUnits_allStimTrials_m2[idx_trialToIsolate,:] = fev
        
#     fev_allUnits_m2 = np.mean(fev_allUnits_allStimTrials_m2,axis=0)
    
#     # method 3
#     fev_allUnits_allStimTrials_m3 = np.zeros((num_trials,numCells))
#     for idx_trialToIsolate in range(num_trials):
#         obs_rate = obs_rate_allStimTrials[idx_trialToIsolate,:,:]
#         obs_rate = obs_rate[filt_temporal_width:,:]
#         resid = obs_rate - pred_rate
#         mse_resid = np.mean(resid**2,axis=0)
#         noise = noise_allStimTrials[idx_trialToIsolate,:]
        
#         resp_all = np.concatenate((pred_rate,obs_rate),axis=0)
        
#         fev = 1 - ((mse_resid - noise_allStimTrials[idx_trialToIsolate,:]) / (np.var(resp_all)-noise))
        
#         fev_allUnits_allStimTrials_m3[idx_trialToIsolate,:] = fev
    
#     fev_allUnits_m3 = np.mean(fev_allUnits_allStimTrials_m3,axis=0)
    
    
#     return fev_allUnits_m2, fev_allUnits_m3
        
        
