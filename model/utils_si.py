#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 12:34:40 2021

@author: saad
"""

import numpy as np
import math

def round_to_even(n):
    rounded = int(n // 1)  # Get the integer part (floor)
    if rounded % 2 == 1:   # If the integer is odd
        return rounded + 1 # Always round up to the next even number
    return rounded         # If already even, return as is

def h5_tostring(arr):
    new_arr = np.empty(arr.shape[0],dtype='object')
    for i in range(arr.shape[0]):
        rgb = arr[i].tostring().decode('ascii')
        rgb = rgb.replace("\x00","")
        new_arr[i] = rgb

    return (new_arr)
    
def splitall(path):
    import os, sys

    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

        

def modelFileName(U=0,P=0,T=0,CB_n=0,C1_n=0,C1_s=0,C1_3d=0,C2_n=0,C2_s=0,C2_3d=0,C3_n=0,C3_s=0,C3_3d=0,C4_n=0,C4_s=0,C4_3d=0,BN=0,MP=0,LR=0,TR=0,with_TR=True,TRSAMPS=0,with_TRSAMPS=True):
    
    """
    Takes in data and model parameters, and parses them to 
    make the foldername where the model will be saved
    U : unit quality threshold
    P : Temporal dimension for the photoreceptor layer
    T : Temporal dimension for the first CNN
    C1, C2, C3 : CNN layers
    C_n : Number of channels in that CNN layer
    C_s : Convolution filter size (widht = height)
    C_3d : Size of the filter's third dimension in case CNNs are 3D Conv
    BN : BatchNormalization layer after each CNN (1=ON, 0=OFF)
    MP : MaxPool layer after first CNN (1=ON, 0=OFF)
    LR : Initial learning rate
    TR : Trial Number
    with_TR : Return filename with or without TR
    """
    
    def parse_param(key,val,fname):
        fname = fname+key+'-'+val+'_'    
        return fname

    fname = ''
    dict_params = {}
    
    if U>0.9:
        U = '%d'%U
    else:
        U = '%0.2f'%U
    fname = parse_param('U',U,fname)    
    
    if P>0:
        P = '%03d' %P
        fname = parse_param('P',P,fname)    
    
    T = '%03d' %T
    fname = parse_param('T',T,fname)    
    
    if CB_n>0:
        CB = '%02d'%CB_n
        dict_params['chans_bp'] = CB_n
        key = 'CB'
        fname = fname+key+'-'+eval(key)+'_'
    
    if C1_3d>0:
        C1 = '%02d-%02d-%02d'%(C1_n,C1_s,C1_3d)
        dict_params['filt1_3rdDim'] = C1_3d
    else:
        C1 = '%02d-%02d'%(C1_n,C1_s)
        
    dict_params['chan1_n'] = C1_n
    dict_params['filt1_size'] = C1_s
    key = 'C1'
    fname = fname+key+'-'+eval(key)+'_'    

    if C2_n>0:
        if C2_3d>0:
            C2 = '%02d-%02d-%02d'%(C2_n,C2_s,C2_3d)
            dict_params['filt2_3rdDim'] = C2_3d
        else:
            C2 = '%02d-%02d'%(C2_n,C2_s)
        
        key = 'C2'
        fname = fname+key+'-'+eval(key)+'_'    
    dict_params['chan2_n'] = C2_n
    dict_params['filt2_size'] = C2_s

        
    if C2_n>0 and C3_n>0:
        if C3_3d>0:
            C3 = '%02d-%02d-%02d'%(C3_n,C3_s,C3_3d)
            dict_params['filt3_3rdDim'] = C3_3d
        else:
            C3 = '%02d-%02d'%(C3_n,C3_s)
            
        key = 'C3'
        fname = fname+key+'-'+eval(key)+'_'    
    dict_params['chan3_n'] = C3_n
    dict_params['filt3_size'] = C3_s
    
    if C2_n>0 and C3_n>0 and C4_n>0:
        if C4_3d>0:
            C4 = '%02d-%02d-%02d'%(C4_n,C4_s,C4_3d)
            dict_params['filt4_3rdDim'] = C3_3d
        else:
            C4 = '%02d-%02d'%(C4_n,C4_s)
            
        key = 'C4'
        fname = fname+key+'-'+eval(key)+'_'    
    dict_params['chan4_n'] = C4_n
    dict_params['filt4_size'] = C4_s

            
    BN = '%d'%BN
    fname = parse_param('BN',BN,fname)    
    dict_params['BatchNorm'] = int(BN)

    MP = '%d'%MP
    fname = parse_param('MP',MP,fname)    
    dict_params['MaxPool'] = int(MP)
    
    # if LR>0:
    # LR = '%0.4f'%LR
    LR = str(LR)
    fname = parse_param('LR',LR,fname)    
        
        
    if with_TRSAMPS==True:
        if TRSAMPS<1:
            TRSAMPS = '%0.2f'%TRSAMPS
        else:
            TRSAMPS = '%03d'%TRSAMPS
        fname = parse_param('TRSAMPS',TRSAMPS,fname)    

    if with_TR==True:
        TR = '%02d'%TR
        fname = parse_param('TR',TR,fname)    
        
    fname_model = fname[:-1]
    
    
    return fname_model,dict_params
