#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:05:08 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""


import jax
from jax import numpy as jnp
from flax import linen as nn
import re
from jax.nn.initializers import glorot_uniform, he_normal
import jax.random as random



def model_definitions():
    """
        How to arrange the datasets depends on which model is being used
    """
    
    models_2D = ('CNN2D','CNN2D_MAXPOOL','CNN2D_FT','CNN2D_FT2','CNN2D_LNORM','CNN2D_MAP','CNN2D_MAP2','CNN2D_MAP3',
                 'PRFR_CNN2D_MAP','PRFR_CNN2D_MAP2',
                 'CNN2D_MAP3_FT','PRFR_CNN2D_MAP_FT')
    
    models_3D = ('CNN_3D','PR_CNN3D')
    
    return (models_2D,models_3D)

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
        if rgb is not None:
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


def model_summary(mdl,inp_shape,console_kwargs={'width':180}):
    from flax.linen import summary

    inputs = jnp.ones([1]+list(inp_shape))    
    tabulate_fn = nn.tabulate(mdl, jax.random.PRNGKey(0),console_kwargs=console_kwargs)
    print(tabulate_fn(inputs,training=False))


def dict_subset(old_dict,exclude_list):
    new_dict = {}
    keys_all = list(old_dict.keys())
    for item in keys_all:
        for key_exclude in exclude_list:
            rgb = re.search(key_exclude,item,re.IGNORECASE)
            if rgb==None:
                new_dict[item] = old_dict[item]
    return new_dict


def get_exactLayers(params_dict,layer_list):
    layer_names = []
    keys_all = list(params_dict.keys())
    for item in keys_all:
        for key in layer_list:
            rgb = re.search(key,item,re.IGNORECASE)
            if rgb!=None:
                layer_names.append(item)
    return layer_names



def transfer_weights(mdl_source,mdl_target,fixedLayer='Dense'):
    params_subset = dict_subset(mdl_source.params,fixedLayer)
    
    for param_name in params_subset.keys():
        mdl_target.params[param_name] = mdl_source.params[param_name]
        
    return mdl_target

def he_normal_arr(key, shape):
    fan_in = shape[0] if len(shape) == 2 else jnp.prod(shape[:-1])
    stddev = jnp.sqrt(2.0 / fan_in)
    normal_samples = jax.random.normal(key, shape)
    return normal_samples * stddev



class InstanceNorm(nn.Module):

    def __call__(self, x):
        mean = jnp.mean(x, axis=0)
        std = jnp.std(x, axis=0) + 1e-6  # Adding a small value to avoid division by zero
        return (x - mean) / std


class CNN2D(nn.Module):
    
    chan1_n : int
    filt1_size : int
    chan2_n : int
    filt2_size : int
    chan3_n : int
    filt3_size : int
    chan4_n : int
    filt4_size : int
    nout : int    
    filt_temporal_width : int    
    BatchNorm : bool
    MaxPool : int
    # dtype : type
    # def __init__(self, **kwargs):
    #     self.__dict__.update(kwargs)

    @nn.compact
    def __call__(self,inputs,training: bool,**kwargs):       
        sigma=0.1       # sigma for noise
        y = jnp.moveaxis(inputs,1,-1)
        y = nn.LayerNorm(feature_axes=-1,reduction_axes=-1,epsilon=1e-7)(y)        # z-score the input across temporal dimension
        y = nn.Conv(features=self.chan1_n, kernel_size=(self.filt1_size,self.filt1_size),padding='VALID')(y)
        
        if self.MaxPool > 0:
            y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

        if self.BatchNorm == 1:
            y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)

        y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
        y = nn.relu(y)
        
        # second layer
        if self.chan2_n>0:
            y = nn.Conv(features=self.chan2_n, kernel_size=(self.filt2_size,self.filt2_size),padding='VALID')(y)
            if self.BatchNorm == 1:
                y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)
            y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            y = nn.relu(y)

        # Third layer
        if self.chan3_n>0:
            y = nn.Conv(features=self.chan3_n, kernel_size=(self.filt3_size,self.filt3_size),padding='VALID')(y)
            if self.BatchNorm == 1:
                y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)
            y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            y = nn.relu(y)
            
        if self.chan4_n>0:
            y = nn.Conv(features=self.chan4_n, kernel_size=(self.filt4_size,self.filt4_size),padding='VALID')(y)
            if self.BatchNorm == 1:
                y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)
            y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            y = nn.relu(y)


                        
        y = y.reshape(y.shape[0],-1)
        y = nn.Dense(features=self.nout)(y)
        outputs = nn.softplus(y)
        
        return outputs

    

class CNN2D_MAXPOOL(nn.Module):
    
    chan1_n : int
    filt1_size : int
    chan2_n : int
    filt2_size : int
    chan3_n : int
    filt3_size : int
    chan4_n : int
    filt4_size : int
    nout : int    
    filt_temporal_width : int    
    BatchNorm : bool
    MaxPool : int
    # dtype : type
    
    # def __init__(self, **kwargs):
    #     self.__dict__.update(kwargs)

    @nn.compact
    def __call__(self,inputs,training: bool,**kwargs):       
        # training = True
        sigma=0.01       # sigma for noise
        y = jnp.moveaxis(inputs,1,-1)       # Because jax is channels last
        # y = nn.LayerNorm(feature_axes=-1,reduction_axes=-1,epsilon=1e-7)(y)        # z-score the input across temporal dimension
        y = nn.Conv(features=self.chan1_n, kernel_size=(self.filt1_size,self.filt1_size),padding='VALID', kernel_init=glorot_uniform())(y)
        
        if self.MaxPool > 0:
            y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

        if self.BatchNorm == 1:
            rgb = y.shape[1:]
            y = y.reshape(y.shape[0],-1)
            y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)
            y = y.reshape(y.shape[0],*rgb)

            # y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)

        # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
        y = nn.relu(y)
        
        # second layer
        if self.chan2_n>0:
            y = nn.Conv(features=self.chan2_n, kernel_size=(self.filt2_size,self.filt2_size),padding='VALID', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)
                y = y.reshape(y.shape[0],*rgb)

            # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            y = nn.relu(y)

        # Third layer
        if self.chan3_n>0:
            y = nn.Conv(features=self.chan3_n, kernel_size=(self.filt3_size,self.filt3_size),padding='VALID', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)
                y = y.reshape(y.shape[0],*rgb)

            # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            y = nn.relu(y)
            
        if self.chan4_n>0:
            y = nn.Conv(features=self.chan4_n, kernel_size=(self.filt4_size,self.filt4_size),padding='VALID', kernel_init=glorot_uniform())(y)
           
            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)
                y = y.reshape(y.shape[0],*rgb)
                
            # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            y = nn.relu(y)


                        
        y = y.reshape(y.shape[0],-1)
        # y = InstanceNorm()(y)
        y = nn.LayerNorm()(y)
        y = nn.Dense(features=self.nout,kernel_init=he_normal())(y)
        self.sow('intermediates', 'dense_activations', y)
        y = nn.LayerNorm()(y)
        outputs = nn.softplus(y)
        
        return outputs
    
class CNN2D_LNORM(nn.Module):
    
    chan1_n : int
    filt1_size : int
    chan2_n : int
    filt2_size : int
    chan3_n : int
    filt3_size : int
    chan4_n : int
    filt4_size : int
    nout : int    
    filt_temporal_width : int    
    BatchNorm : bool
    MaxPool : int
    # dtype : type
    
    # def __init__(self, **kwargs):
    #     self.__dict__.update(kwargs)

    @nn.compact
    def __call__(self,inputs,training: bool,**kwargs):       
        sigma=0.01       # sigma for noise
        y = jnp.moveaxis(inputs,1,-1)       # Because jax is channels last
        # y = nn.LayerNorm(feature_axes=-1,reduction_axes=-1,epsilon=1e-7)(y)        # z-score the input across temporal dimension
        y = nn.Conv(features=self.chan1_n, kernel_size=(self.filt1_size,self.filt1_size),padding='VALID', kernel_init=glorot_uniform())(y)
        
        if self.MaxPool > 0:
            y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

        if self.BatchNorm == 1:
            rgb = y.shape[1:]
            y = y.reshape(y.shape[0],-1)
            y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
            y = y.reshape(y.shape[0],*rgb)

            # y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)

        # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
        y = nn.relu(y)
        
        # second layer
        if self.chan2_n>0:
            y = nn.Conv(features=self.chan2_n, kernel_size=(self.filt2_size,self.filt2_size),padding='VALID', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
                y = y.reshape(y.shape[0],*rgb)

            # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            y = nn.relu(y)

        # Third layer
        if self.chan3_n>0:
            y = nn.Conv(features=self.chan3_n, kernel_size=(self.filt3_size,self.filt3_size),padding='VALID', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
                y = y.reshape(y.shape[0],*rgb)

            # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            y = nn.relu(y)
            
        if self.chan4_n>0:
            y = nn.Conv(features=self.chan4_n, kernel_size=(self.filt4_size,self.filt4_size),padding='VALID', kernel_init=glorot_uniform())(y)
           
            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
                y = y.reshape(y.shape[0],*rgb)
                
            # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            y = nn.relu(y)


                        
        y = y.reshape(y.shape[0],-1)
        y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
        y = nn.Dense(features=self.nout,kernel_init=he_normal())(y)
        self.sow('intermediates', 'dense_activations', y)
        outputs = nn.softplus(y)
        
        return outputs    
    
class CNN2D_MAP(nn.Module):
    
    chan1_n : int
    filt1_size : int
    chan2_n : int
    filt2_size : int
    chan3_n : int
    filt3_size : int
    chan4_n : int
    filt4_size : int
    nout : int    
    filt_temporal_width : int    
    BatchNorm : bool
    MaxPool : int
    # dtype : type
    
    # def __init__(self, **kwargs):
    #     self.__dict__.update(kwargs)

    @nn.compact
    def __call__(self,inputs,training: bool,**kwargs):       
        sigma=0.01       # sigma for noise
        y = jnp.moveaxis(inputs,1,-1)       # Because jax is channels last
        # y = nn.LayerNorm(feature_axes=-1,reduction_axes=-1,epsilon=1e-7)(y)        # z-score the input across temporal dimension
        y = nn.Conv(features=self.chan1_n, kernel_size=(self.filt1_size,self.filt1_size),padding='SAME', kernel_init=glorot_uniform())(y)
        
        if self.MaxPool > 0:
            y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

        if self.BatchNorm == 1:
            rgb = y.shape[1:]
            y = y.reshape(y.shape[0],-1)
            y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
            y = y.reshape(y.shape[0],*rgb)

            # y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)

        # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
        # y = nn.relu(y)
        y = TrainableAF()(y)

        
        # second layer
        if self.chan2_n>0:
            y = nn.Conv(features=self.chan2_n, kernel_size=(self.filt2_size,self.filt2_size),padding='SAME', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
                y = y.reshape(y.shape[0],*rgb)

            # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            # y = nn.relu(y)
            y = TrainableAF()(y)


        # Third layer
        if self.chan3_n>0:
            y = nn.Conv(features=self.chan3_n, kernel_size=(self.filt3_size,self.filt3_size),padding='SAME', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
                y = y.reshape(y.shape[0],*rgb)

            # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            # y = nn.relu(y)
            y = TrainableAF()(y)

            
        if self.chan4_n>0:
            y = nn.Conv(features=self.chan4_n, kernel_size=(self.filt4_size,self.filt4_size),padding='SAME', kernel_init=glorot_uniform())(y)
           
            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
                y = y.reshape(y.shape[0],*rgb)
                
            # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            # y = nn.relu(y)
            y = TrainableAF()(y)


        
        rgb = y.shape[1:]
        y = y.reshape(y.shape[0],-1)
        y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
        y = y.reshape(y.shape[0],*rgb)

        y = nn.Conv(features=self.nout, kernel_size=(1,1),padding='SAME', kernel_init=he_normal(),name='output')(y)
        outputs = nn.softplus(y)
        self.sow('intermediates', 'dense_activations', outputs)

        return outputs        
    
    
    
class CNN2D_MAP2(nn.Module):
    
    chan1_n : int
    filt1_size : int
    chan2_n : int
    filt2_size : int
    chan3_n : int
    filt3_size : int
    chan4_n : int
    filt4_size : int
    nout : int    
    filt_temporal_width : int    
    BatchNorm : bool
    MaxPool : int
    # dtype : type
    
    # def __init__(self, **kwargs):
    #     self.__dict__.update(kwargs)

    @nn.compact
    def __call__(self,inputs,training: bool,**kwargs):       
        sigma=0.01       # sigma for noise
        y = jnp.moveaxis(inputs,1,-1)       # Because jax is channels last
        # y = nn.LayerNorm(feature_axes=-1,reduction_axes=-1,epsilon=1e-7)(y)        # z-score the input across temporal dimension
        y = nn.Conv(features=self.chan1_n, kernel_size=(self.filt1_size,self.filt1_size),padding='SAME', kernel_init=glorot_uniform())(y)
        
        if self.MaxPool > 0:
            y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

        if self.BatchNorm == 1:
            # rgb = y.shape[1:]
            # y = y.reshape(y.shape[0],-1)
            y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
            # y = y.reshape(y.shape[0],*rgb)

            # y = nn.BatchNorm(axis=-1,epsilon=1e-7,use_running_average=not training)(y)

        # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
        # y = nn.relu(y)
        y = TrainableAF()(y)

        
        # second layer
        if self.chan2_n>0:
            y = nn.Conv(features=self.chan2_n, kernel_size=(self.filt2_size,self.filt2_size),padding='SAME', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

            if self.BatchNorm == 1:
                # rgb = y.shape[1:]
                # y = y.reshape(y.shape[0],-1)
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
                # y = y.reshape(y.shape[0],*rgb)

            # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            # y = nn.relu(y)
            y = TrainableAF()(y)


        # Third layer
        if self.chan3_n>0:
            y = nn.Conv(features=self.chan3_n, kernel_size=(self.filt3_size,self.filt3_size),padding='SAME', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

            if self.BatchNorm == 1:
                # rgb = y.shape[1:]
                # y = y.reshape(y.shape[0],-1)
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
                # y = y.reshape(y.shape[0],*rgb)

            # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            # y = nn.relu(y)
            y = TrainableAF()(y)

            
        if self.chan4_n>0:
            y = nn.Conv(features=self.chan4_n, kernel_size=(self.filt4_size,self.filt4_size),padding='SAME', kernel_init=glorot_uniform())(y)
           
            if self.BatchNorm == 1:
                # rgb = y.shape[1:]
                # y = y.reshape(y.shape[0],-1)
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
                # y = y.reshape(y.shape[0],*rgb)
                
            # y = y + sigma*jax.random.normal(jax.random.PRNGKey(1),y.shape)
            # y = nn.relu(y)
            y = TrainableAF()(y)


        
        # rgb = y.shape[1:]
        # y = y.reshape(y.shape[0],-1)
        y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
        # y = y.reshape(y.shape[0],*rgb)

        y = nn.Conv(features=self.nout, kernel_size=(1,1),padding='SAME', kernel_init=he_normal(),name='output')(y)
        
        # rgb = y.shape[1:]
        # y = y.reshape(y.shape[0],-1)
        y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
        # y = y.reshape(y.shape[0],*rgb)

        # outputs = nn.softplus(y)
        outputs = TrainableAF()(y)
        # outputs = nn.relu(outputs)

        self.sow('intermediates', 'dense_activations', outputs)

        return outputs 
    
    
class CNN2D_MAP3(nn.Module):
    
    chan1_n : int
    filt1_size : int
    chan2_n : int
    filt2_size : int
    chan3_n : int
    filt3_size : int
    chan4_n : int
    filt4_size : int
    nout : int    
    filt_temporal_width : int    
    BatchNorm : bool
    MaxPool : int
    # dtype : type
    
    # def __init__(self, **kwargs):
    #     self.__dict__.update(kwargs)

    @nn.compact
    def __call__(self,inputs,training: bool,rng=None,**kwargs):       
        y = jnp.moveaxis(inputs,1,-1)       # Because jax is channels last
        y = nn.Conv(features=self.chan1_n, kernel_size=(self.filt1_size,self.filt1_size),padding='SAME', kernel_init=glorot_uniform())(y)
        
        
        if self.MaxPool > 0:
            y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

        if self.BatchNorm == 1:
            y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
        y = TrainableAF()(y)

        
        # second layer
        if self.chan2_n>0:
            y = nn.Conv(features=self.chan2_n, kernel_size=(self.filt2_size,self.filt2_size),padding='SAME', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

            if self.BatchNorm == 1:
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
            y = TrainableAF()(y)


        # Third layer
        if self.chan3_n>0:
            y = nn.Conv(features=self.chan3_n, kernel_size=(self.filt3_size,self.filt3_size),padding='SAME', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

            if self.BatchNorm == 1:
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)

            y = TrainableAF()(y)

            
        if self.chan4_n>0:
            y = nn.Conv(features=self.chan4_n, kernel_size=(self.filt4_size,self.filt4_size),padding='SAME', kernel_init=glorot_uniform())(y)
           
            if self.BatchNorm == 1:
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
                
            y = TrainableAF()(y)
        
        y = nn.Conv(features=self.nout, kernel_size=(1,1),padding='SAME', kernel_init=he_normal(),name='output')(y)
        
        y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)

        outputs = TrainableAF()(y)

        self.sow('intermediates', 'dense_activations', outputs)

        return outputs    
        
class CNN2D_MAPN(nn.Module):
    
    chan1_n : int
    filt1_size : int
    chan2_n : int
    filt2_size : int
    chan3_n : int
    filt3_size : int
    chan4_n : int
    filt4_size : int
    nout : int    
    filt_temporal_width : int    
    BatchNorm : bool
    MaxPool : int
    # dtype : type
    
    # def __init__(self, **kwargs):
    #     self.__dict__.update(kwargs)

    @nn.compact
    def __call__(self,inputs,training: bool,rng=None,**kwargs):       
        sig = 0.1
        
        y = jnp.moveaxis(inputs,1,-1)       # Because jax is channels last
        y = nn.Conv(features=self.chan1_n, kernel_size=(self.filt1_size,self.filt1_size),padding='SAME', kernel_init=glorot_uniform())(y)
        
        if training and rng is not None:
            noise = sig * jax.random.normal(rng, y.shape)  # Std-dev of 0.1; adjust as needed
            y = y + noise

        if self.MaxPool > 0:
            y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

        if self.BatchNorm == 1:
            y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
        y = TrainableAF()(y)

        
        # second layer
        if self.chan2_n>0:
            y = nn.Conv(features=self.chan2_n, kernel_size=(self.filt2_size,self.filt2_size),padding='SAME', kernel_init=glorot_uniform())(y)
            if training and rng is not None:
                noise = sig * jax.random.normal(rng, y.shape)  # Std-dev of 0.1; adjust as needed
                y = y + noise

            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

            if self.BatchNorm == 1:
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
            y = TrainableAF()(y)


        # Third layer
        if self.chan3_n>0:
            y = nn.Conv(features=self.chan3_n, kernel_size=(self.filt3_size,self.filt3_size),padding='SAME', kernel_init=glorot_uniform())(y)
            if training and rng is not None:
                noise = sig * jax.random.normal(rng, y.shape)  # Std-dev of 0.1; adjust as needed
                y = y + noise

            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

            if self.BatchNorm == 1:
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)

            y = TrainableAF()(y)

            
        if self.chan4_n>0:
            y = nn.Conv(features=self.chan4_n, kernel_size=(self.filt4_size,self.filt4_size),padding='SAME', kernel_init=glorot_uniform())(y)
            if training and rng is not None:
                noise = sig * jax.random.normal(rng, y.shape)  # Std-dev of 0.1; adjust as needed
                y = y + noise

            if self.BatchNorm == 1:
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
                
            y = TrainableAF()(y)
        
        y = nn.Conv(features=self.nout, kernel_size=(1,1),padding='SAME', kernel_init=he_normal(),name='output')(y)
        
        y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)

        outputs = TrainableAF()(y)

        self.sow('intermediates', 'dense_activations', outputs)

        return outputs    
class CNN2D_MAP3_FT(nn.Module):
    
    chan1_n : int
    filt1_size : int
    chan2_n : int
    filt2_size : int
    chan3_n : int
    filt3_size : int
    chan4_n : int
    filt4_size : int
    nout : int    
    filt_temporal_width : int    
    BatchNorm : bool
    MaxPool : int
    # dtype : type
    
    # def __init__(self, **kwargs):
    #     self.__dict__.update(kwargs)

    @nn.compact
    def __call__(self,inputs,training: bool,**kwargs):       
        y = jnp.moveaxis(inputs,1,-1)       # Because jax is channels last
        
        y = nn.LayerNorm(use_bias=True,use_scale=True,name='inputnorm',feature_axes=-1,reduction_axes=(1,2,3))(y)
        # y=ActivityScaler(name='inputnorm')(y)

        y = nn.Conv(features=self.chan1_n, kernel_size=(self.filt1_size,self.filt1_size),padding='SAME', kernel_init=glorot_uniform())(y)
        
        if self.MaxPool > 0:
            y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

        if self.BatchNorm == 1:
            y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
        y = TrainableAF()(y)

        
        # second layer
        if self.chan2_n>0:
            y = nn.Conv(features=self.chan2_n, kernel_size=(self.filt2_size,self.filt2_size),padding='SAME', kernel_init=glorot_uniform())(y)
            

            if self.BatchNorm == 1:
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
            y = TrainableAF()(y)


        # Third layer
        if self.chan3_n>0:
            y = nn.Conv(features=self.chan3_n, kernel_size=(self.filt3_size,self.filt3_size),padding='SAME', kernel_init=glorot_uniform())(y)
            

            if self.BatchNorm == 1:
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)

            y = TrainableAF()(y)

            
        if self.chan4_n>0:
            y = nn.Conv(features=self.chan4_n, kernel_size=(self.filt4_size,self.filt4_size),padding='SAME', kernel_init=glorot_uniform())(y)
           
            if self.BatchNorm == 1:
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
                
            y = TrainableAF()(y)
        
        y = nn.Conv(features=self.nout, kernel_size=(1,1),padding='SAME', kernel_init=he_normal(),name='output')(y)
        
        y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)

        outputs = TrainableAF()(y)

        # self.sow('intermediates', 'dense_activations', outputs)
        
        outputs=ActivityScaler(name='outputscale')(outputs)
        # outputs=ActivityScalerLog(name='outputscale')(outputs)

        # outputs = TrainableAF(name='outputaf')(outputs)
        self.sow('intermediates', 'dense_activations', outputs)

        return outputs  
    

class ActivityScaler(nn.Module):
    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', lambda rng, shape: jnp.ones(shape)*0.1, (1,))*10
        scale = jnp.clip(scale,min=0.2)
        bias = self.param('bias', lambda rng, shape: jnp.zeros(shape), (1,))
        return x * (scale*1) + bias
    
class ActivityScalerLog(nn.Module):
    @nn.compact
    def __call__(self, x):
        log_scale = self.param('log_scale', lambda rng, shape: jnp.zeros(shape), (1,))
        bias = self.param('bias', lambda rng, shape: jnp.zeros(shape), (1,))
        scale = jnp.exp(log_scale)
        return x * scale + bias


class TrainableAF(nn.Module):
    sat_init: float = 0.01
    gain_init: float = 0.95
    
    @nn.compact
    def __call__(self,x):
        sat = self.param('sat',lambda rng,shape: jnp.full(shape, self.sat_init), x.shape[-1:])
        gain = self.param('gain',lambda rng,shape: jnp.full(shape, self.gain_init), x.shape[-1:])
        
        a = ((1-sat+1e-6)*jnp.log(1+jnp.exp(gain*x)))/(gain+1e-6)
        b = (sat*(jnp.exp(gain*x)))/(1+jnp.exp(gain*x)+1e-6)
        
        outputs = a+b
        # outputs = jnp.clip(outputs,0)
        return outputs

class TimeModulator(nn.Module):
    # log_timescale = 0.0
    @nn.compact
    def __call__(self, x):  # x: [B, H, W, T]
        B, H, W, T = x.shape

        # Learnable log-timescale (starts at 0 → timescale=1)
        log_timescale = self.param('log_timescale', lambda rng, shape: jnp.zeros(shape), (1,))
        timescale = jnp.exp(log_timescale*100)  # shape: (1,)

        # Compute virtual time indices after rescaling
        t_idx = jnp.arange(T) / timescale  # shape: (T,)
        t_idx = jnp.clip(t_idx, 0, T - 1)

        # Linear interpolation indices
        lower = jnp.floor(t_idx).astype(jnp.int32)
        upper = jnp.clip(lower + 1, 0, T - 1)
        weight = (t_idx - lower)[None, None, None, :]  # broadcast to [1,1,1,T]

        # Gather frames
        x_lower = jnp.take(x, lower, axis=-1)  # shape: [B, H, W, T]
        x_upper = jnp.take(x, upper, axis=-1)  # shape: [B, H, W, T]

        # Interpolate
        x_mod = (1 - weight) * x_lower + weight * x_upper  # shape: [B, H, W, T]
        return x_mod

# class TrainableAFClipped(nn.Module):
#     sat_init: float = 0.01
#     gain_init: float = 0.95
    
#     @nn.compact
#     def __call__(self,x):
#         sat = self.param('sat',lambda rng,shape: jnp.full(shape, self.sat_init), x.shape[-1:])
#         gain = self.param('gain',lambda rng,shape: jnp.full(shape, self.gain_init), x.shape[-1:])
        
#         a = ((1-sat+1e-6)*jnp.log(1+jnp.exp(gain*x)))/(gain+1e-6)
#         b = (sat*(jnp.exp(gain*x)))/(1+jnp.exp(gain*x)+1e-6)
        
#         outputs = a+b
#         outputs = jnp.clip(outputs,0)
#         return outputs

    
    
class CNN2D_FT(nn.Module):
    
    chan1_n : int
    filt1_size : int
    chan2_n : int
    filt2_size : int
    chan3_n : int
    filt3_size : int
    chan4_n : int
    filt4_size : int
    nout : int    
    filt_temporal_width : int    
    BatchNorm : bool
    MaxPool : int
    # dtype : type
    

    @nn.compact
    def __call__(self,inputs,training: bool,**kwargs):       

        y = jnp.moveaxis(inputs,1,-1)       # Because jax is channels last
        
        rgb = y.shape[1:]
        y = y.reshape(y.shape[0],-1)
        y = nn.LayerNorm(use_bias=True,use_scale=True,name='LayerNorm_IN')(y)
        y = y.reshape(y.shape[0],*rgb)

        y = nn.Conv(features=self.chan1_n, kernel_size=(self.filt1_size,self.filt1_size),padding='VALID', kernel_init=glorot_uniform())(y)
        
        if self.MaxPool > 0:
            y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

        if self.BatchNorm == 1:
            rgb = y.shape[1:]
            y = y.reshape(y.shape[0],-1)
            y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
            y = y.reshape(y.shape[0],*rgb)

        y = nn.relu(y)
        
        # second layer
        if self.chan2_n>0:
            y = nn.Conv(features=self.chan2_n, kernel_size=(self.filt2_size,self.filt2_size),padding='VALID', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
                y = y.reshape(y.shape[0],*rgb)

            y = nn.relu(y)

        # Third layer
        if self.chan3_n>0:
            y = nn.Conv(features=self.chan3_n, kernel_size=(self.filt3_size,self.filt3_size),padding='VALID', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
                y = y.reshape(y.shape[0],*rgb)

            y = nn.relu(y)
            
        if self.chan4_n>0:
            y = nn.Conv(features=self.chan4_n, kernel_size=(self.filt4_size,self.filt4_size),padding='VALID', kernel_init=glorot_uniform())(y)
           
            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
                y = y.reshape(y.shape[0],*rgb)
                
            y = nn.relu(y)

        y = y.reshape(y.shape[0],-1)
        y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
        y = nn.Dense(features=self.nout,kernel_init=he_normal())(y)
        self.sow('intermediates', 'dense_activations', y)
        y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
        y = TrainableAF()(y)
        self.sow('intermediates', 'outputs_activations', y)
        # y = nn.softplus(y)
        outputs = y

        return outputs    


class CNN2D_FT2(nn.Module):
    
    chan1_n : int
    filt1_size : int
    chan2_n : int
    filt2_size : int
    chan3_n : int
    filt3_size : int
    chan4_n : int
    filt4_size : int
    nout : int    
    filt_temporal_width : int    
    BatchNorm : bool
    MaxPool : int
    # dtype : type
    

    @nn.compact
    def __call__(self,inputs,training: bool,**kwargs):       

        y = jnp.moveaxis(inputs,1,-1)       # Because jax is channels last
        
        rgb = y.shape[1:]
        y = y.reshape(y.shape[0],-1)
        
        
        # Encoder Conv
        y = nn.Conv(features=1, kernel_size=(1,1),padding='SAME',name='Conv_IN',kernel_init=glorot_uniform())(y)
        print(y.shape)
        rgb = y.shape[1:]
        y = y.reshape(y.shape[0],-1)
        y = nn.LayerNorm(use_bias=True,use_scale=True,name='LayerNorm_IN')(y)
        y = y.reshape(y.shape[0],*rgb)
        
        y = nn.relu(y)


        
        # CNNs start
        y = nn.Conv(features=self.chan1_n, kernel_size=(self.filt1_size,self.filt1_size),padding='VALID', kernel_init=glorot_uniform())(y)
        
        if self.MaxPool > 0:
            y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

        if self.BatchNorm == 1:
            rgb = y.shape[1:]
            y = y.reshape(y.shape[0],-1)
            y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
            y = y.reshape(y.shape[0],*rgb)

        y = nn.relu(y)
        
        # second layer
        if self.chan2_n>0:
            y = nn.Conv(features=self.chan2_n, kernel_size=(self.filt2_size,self.filt2_size),padding='VALID', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
                y = y.reshape(y.shape[0],*rgb)

            y = nn.relu(y)

        # Third layer
        if self.chan3_n>0:
            y = nn.Conv(features=self.chan3_n, kernel_size=(self.filt3_size,self.filt3_size),padding='VALID', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(self.MaxPool,self.MaxPool),padding='VALID')

            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
                y = y.reshape(y.shape[0],*rgb)

            y = nn.relu(y)
            
        if self.chan4_n>0:
            y = nn.Conv(features=self.chan4_n, kernel_size=(self.filt4_size,self.filt4_size),padding='VALID', kernel_init=glorot_uniform())(y)
           
            if self.BatchNorm == 1:
                rgb = y.shape[1:]
                y = y.reshape(y.shape[0],-1)
                y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
                y = y.reshape(y.shape[0],*rgb)
                
            y = nn.relu(y)

        y = y.reshape(y.shape[0],-1)
        y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
        y = nn.Dense(features=self.nout,kernel_init=he_normal())(y)
        y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
        self.sow('intermediates', 'dense_activations', y)
        outputs = TrainableAF()(y)

        
        return outputs    
    
    
# %% PR-CNN

@jax.jit
def runRiekeModel(X_fun, TimeStep, sigma, phi, eta, cgmp2cur, cgmphill, cdark, beta, betaSlow, hillcoef, hillaffinity, gamma, gdark):
    darkCurrent = gdark**cgmphill * cgmp2cur/2
    gdark = (2 * darkCurrent / cgmp2cur) **(1/cgmphill)
    
    cur2ca = beta * cdark / darkCurrent                # get q using steady state
    smax = eta/phi * gdark * (1 + (cdark / hillaffinity) **hillcoef)		# get smax using steady state
    
    NumPts = X_fun.shape[1]
    
    # initial conditions   
    g_prev = gdark + (X_fun[:,0,:] * 0)
    s_prev = (gdark * eta/phi) + (X_fun[:,0,:] * 0)
    c_prev = cdark + (X_fun[:,0,:] * 0)
    cslow_prev = cdark + (X_fun[:,0,:] * 0)
    r_prev = X_fun[:,0,:] * gamma / sigma
    p_prev = (eta + r_prev) / phi

    g = jnp.zeros_like(X_fun)
    
    # solve difference equations
    for pnt in range(1, NumPts):
        r_curr = r_prev + TimeStep * (-1 * sigma * r_prev)
        r_curr = r_curr + gamma * X_fun[:, pnt-1, :]
        p_curr = p_prev + TimeStep * (r_prev + eta - phi * p_prev)
        c_curr = c_prev + TimeStep * (cur2ca * cgmp2cur * g_prev**cgmphill / (1 + (cslow_prev/cdark)) - beta * c_prev)
        cslow_curr = cslow_prev - TimeStep * (betaSlow * (cslow_prev - c_prev))
        s_curr = smax / (1 + (c_curr / hillaffinity) ** hillcoef)
        g_curr = g_prev + TimeStep * (s_prev - p_prev * g_prev)

        g = g.at[:, pnt, :].set(g_curr)
        
        # update prev values to current
        g_prev = g_curr
        s_prev = s_curr
        c_prev = c_curr
        p_prev = p_curr
        r_prev = r_curr
        cslow_prev = cslow_curr
    
    outputs = -(cgmp2cur * g ** cgmphill) / 2
    return outputs


# JAX Layer Class
class PRFR(nn.Module):
    pr_params: dict
    units: int = 1

    def setup(self):
        self.dtype = jnp.float32
        shape = (1, self.units)

        
        param_names = [
            'sigma','phi','eta','beta','cgmp2cur','cgmphill','cdark',
            'betaSlow','hillcoef','hillaffinity','gamma','gdark',
        ]
        
        param_names_trainable = []
        for name in param_names:
            trainable = self.pr_params[name+'_trainable']
            if trainable==True:
                param_names_trainable.append(name)
        
        
        for name in param_names_trainable:
            init_value = self.pr_params[name]
            param = self.param(name, nn.initializers.constant(init_value),shape=shape,dtype=self.dtype)
            setattr(self, name, param)

        if len(param_names_trainable)==0:
            param = self.param('dummy', nn.initializers.constant(1.),shape=shape,dtype=self.dtype)
            setattr(self, 'dummy', param)

        param_names_nontrainable = list(set(param_names)-set(param_names_trainable))
        for name in param_names_nontrainable:
            init_value = self.pr_params[name]
            setattr(self, name, init_value)


    def __call__(self, X_fun):
        # X_fun is the input tensor of shape (batch, time_steps, units)
        timeBin = float(self.pr_params['timeBin'])  # ms
        frameTime = 8  # ms
        upSamp_fac = int(frameTime/timeBin)
        TimeStep = 1e-3 * timeBin

        # Upsample if needed
        if upSamp_fac > 1:
            X_fun = jnp.repeat(X_fun, upSamp_fac, axis=1)
            X_fun = X_fun / upSamp_fac  # scale photons/ms

        # Use parameters and scale factors
        sigma = self.sigma * self.pr_params['sigma_scaleFac']
        phi = self.phi * self.pr_params['phi_scaleFac']
        eta = self.eta * self.pr_params['eta_scaleFac']
        cgmp2cur = self.cgmp2cur
        cgmphill = self.cgmphill * self.pr_params['cgmphill_scaleFac']
        cdark = self.cdark
        beta = self.beta * self.pr_params['beta_scaleFac']
        betaSlow = self.betaSlow * self.pr_params['betaSlow_scaleFac']
        hillcoef = self.hillcoef * self.pr_params['hillcoef_scaleFac']
        hillaffinity = self.hillaffinity * self.pr_params['hillaffinity_scaleFac']
        gamma = (self.gamma * self.pr_params['gamma_scaleFac'])
        gdark = self.gdark * self.pr_params['gdark_scaleFac']

        # Call the custom Rieke model function
        outputs = runRiekeModel(X_fun, TimeStep, sigma, phi, eta, cgmp2cur, cgmphill, cdark, beta, betaSlow, hillcoef, hillaffinity, gamma, gdark)

        # Downsample if needed
        if upSamp_fac > 1:
            outputs = outputs[:, upSamp_fac-1::upSamp_fac]
        return outputs

    
class PRFR_CNN2D_MAP(nn.Module):
    chan1_n : int
    filt1_size : int
    chan2_n : int
    filt2_size : int
    chan3_n : int
    filt3_size : int
    chan4_n : int
    filt4_size : int
    nout : int    
    filt_temporal_width : int    
    BatchNorm : bool
    MaxPool : int
    pr_params: dict

    @nn.compact
    def __call__(self,inputs,training: bool,rng=None,**kwargs):       
        # pr_params = fr_rods_trainable()
        
        N_trunc = inputs.shape[1]-self.filt_temporal_width

        photoreceptor_layer = PRFR(pr_params=self.pr_params)  # Instantiate the layer with pr_params
        y = inputs
        y = jnp.reshape(y,(y.shape[0],y.shape[1],y.shape[-2]*y.shape[-1]))
        y = photoreceptor_layer(y)  # Apply the layer to inputs
        y = jnp.reshape(y,inputs.shape)
        y = y[:,N_trunc:]    # truncate first 20 points
        y = nn.LayerNorm(feature_axes=1,reduction_axes=(1,2,3),use_bias=True,use_scale=True)(y)      # Along the temporal axis

        y = jnp.moveaxis(y,1,-1)       # Because jax is channels last
        y = nn.Conv(features=self.chan1_n, kernel_size=(self.filt1_size,self.filt1_size),padding='SAME', kernel_init=glorot_uniform())(y)
        
        if self.MaxPool > 0:
            y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

        if self.BatchNorm == 1:
            y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
        y = TrainableAF()(y)

        
        # second layer
        if self.chan2_n>0:
            y = nn.Conv(features=self.chan2_n, kernel_size=(self.filt2_size,self.filt2_size),padding='SAME', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

            if self.BatchNorm == 1:
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
            y = TrainableAF()(y)


        # Third layer
        if self.chan3_n>0:
            y = nn.Conv(features=self.chan3_n, kernel_size=(self.filt3_size,self.filt3_size),padding='SAME', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

            if self.BatchNorm == 1:
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)

            y = TrainableAF()(y)

            
        if self.chan4_n>0:
            y = nn.Conv(features=self.chan4_n, kernel_size=(self.filt4_size,self.filt4_size),padding='SAME', kernel_init=glorot_uniform())(y)
           
            if self.BatchNorm == 1:
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
                
            y = TrainableAF()(y)
        
        y = nn.Conv(features=self.nout, kernel_size=(1,1),padding='SAME', kernel_init=he_normal(),name='output')(y)
        
        y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)

        outputs = TrainableAF()(y)

        self.sow('intermediates', 'dense_activations', outputs)

        return outputs    
    
class PRFR_CNN2D_MAP_FT(nn.Module):
    chan1_n : int
    filt1_size : int
    chan2_n : int
    filt2_size : int
    chan3_n : int
    filt3_size : int
    chan4_n : int
    filt4_size : int
    nout : int    
    filt_temporal_width : int    
    BatchNorm : bool
    MaxPool : int
    pr_params: dict

    @nn.compact
    def __call__(self,inputs,training: bool,**kwargs):       
        # pr_params = fr_rods_trainable()
        
        N_trunc = inputs.shape[1]-self.filt_temporal_width

        photoreceptor_layer = PRFR(pr_params=self.pr_params)  # Instantiate the layer with pr_params
        y = inputs
        y = jnp.reshape(y,(y.shape[0],y.shape[1],y.shape[-2]*y.shape[-1]))
        y = photoreceptor_layer(y)  # Apply the layer to inputs
        y = jnp.reshape(y,inputs.shape)
        y = y[:,N_trunc:]    # truncate first 20 points
        y = nn.LayerNorm(feature_axes=1,reduction_axes=(1,2,3),use_bias=True,use_scale=True)(y)      # Along the temporal axis

        y = jnp.moveaxis(y,1,-1)       # Because jax is channels last
        y = nn.Conv(features=self.chan1_n, kernel_size=(self.filt1_size,self.filt1_size),padding='SAME', kernel_init=glorot_uniform())(y)
        
        if self.MaxPool > 0:
            y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

        if self.BatchNorm == 1:
            y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
        y = TrainableAF()(y)

        
        # second layer
        if self.chan2_n>0:
            y = nn.Conv(features=self.chan2_n, kernel_size=(self.filt2_size,self.filt2_size),padding='SAME', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

            if self.BatchNorm == 1:
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
            y = TrainableAF()(y)


        # Third layer
        if self.chan3_n>0:
            y = nn.Conv(features=self.chan3_n, kernel_size=(self.filt3_size,self.filt3_size),padding='SAME', kernel_init=glorot_uniform())(y)
            
            if self.MaxPool > 0:
                y = nn.max_pool(y,window_shape=(self.MaxPool,self.MaxPool),strides=(1,1),padding='SAME')

            if self.BatchNorm == 1:
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)

            y = TrainableAF()(y)

            
        if self.chan4_n>0:
            y = nn.Conv(features=self.chan4_n, kernel_size=(self.filt4_size,self.filt4_size),padding='SAME', kernel_init=glorot_uniform())(y)
           
            if self.BatchNorm == 1:
                y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)
                
            y = TrainableAF()(y)
        
        y = nn.Conv(features=self.nout, kernel_size=(1,1),padding='SAME', kernel_init=he_normal(),name='output')(y)
        
        y = nn.LayerNorm(use_bias=True,use_scale=True,feature_axes=-1,reduction_axes=(1,2,3))(y)

        outputs = TrainableAF()(y)
        
        outputs=ActivityScaler(name='outputscale')(outputs)
        self.sow('intermediates', 'dense_activations', outputs)
        return outputs    
    
