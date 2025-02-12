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



def model_definitions():
    """
        How to arrange the datasets depends on which model is being used
    """
    
    models_2D = ('CNN2D','CNN2D_MAXPOOL','CNN2D_FT','CNN2D_FT2','CNN2D_LNORM','CNN2D_MAP')
    
    models_3D = ('CNN_3D','PR_CNN3D')
    
    return (models_2D,models_3D)


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
        
        rgb = y.shape[1:]
        y = y.reshape(y.shape[0],-1)
        y = nn.LayerNorm(use_bias=True,use_scale=True)(y)
        y = y.reshape(y.shape[0],*rgb)

        # outputs = nn.softplus(y)
        outputs = TrainableAF()(y)
        outputs = nn.relu(outputs)

        self.sow('intermediates', 'dense_activations', outputs)

        return outputs    
    
    
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