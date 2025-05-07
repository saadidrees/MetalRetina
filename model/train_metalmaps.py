
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:26:35 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""
import gc
import h5py
import shutil
import os
import numpy as np
import jax
import jax.lax as lax
from jax import numpy as jnp
import optax
from tqdm.auto import tqdm
import flax
from flax.training import train_state
from flax.training.train_state import TrainState
import time
from model import models_jax
import matplotlib.pyplot as plt
import cloudpickle


from model.performance import model_evaluate_new


from flax.training import orbax_utils
import orbax.checkpoint

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from functools import partial
from jax.tree_util import Partial

from model.dataloaders import RetinaDataset,jnp_collate
from torch.utils.data import DataLoader
import re

MAX_RGCS = 500

LOSS_FUN = 'mad' #'mad'        # poisson poissonreg madpoissonreg mad  madreg  msereg


def to_cpu(grads):
    return jax.tree_map(lambda g: np.asarray(g),grads)

def weighted_mae(y_true, y_pred, weight_factor=10.0, threshold=0.05):
    weights = jnp.where(y_true > threshold, weight_factor, 1.0)
    loss = jnp.mean(weights * jnp.abs(y_true - y_pred))
    return loss

def append_dicts(dict1, dict2):
    result = {}
    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            result[key] = append_dicts(dict1[key], dict2[key])
        elif isinstance(dict1[key], jnp.ndarray) and isinstance(dict2[key], jnp.ndarray):
            result[key] = jnp.append(dict1[key], dict2[key][None,:], axis=0)
        else:
            raise ValueError("Mismatched structure or non-numpy array value at the lowest level")
    return result

def expand_dicts(dict1):
    dict2 = dict1
    result = {}
    for key in dict1:
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            result[key] = expand_dicts(dict1[key])
        elif isinstance(dict1[key], jnp.ndarray) and isinstance(dict2[key], jnp.ndarray):
            result[key] = dict1[key][None,:]
        else:
            raise ValueError("Mismatched structure or non-numpy array value at the lowest level")
    return result

def mask_params(dict_params,layers_to_mask,mask_value=0):
    masked_params = dict_params
    for key in dict_params.keys():
        if key in layers_to_mask:
            for subkey in dict_params[key].keys():
                masked_params[key][subkey] = mask_value*jnp.ones_like(masked_params[key][subkey])
                
    return masked_params
            
def dict_subset(old_dict,exclude_list):
    new_dict1 = {}
    new_dict2 = {}
    keys_all = list(old_dict.keys())
    for item in keys_all:
        for key_exclude in exclude_list:
            rgb = re.search(key_exclude,item,re.IGNORECASE)
            if rgb==None:
                break;
                new_dict1[item] = old_dict[item]
            else:
                new_dict2[item] = old_dict[item]
    return new_dict1,new_dict2

def split_dict(old_dict,exclude_list):
    def should_exclude(key, patterns):
        return any(re.match(pattern, key) for pattern in patterns)

    new_dict1 = {k: v for k, v in old_dict.items() if not should_exclude(k, exclude_list)}
    new_dict2 = {k: v for k, v in old_dict.items() if should_exclude(k, exclude_list)}
    
    return new_dict1,new_dict2


def compute_means_across_nested_dicts(data_list):
    # Initialize dictionaries to hold stacked arrays for each variable
    nested_keys = list(data_list[0].keys())
    variable_keys = data_list[0][nested_keys[0]].keys()
    
    stacked_data = {nested_key: {var_key: [] for var_key in variable_keys} for nested_key in nested_keys}
    
    # Stack values for each variable
    for data in data_list:
        for nested_key, nested_dict in data.items():
            for var_key, value in nested_dict.items():
                stacked_data[nested_key][var_key].append(value)
    
    # Convert lists to arrays and compute the mean for each variable
    means = {nested_key: {var_key: jnp.mean(jnp.stack(values), axis=0) 
                          for var_key, values in nested_vars.items()} 
             for nested_key, nested_vars in stacked_data.items()}
    
    return means

def clip_grads(grads, clip_value=1.0):
    clipped_grads = jax.tree_util.tree_map(
        lambda grad: jnp.clip(grad, -clip_value, clip_value),
        grads
    )
    return clipped_grads

@jax.jit
def pred_psfavg(y_pred,coords,segment_size,MAX_RGCS=MAX_RGCS):   
    """
    test_sumseg = np.zeros((83,16))
    u_id_unique = np.unique(u_id)
    u=0
    for u in range(len(u_id_unique)):
        idx = u_id==u_id_unique[u]
        test_sumseg[u,:] = np.sum(y_pred_allpixs[idx,:],axis=0)
    """
    # MAX_RGCS = 600#jnp.array(MAX_RGCS,int)
    
    spatial_y = coords[:,3]  # Shape: (M,)
    spatial_x = coords[:,2]  # Shape: (M,)
    type_idx = coords[:,1] - 1  # Shape: (M,)
    u_id = coords[:,0]

    y_pred_allpixs = y_pred[:,spatial_y,spatial_x,type_idx].T
    # y_pred_allpixs = y_pred_allpixs*mask_tr[:,None]       # There's no need for this as segment_sum handles it because we have -1 for coords not to be used
    y_pred_sumsegs = jax.ops.segment_sum(y_pred_allpixs,u_id,MAX_RGCS)
    y_pred_units = y_pred_sumsegs / segment_size  # Start from 1st index because index 0 is the ones we dont want to compare as thats padded
        
    return y_pred_units.T

@jax.jit
def calc_loss(y_pred,y,coords,segment_size,N_tr,mask_tr):
    y_pred_units = pred_psfavg(y_pred,coords,segment_size)
    if jnp.ndim(y) == jnp.ndim(y_pred): # That is both are in terms of MAPS
        y_units = pred_psfavg(y,coords,segment_size)      # This is just going to be the actual value at a single pixel
    else:
        y_units=y
    
    y_pred_units = jnp.where(y_pred_units == 0, 1e-6, y_pred_units)

    # y_pred_units = jnp.where(y_pred_units == 0, 1e-6, y_pred_units)
    if LOSS_FUN=='poisson':
        loss = y_pred_units-y_units*jax.lax.log(y_pred_units)
    elif LOSS_FUN=='poissonreg':
        poisson_loss = y_pred_units-y_units*jax.lax.log(y_pred_units)
        reg_loss = 1e-1*y_pred_units
        loss = poisson_loss+reg_loss
    elif LOSS_FUN=='madpoissonreg':
        poisson_loss = y_pred_units-y_units*jax.lax.log(y_pred_units)
        mad_loss = jnp.abs(y_units-y_pred_units)
        reg_loss = 1e-1*y_pred_units
        loss = poisson_loss+mad_loss+reg_loss
    elif LOSS_FUN=='mad':
        loss = jnp.abs(y_units-y_pred_units)
    elif LOSS_FUN=='madreg':
        mad_loss = jnp.abs(y_units-y_pred_units)
        reg_loss = 1e-2*y_pred_units
        loss = mad_loss+reg_loss
    elif LOSS_FUN=='mse':
        loss = (y_units-y_pred_units)**2
    elif LOSS_FUN=='wmad':
        weight_factor=10
        threshold=0.1
        weights = jnp.where(y_units > threshold, weight_factor, 1.0)
        loss = weights * jnp.abs(y_units - y_pred_units)
    elif LOSS_FUN=='rmad':
        eps=1e-3
        loss = jnp.abs(y_units - y_pred_units)/(jnp.abs(y_units)+eps)
    else:
        raise Exception('Loss Function Not Found')



    loss = (loss*mask_tr[None,:])
    loss=jnp.nansum(loss)/(N_tr*loss.shape[0])
    
    return loss,y_units,y_pred_units

@jax.jit
def task_loss(state,params,batch,coords,N_tr,segment_size,mask_tr):
    """
    batch = batch_train
    coords = coords_tr#[coords_tr[:,1]>0,:]
    state = mdl_state
    params = mdl_state.params
    N_tr = N_tr
    N_val = N_val
    # N_points=N_points_tr
    """
    X,y = batch
    rng = jax.random.PRNGKey(0)
    y_pred,state = state.apply_fn({'params': params},X,training=True,mutable=['intermediates'],rng=rng)
    
    # intermediates = state['intermediates']
    # dense_activations = intermediates['dense_activations'][0]
    loss,y_units,y_pred_units = calc_loss(y_pred,y,coords,segment_size,N_tr,mask_tr)
    loss = loss + weight_regularizer(params,alpha=1e-3)
    # loss = loss + activity_regularizer(dense_activations,alpha=1e-4)

    return loss,y_pred

@jax.jit
def train_step_metalzero(mdl_state,batch,weights_output,lr,dinf_tr):        # Make unit vectors then scale by num of RGCs
    """
    State is the grand model state that actually gets updated
    state_task is the "state" after gradients are applied for a specific task
        task_idx = 11
        conv_kern = conv_kern_all[task_idx]
        conv_bias = conv_bias_all[task_idx]
        train_x_tr = train_x_tr[task_idx]
        train_y_tr = train_y_tr[task_idx]
        train_x_val = train_x_val[task_idx]
        train_y_val = train_y_val[task_idx]
        coords_tr = umaskcoords_trtr[task_idx]
        coords_val = umaskcoords_trval[task_idx]
        N_tr = N_trtr[task_idx]
        N_val = N_trval[task_idx]
        mask_tr = mask_trtr[task_idx]
        mask_val = mask_trval[task_idx]
        
        loss,mdl_state,weights_output,grads = train_step_metal(mdl_state,batch_train,weights_output,current_lr,dinf_tr)
    """
    @jax.jit
    def metalzero_grads(mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size,lr_inner,train_x_tr,train_y_tr,train_x_val,train_y_val,coords_tr,coords_val,N_tr,N_val,mask_tr,mask_val,conv_kern,conv_bias):

       
        batch_train = (train_x_tr,train_y_tr)
        batch_val = (train_x_val,train_y_val)
        local_mdl_state = mdl_state #.replace(params=global_params)

        # Calculate gradients of the local model wrt to local params    
        grad_fn = jax.value_and_grad(task_loss,argnums=1,has_aux=True)
        (local_loss_train,y_pred_train),local_grads = grad_fn(local_mdl_state,global_params,batch_train,coords_tr,N_tr,segment_size,mask_tr)
        
        # scale the local gradients according to ADAM's first step. Helps to stabilize
        # And update the parameters
        local_params = jax.tree_map(lambda p, g: p - lr_inner*(g/(jnp.abs(g)+1e-8)), global_params, local_grads)

        # Calculate gradients of the loss of the resulting local model but using the validation set
        # local_mdl_state = mdl_state.replace(params=local_params)
        (local_loss_val,y_pred_val),local_grads_val = grad_fn(local_mdl_state,local_params,batch_val,coords_val,N_val,segment_size,mask_val)
        
        # Update only the Dense layer weights since we retain it
        local_params_val = jax.tree_map(lambda p, g: p - lr_inner*(g/(jnp.abs(g)+1e-8)), local_params, local_grads_val)
        
        # Get the direction of generalization
        local_grads_total = jax.tree_map(lambda g_1, g_2: g_1+g_2, local_grads,local_grads_val)
        
        # Normalize the grads to unit vector
        local_grads_total = jax.tree_map(lambda g: g/jnp.linalg.norm(g), local_grads_total)
        
        # Scale vectors by num of RGCs
        scaleFac = (N_tr+N_val)/MAX_RGCS
        local_grads_total = jax.tree_map(lambda g: g*scaleFac, local_grads_total)


        # Record dense layer weights
        conv_kern = local_params_val['output']['kernel']
        conv_bias = local_params_val['output']['bias']
        
        return local_loss_val,y_pred_val,local_mdl_state,local_grads_total,conv_kern,conv_bias
    
    """
    batch_train = next(iter(dataloader_train)); batch=batch_train; 
    """
    NUM_SPLITS=0
    global_params = mdl_state.params
    
    train_x_tr,train_y_tr,train_x_val,train_y_val = batch
    conv_kern_all,conv_bias_all = weights_output
    umaskcoords_trtr = dinf_tr['umaskcoords_trtr']
    umaskcoords_trval = dinf_tr['umaskcoords_trval']
    N_trtr = dinf_tr['N_trtr']
    N_trval = dinf_tr['N_trval']
    mask_trtr = dinf_tr['maskunits_trtr']
    mask_trval = dinf_tr['maskunits_trval']

    MAX_RGCS = dinf_tr['MAX_RGCS']
    cell_types_unique = dinf_tr['cell_types_unique']
    segment_size = dinf_tr['segment_size']
    
    lr_inner = lr*10


    if NUM_SPLITS==0:
        local_losses,local_y_preds,local_mdl_states,local_grads_all,local_kerns,local_biases = jax.vmap(Partial(metalzero_grads,\
                                                                                                                  mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size,lr_inner))\
                                                                                                                  (train_x_tr,train_y_tr,train_x_val,train_y_val,
                                                                                                                   umaskcoords_trtr,umaskcoords_trval,N_trtr,N_trval,mask_trtr,mask_trval,
                                                                                                                   conv_kern_all,conv_bias_all)
                                                                                                              
    else:       # Otherwise split the vmap. This avoids running out of GPU memory when we have many retinas and large batch size
        (local_losses, local_y_preds, local_kerns, local_biases),local_grads_all = batched_metal_grads(metalzero_grads,
        mdl_state, global_params,MAX_RGCS,cell_types_unique,segment_size,lr_inner,train_x_tr, train_y_tr, train_x_val, train_y_val,
        umaskcoords_trtr, umaskcoords_trval, N_trtr, N_trval, mask_trtr, mask_trval, conv_kern_all, conv_bias_all, 
        NUM_SPLITS=NUM_SPLITS)
                                                                                                          
                                                                                                              
                  
    local_losses_summed = jnp.sum(local_losses)
    local_grads_summed = jax.tree_map(lambda g: jnp.sum(g,axis=0), local_grads_all)
    local_grads_summed = clip_grads(local_grads_summed)

    
    weights_output = (local_kerns,local_biases)
    
    mdl_state = mdl_state.apply_gradients(grads=local_grads_summed)
    
           
    # print(local_losses_summed)   
        
    
    """
    for key in local_grads_summed.keys():
        try:
            print('%s kernel: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['kernel']))))
        except:
            print('%s bias: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['bias']))))
    
    """

    return local_losses_summed,mdl_state,weights_output,local_grads_summed

@jax.jit
def train_step_metalzeroperturb(mdl_state,batch,weights_output,lr,dinf_tr):        # Make unit vectors then scale by num of RGCs
    """
    State is the grand model state that actually gets updated
    state_task is the "state" after gradients are applied for a specific task
        task_idx = 11
        conv_kern = conv_kern_all[task_idx]
        conv_bias = conv_bias_all[task_idx]
        train_x_tr = train_x_tr[task_idx]
        train_y_tr = train_y_tr[task_idx]
        train_x_val = train_x_val[task_idx]
        train_y_val = train_y_val[task_idx]
        coords_tr = umaskcoords_trtr[task_idx]
        coords_val = umaskcoords_trval[task_idx]
        N_tr = N_trtr[task_idx]
        N_val = N_trval[task_idx]
        mask_tr = mask_trtr[task_idx]
        mask_val = mask_trval[task_idx]
        
        loss,mdl_state,weights_output,grads = train_step_metal(mdl_state,batch_train,weights_output,current_lr,dinf_tr)
    """
    @jax.jit
    def metalzero_grads(mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size,lr_inner,train_x_tr,train_y_tr,train_x_val,train_y_val,coords_tr,coords_val,N_tr,N_val,mask_tr,mask_val,conv_kern,conv_bias):

       
        batch_train = (train_x_tr,train_y_tr)
        batch_val = (train_x_val,train_y_val)
        local_mdl_state = mdl_state #.replace(params=global_params)

        # Calculate gradients of the local model wrt to local params    
        grad_fn = jax.value_and_grad(task_loss,argnums=1,has_aux=True)
        (local_loss_train,y_pred_train),local_grads = grad_fn(local_mdl_state,global_params,batch_train,coords_tr,N_tr,segment_size,mask_tr)
        
        # scale the local gradients according to ADAM's first step. Helps to stabilize
        # And update the parameters
        local_params = jax.tree_map(lambda p, g: p - lr_inner*(g/(jnp.abs(g)+1e-8)), global_params, local_grads)

        # Calculate gradients of the loss of the resulting local model but using the validation set
        # local_mdl_state = mdl_state.replace(params=local_params)
        (local_loss_val,y_pred_val),local_grads_val = grad_fn(local_mdl_state,local_params,batch_val,coords_val,N_val,segment_size,mask_val)
        
        # Update only the Dense layer weights since we retain it
        local_params_val = jax.tree_map(lambda p, g: p - lr_inner*(g/(jnp.abs(g)+1e-8)), local_params, local_grads_val)
        
        # Get the direction of generalization
        local_grads_total = jax.tree_map(lambda g_1, g_2: g_1+g_2, local_grads,local_grads_val)
        
        # Normalize the grads to unit vector
        local_grads_total = jax.tree_map(lambda g: g/jnp.linalg.norm(g), local_grads_total)
        
        # Scale vectors by num of RGCs
        scaleFac = (N_tr+N_val)/MAX_RGCS
        local_grads_total = jax.tree_map(lambda g: g*scaleFac, local_grads_total)


        # Record dense layer weights
        conv_kern = local_params_val['output']['kernel']
        conv_bias = local_params_val['output']['bias']
        
        
        rng = jax.random.PRNGKey(0)
        sig = 0.01
        perturbed_grads = jax.tree_map(lambda g: g+(jax.random.normal(rng, g.shape)*sig), local_grads_total) 
        
        relax_steps=3
        relaxed_grad = perturbed_grads
        params = jax.tree_util.tree_map(lambda p, g: p - lr_inner*(g/(jnp.abs(g)+1e-8)), global_params, relaxed_grad)
        relax_alpha = 0.8


        for _ in range(relax_steps):
            # Recompute current gradient at new params
            # current_grad = grad(loss_fn)(params, data)
            (_,_),current_grads = grad_fn(local_mdl_state,params,batch_train,coords_tr,N_tr,segment_size,mask_tr)
            # Exponential moving average (gradient smoothing)
            relaxed_grad = jax.tree_util.tree_map(lambda r, c: relax_alpha * r + (1 - relax_alpha) * c,relaxed_grad,current_grads)
            # Optionally: simulate moving params slightly in relaxed direction
            params = jax.tree_util.tree_map(lambda p, g: p - lr_inner*(g/(jnp.abs(g)+1e-8)), params, relaxed_grad)

        local_grads_total = relaxed_grad
        
        return local_loss_val,y_pred_val,local_mdl_state,local_grads_total,conv_kern,conv_bias
    
    """
    batch_train = next(iter(dataloader_train)); batch=batch_train; 
    """
    NUM_SPLITS=0
    global_params = mdl_state.params
    
    train_x_tr,train_y_tr,train_x_val,train_y_val = batch
    conv_kern_all,conv_bias_all = weights_output
    umaskcoords_trtr = dinf_tr['umaskcoords_trtr']
    umaskcoords_trval = dinf_tr['umaskcoords_trval']
    N_trtr = dinf_tr['N_trtr']
    N_trval = dinf_tr['N_trval']
    mask_trtr = dinf_tr['maskunits_trtr']
    mask_trval = dinf_tr['maskunits_trval']

    MAX_RGCS = dinf_tr['MAX_RGCS']
    cell_types_unique = dinf_tr['cell_types_unique']
    segment_size = dinf_tr['segment_size']
    
    lr_inner = lr*10


    if NUM_SPLITS==0:
        local_losses,local_y_preds,local_mdl_states,local_grads_all,local_kerns,local_biases = jax.vmap(Partial(metalzero_grads,\
                                                                                                                  mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size,lr_inner))\
                                                                                                                  (train_x_tr,train_y_tr,train_x_val,train_y_val,
                                                                                                                   umaskcoords_trtr,umaskcoords_trval,N_trtr,N_trval,mask_trtr,mask_trval,
                                                                                                                   conv_kern_all,conv_bias_all)
                                                                                                              
    else:       # Otherwise split the vmap. This avoids running out of GPU memory when we have many retinas and large batch size
        (local_losses, local_y_preds, local_kerns, local_biases),local_grads_all = batched_metal_grads(metalzero_grads,
        mdl_state, global_params,MAX_RGCS,cell_types_unique,segment_size,lr_inner,train_x_tr, train_y_tr, train_x_val, train_y_val,
        umaskcoords_trtr, umaskcoords_trval, N_trtr, N_trval, mask_trtr, mask_trval, conv_kern_all, conv_bias_all, 
        NUM_SPLITS=NUM_SPLITS)
                                                                                                          
                                                                                                              
                  
    local_losses_summed = jnp.sum(local_losses)
    local_grads_summed = jax.tree_map(lambda g: jnp.sum(g,axis=0), local_grads_all)
    local_grads_summed = clip_grads(local_grads_summed)

    
    weights_output = (local_kerns,local_biases)
    
    mdl_state = mdl_state.apply_gradients(grads=local_grads_summed)
    
           
    # print(local_losses_summed)   
        
    
    """
    for key in local_grads_summed.keys():
        try:
            print('%s kernel: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['kernel']))))
        except:
            print('%s bias: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['bias']))))
    
    """

    return local_losses_summed,mdl_state,weights_output,local_grads_summed


@jax.jit
def train_step_metalzero1step(mdl_state,batch,weights_output,lr,dinf_tr):        # Make unit vectors then scale by num of RGCs
    """
    State is the grand model state that actually gets updated
    state_task is the "state" after gradients are applied for a specific task
        task_idx = 1
        conv_kern = conv_kern_all[task_idx]
        conv_bias = conv_bias_all[task_idx]
        train_x_tr = train_x_tr[task_idx]
        train_y_tr = train_y_tr[task_idx]
        train_x_val = train_x_val[task_idx]
        train_y_val = train_y_val[task_idx]
        coords_tr = umaskcoords_trtr[task_idx]
        coords_val = umaskcoords_trval[task_idx]
        N_tr = N_trtr[task_idx]
        N_val = N_trval[task_idx]
        mask_tr = mask_trtr[task_idx]
        mask_val = mask_trval[task_idx]
        
        loss,mdl_state,weights_output,grads = train_step_metal(mdl_state,batch_train,weights_output,current_lr,dinf_tr)
    """
    @jax.jit
    def metalzero_grads1step(mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size,lr_inner,train_x_tr,train_y_tr,train_x_val,train_y_val,coords_tr,coords_val,N_tr,N_val,mask_tr,mask_val,conv_kern,conv_bias):
       
        train_x = jnp.concatenate((train_x_tr,train_x_val),axis=0)
        train_y = jnp.concatenate((train_y_tr,train_y_val),axis=0)
        batch_train = (train_x,train_y)
        # batch_train = (train_x_tr,train_y_tr)
        # batch_val = (train_x_val,train_y_val)

        # Make local model by using global params but local dense layer weights
        # local_params = global_params
        local_mdl_state = mdl_state#.replace(params=global_params)

        # Calculate gradients of the local model wrt to local params    
        grad_fn = jax.value_and_grad(task_loss,argnums=1,has_aux=True)
        (local_loss_train,y_pred_train),local_grads = grad_fn(local_mdl_state,global_params,batch_train,coords_tr,N_tr,segment_size,mask_tr)
        
        # scale the local gradients according to ADAM's first step. Helps to stabilize
        # And update the parameters
        local_params = jax.tree_map(lambda p, g: p-lr_inner*(g/(jnp.abs(g)+1e-8)),global_params,local_grads)
      
        # Normalize the grads to unit vector
        local_grads = jax.tree_map(lambda g: g/jnp.linalg.norm(g), local_grads)
        
        # Scale vectors by num of RGCs
        scaleFac = (N_tr+N_val)/MAX_RGCS
        local_grads = jax.tree_map(lambda g: g*scaleFac, local_grads)

        # Record dense layer weights
        conv_kern = local_params['output']['kernel']
        conv_bias = local_params['output']['bias']
        
        return local_loss_train,y_pred_train,local_mdl_state,local_grads,conv_kern,conv_bias
    
    """
    batch_train = next(iter(dataloader_train)); batch=batch_train; 
    """
    NUM_SPLITS=0
    global_params = mdl_state.params
    
    train_x_tr,train_y_tr,train_x_val,train_y_val = batch
    conv_kern_all,conv_bias_all = weights_output
    umaskcoords_trtr = dinf_tr['umaskcoords_trtr']
    umaskcoords_trval = dinf_tr['umaskcoords_trval']
    N_trtr = dinf_tr['N_trtr']
    N_trval = dinf_tr['N_trval']
    mask_trtr = dinf_tr['maskunits_trtr']
    mask_trval = dinf_tr['maskunits_trval']

    MAX_RGCS = dinf_tr['MAX_RGCS']
    cell_types_unique = dinf_tr['cell_types_unique']
    segment_size = dinf_tr['segment_size']
    lr_inner = lr*10


    if NUM_SPLITS==0:
        local_losses,local_y_preds,local_mdl_states,local_grads_all,local_kerns,local_biases = jax.vmap(Partial(metalzero_grads1step,\
                                                                                                                  mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size,lr_inner))\
                                                                                                                  (train_x_tr,train_y_tr,train_x_val,train_y_val,
                                                                                                                   umaskcoords_trtr,umaskcoords_trval,N_trtr,N_trval,mask_trtr,mask_trval,
                                                                                                                   conv_kern_all,conv_bias_all)
                                                                                                              
    else:       # Otherwise split the vmap. This avoids running out of GPU memory when we have many retinas and large batch size
        (local_losses, local_y_preds, local_kerns, local_biases),local_grads_all = batched_metal_grads(metalzero_grads1step,
        mdl_state, global_params,MAX_RGCS,cell_types_unique,segment_size,lr_inner,train_x_tr, train_y_tr, train_x_val, train_y_val,
        umaskcoords_trtr, umaskcoords_trval, N_trtr, N_trval, mask_trtr, mask_trval, conv_kern_all, conv_bias_all, 
        NUM_SPLITS=NUM_SPLITS)
                                                                                                          
                                                                                                              
                  
    local_losses_summed = jnp.sum(local_losses)
    local_grads_summed = jax.tree_map(lambda g: jnp.sum(g,axis=0), local_grads_all)
    local_grads_summed = clip_grads(local_grads_summed)

    
    weights_output = (local_kerns,local_biases)
    
    mdl_state = mdl_state.apply_gradients(grads=local_grads_summed)
    
           
    # print(local_losses_summed)   
        
    
    """
    for key in local_grads_summed.keys():
        try:
            print('%s kernel: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['kernel']))))
        except:
            print('%s bias: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['bias']))))
    
    """

    return local_losses_summed,mdl_state,weights_output,local_grads_summed


def eval_step(state,batch_val,dinf_batch_val,n_batches=1e5):
    """
    idx_task = idx_valdset
    N_val = dinf_batch_val['N_val']
    mask_val = dinf_batch_val['maskunits_val']
    coords = dinf_batch_val['umaskcoords_val']
    segment_size =  dinf_batch_val['segment_size']
    
    """
    N_val = dinf_batch_val['N_val']
    mask_val = dinf_batch_val['maskunits_val']
    coords = dinf_batch_val['umaskcoords_val']
    segment_size =  dinf_batch_val['segment_size']

    if type(batch_val) is tuple:
        X,y = batch_val
        y_pred = state.apply_fn({'params': state.params},X,training=True)
        # loss,y_pred = task_loss_eval(state,state.params,data)
        loss,y_units,y_pred_units = calc_loss(y_pred,y,coords,segment_size,N_val,mask_val)
        y_units  = y_units[:,:N_val]
        y_pred_units = y_pred_units[:,:N_val]
        
        
        # return loss,y_pred,y,y_pred_units,y_units
    
    else:       # if the data is in dataloader format
        batch = next(iter(batch_val))
        y_shape = (*batch[0].shape[-2:],2)
        y_pred_units = jnp.empty((0,N_val))
        y_units = jnp.empty((0,N_val))
        y_pred = jnp.empty((0,*y_shape))
        y = jnp.empty((0,*y_shape))

        loss = []
        count_batch = 0
        # batch = next(iter(batch_val))
        for batch in batch_val:
            if count_batch<n_batches:
                X_batch,y_batch = batch
                y_pred_batch = state.apply_fn({'params': state.params},X_batch,training=True)
                loss_batch,y_units_b,y_pred_units_b = calc_loss(y_pred_batch,y_batch,coords,segment_size,N_val,mask_val)
                y_units_b = y_units_b[:,:N_val]
                y_pred_units_b = y_pred_units_b[:,:N_val]
                
                loss.append(loss_batch)
                y_pred = jnp.concatenate((y_pred,y_pred_batch),axis=0)
                if jnp.ndim(y_batch) != jnp.ndim(y_pred_batch):     # MEaning that resp format is individual units
                    y_batch = generate_activity_map(coords,y_batch,N_val,frame_size=y_pred_batch.shape[1:3])
                y = jnp.concatenate((y,y_batch),axis=0)
                y_pred_units = jnp.concatenate((y_pred_units,y_pred_units_b),axis=0)
                y_units = jnp.concatenate((y_units,y_units_b),axis=0)

                count_batch+=1
            else:
                break
    return loss,y_pred,y,y_pred_units,y_units



import torch

def torch_to_jax(tensor):
    """Convert a PyTorch tensor to a JAX array."""
    return jnp.array(tensor.numpy()) if isinstance(tensor, torch.Tensor) else tensor

# @jax.jit
def process_single_batch(state, X_batch, y_batch, coords, segment_size, N_val, mask_val):
    """JIT-compiled function to process a single batch."""
    y_pred_batch = state.apply_fn({'params': state.params}, X_batch, training=True)

    loss_batch, y_units_b, y_pred_units_b = calc_loss(
        y_pred_batch, y_batch, coords, segment_size, N_val, mask_val
    )
    
    y_units_b = y_units_b[:, :N_val]
    y_pred_units_b = y_pred_units_b[:, :N_val]

    return loss_batch, y_pred_batch, y_batch, y_pred_units_b, y_units_b

def eval_step_dl(state, batch_val, dinf_batch_val, n_batches=1e5):
    """Iterates over the PyTorch DataLoader and calls the JIT-compiled function."""
    N_val = dinf_batch_val['N_val']
    mask_val = dinf_batch_val['maskunits_val']
    coords = dinf_batch_val['umaskcoords_val']
    segment_size =  dinf_batch_val['segment_size']

    y_pred_list = []
    y_list = []
    y_pred_units_list = []
    y_units_list = []
    loss_list = []

    count_batch = 0

    for batch in batch_val:
        if count_batch < n_batches:
            X_batch, y_batch = batch
            X_batch, y_batch = torch_to_jax(X_batch), torch_to_jax(y_batch)  # Convert to JAX arrays

            loss_batch, y_pred_batch, y_batch, y_pred_units_b, y_units_b = process_single_batch(
                state, X_batch, y_batch, coords, segment_size, N_val, mask_val
            )

            loss_list.append(loss_batch)
            y_pred_list.append(y_pred_batch)
            y_list.append(y_batch)
            y_pred_units_list.append(y_pred_units_b)
            y_units_list.append(y_units_b)

            count_batch += 1
        else:
            break

    # Concatenate only once at the end
    y_pred = jnp.concatenate(y_pred_list, axis=0) if y_pred_list else jnp.empty((0, *y_pred_list[0].shape[1:]))
    y = jnp.concatenate(y_list, axis=0) if y_list else jnp.empty((0, *y_list[0].shape[1:]))
    y_pred_units = jnp.concatenate(y_pred_units_list, axis=0) if y_pred_units_list else jnp.empty((0, N_val))
    y_units = jnp.concatenate(y_units_list, axis=0) if y_units_list else jnp.empty((0, N_val))
    loss = jnp.array(loss_list) if loss_list else jnp.empty((0,))

    return loss, y_pred, y, y_pred_units, y_units


def generate_activity_map(cells,activity,N_val,frame_size=(40, 80)):
    """
    activity = y_batch
    cells = coords
    """
    T, C = activity[:,:N_val].shape
    N = cells.shape[0]
    
    activity_map = np.zeros((T, frame_size[0], frame_size[1], 2))
    
    for i in range(N):
        cell_id = int(cells[i, 0])
        cell_type = int(cells[i, 1])-1
        x, y = int(cells[i, 2]), int(cells[i, 3])

        activity_map[:,y,x,cell_type] = activity[:,cell_id] 

    return activity_map

def activity_regularizer(activations,alpha=1e-4):
    l1_penalty = alpha*jnp.mean(jnp.abs(activations))
    return l1_penalty

def weight_regularizer(params,alpha=1e-4):
    regularizer_exclude_list = ['BatchNorm',]
    params_subset = dict_subset(params,regularizer_exclude_list)
    
    l2_loss=0
    for w in jax.tree_leaves(params_subset):
        l2_loss = l2_loss + alpha * (w**2).mean()
    return l2_loss

def load_h5_to_nested_dict(group):
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = load_h5_to_nested_dict(item)
        else:
            result[key] = item[()]
    return result

def save_epoch(state,config,weights_output,fname_cp,weights_all=None,aux=None):
    def save_nested_dict_to_h5(group, dictionary):
        for key, item in dictionary.items():
            if isinstance(item, dict):
                # If the item is a dictionary, create a group and recurse
                subgroup = group.create_group(key)
                save_nested_dict_to_h5(subgroup, item)
            else:
                # Otherwise, create a dataset
                group.create_dataset(key, data=item)


    if os.path.exists(fname_cp):
        shutil.rmtree(fname_cp)  # Remove any existing checkpoints from the last notebook run.
    ckpt = {'model': state, 'config': config}
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_checkpointer.save(fname_cp, ckpt, save_args=save_args)
    if len(weights_output)>0:
        fname_weights_output = os.path.join(fname_cp,'weights_output.h5')
        with h5py.File(fname_weights_output,'w') as f:
            f.create_dataset('weights_output_kernel',data=np.array(weights_output[0],dtype='float32'),compression='gzip')
            f.create_dataset('weights_output_bias',data=np.array(weights_output[1],dtype='float32'),compression='gzip')

    if weights_all!=None:
        fname_weights_all = os.path.join(fname_cp,'weights_all.h5')
        with h5py.File(fname_weights_all,'w') as f:
            save_nested_dict_to_h5(f,weights_all)
    
    if aux!=None:
        fname_aux = os.path.join(fname_cp,'batch_losses.pkl')
        # with h5py.File(fname_aux,'w') as f:
        #     save_nested_dict_to_h5(f,aux)
            
        with open(fname_aux, 'wb') as f:      
            cloudpickle.dump(aux, f)
            
        # fname_aux = os.path.join(fname_cp,'batch_gradients.h5')
        # save_gradients_as_h5(aux['grads_batches'], fname_aux)


def save_gradients_as_h5(gradients, fname_aux):
    if os.path.exists(fname_aux):
        os.remove(fname_aux) 
    with h5py.File(fname_aux, 'w') as f:
        for i, grad_entry in enumerate(gradients):
            # Creating a group for each gradient entry (if you want to separate them)
            grad_group = f.create_group(f"batch_{i}")
            
            # Now iterate over the gradient dictionary
            for layer_name, _ in grad_entry.items():
                try:
                    # print(layer_name)
                    # Create a dataset for each layer's gradient (with compression)
                    layer_group = grad_group.create_group(layer_name)
                    for key,val in grad_entry[layer_name].items():
                        layer_group.create_dataset(
                            key, 
                            data=val, 
                            compression="gzip",  # you can change the compression type if needed
                            compression_opts=9  # compression level (1-9)
                        )
                except:
                    pass




    
def initialize_model(mdl,dict_params,inp_shape,lr,save_model=True,lr_schedule=None):
    classvars = list(mdl.__dataclass_fields__.keys())
    vars_intersect = list(set(classvars)&set(list(dict_params.keys())))
    config = {}
    for key in vars_intersect:
        config[key] = dict_params[key]
        
    mdl = mdl(**config)

    rng = jax.random.PRNGKey(1)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)

    inp = jnp.ones([1]+list(inp_shape))
    variables = mdl.init(rng,inp,training=False)
    # variables['batch_stats']

    if lr_schedule is None:
        optimizer = optax.adam(learning_rate = lr)
    else:
        # scheduler_fn = create_learning_rate_scheduler(lr_schedule)
        optimizer = optax.adam(learning_rate=lr_schedule)

    if 'batch_stats' in variables:
        state = TrainState.create(apply_fn=mdl.apply,params=variables['params'],tx=optimizer,batch_stats=variables['batch_stats'])
    # elif 'non_trainable' in variables:
    #     state = TrainState.create(apply_fn=mdl.apply,params=variables['params'],tx=optimizer,non_trainable=variables['non_trainable'])
    else:
        state = TrainState.create(apply_fn=mdl.apply,params=variables['params'],tx=optimizer)

    ckpt = {'mdl': state, 'config': config}

    if save_model==True:
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        
    return state,mdl,config


def load(mdl,variables,lr):
    optimizer = optax.adam(learning_rate = lr)
    mdl_state = TrainState.create(apply_fn=mdl.apply,params=variables['params'],tx=optimizer)
    return mdl_state


def batched_metal_grads(fn,mdl_state, global_params,MAX_RGCS,cell_types_unique,segment_size, *inputs, NUM_SPLITS=2):
    """
    fn = metal_grads
    inputs = (train_x_tr, train_y_tr, train_x_val, train_y_val,
    umaskcoords_trtr, umaskcoords_trval, N_trtr, N_trval, mask_trtr, mask_trval, conv_kern_all, conv_bias_all)
    """
    # Manually split data into `num_splits` roughly equal groups
    split_sizes = np.array_split(np.arange(len(inputs[0])), NUM_SPLITS)  # Index-based split
    results = []
    grads_append=[]

    for idxs in split_sizes:
        batch = [jnp.array(inp[idxs]) for inp in inputs]  # Extract sub-batch
        batch = tuple(jnp.atleast_1d(b) for b in batch)   # Ensure at least 1D
        local_losses,local_y_preds,local_mdl_states,local_grads_all,local_kerns,local_biases = jax.vmap(partial(fn, mdl_state, global_params,MAX_RGCS,cell_types_unique,segment_size))(*batch)
        result = (local_losses,local_y_preds,local_kerns,local_biases)
        grads_append.append(local_grads_all)
        results.append(result)
    # THE ISSUE IS WITH LOCAL_GRADS_ALL AS ITS A DICT. SO I JUST NEED TO DEAL WITH THIS SEPERATELY IN THE FOLLOWING LINE
    # Concatenate results along batch axis
    grads_cat = jax.tree_map(lambda *x: jnp.concatenate(x, axis=0), *grads_append)
    results_cat = tuple(jnp.concatenate(r, axis=0) for r in zip(*results))

    return results_cat,grads_cat

# %% Training func

def train_step(mdl_state,weights_output,config,training_params,dataloader_train,dataloader_val,dinf_tr,dinf_val,nb_epochs,path_model_save,save=False,lr_schedule=None,step_start=0,
               APPROACH='metal',idx_valdset=0,runOnCluster=0):
    """
    RESP_FORMAT='MAPS'
    RESP_FORMAT='UNITS'

    """
    print('Training scheme: %s'%APPROACH)
    save = True
    step_start=0
    
    
    # learning_rate_fn = create_learning_rate_scheduler(lr_schedule)
    
    loss_epoch_train = []
    loss_epoch_val = []
    
    loss_batch_train = []
    loss_batch_val = []
    
    fev_epoch_train = []
    fev_epoch_val = []

    idx_valdset = 0#7
    dinf_batch_val = jax.tree_map(lambda x: x[idx_valdset] if isinstance(x, np.ndarray) else x, dinf_val)
    dinf_batch_valtr = dict(N_val=dinf_tr['N_trtr'][idx_valdset],
                            maskunits_val=dinf_tr['maskunits_trtr'][idx_valdset],
                            umaskcoords_val=dinf_tr['umaskcoords_trtr'][idx_valdset],
                            segment_size=dinf_tr['segment_size'])

    n_batches = len(dataloader_train)
    print('Total batches: %d'%n_batches)
    epoch=0
    steps_total = nb_epochs*n_batches
    pbar = tqdm(total=steps_total, desc="Processing Steps")  # Initialize progress bar
    
    
    # cp_interval=training_params['cp_interval']

    ctr_step = -1
    step=0
    for epoch in range(step_start,nb_epochs):
        _ = gc.collect()
        loss_batch_train=[]
        # batch_train = next(iter(dataloader_train)); batch=batch_train; 
        ctr_batch=-1
        ctr_batch_master = -1
        t_dl = time.time()
        loss_step_train=[]
        for batch_train in dataloader_train:
            ctr_step = ctr_step+1

            t_dl = time.time()-t_dl
            ctr_batch = ctr_batch+1
            ctr_batch_master=ctr_batch_master+1
            
            current_lr = lr_schedule(mdl_state.step)     
            
            t_tr=time.time()
            if APPROACH == 'metalzero':
                loss,mdl_state,weights_output,grads = train_step_metalzero(mdl_state,batch_train,weights_output,current_lr,dinf_tr)
            elif APPROACH == 'metalzero1step':
                loss,mdl_state,weights_output,grads = train_step_metalzero1step(mdl_state,batch_train,weights_output,current_lr,dinf_tr)
            elif APPROACH == 'metalzeroperturb':
                loss,mdl_state,weights_output,grads = train_step_metalzeroperturb(mdl_state,batch_train,weights_output,current_lr,dinf_tr)

                
            t_tr = time.time()-t_tr
            t_other = time.time()
            loss_batch_train.append(loss)
            loss_step_train.append(loss)
            # if ctr_batch_master==0 or ctr_batch==10:
            #     ctr_batch=0
            # else:
                
            # gc.collect()
            t_other = time.time()-t_other
            pbar.set_postfix({"Epoch": epoch, "Loss": f"{loss:.2f}", "LR": f"{np.array(current_lr):.3E}"})
            # print('Epoch %d | Loss: %0.2f | LR: %0.3E'%(epoch,loss,np.array(current_lr)))

            pbar.update(1)
            
            
            if ctr_step%training_params['cp_interval']==0:# or ctr_step%n_batches==0:         # Save if either cp interval reached or end of epoch reached
                # print('Epoch %d | Loss: %0.2f | LR: %0.3E'%(epoch,loss,np.array(current_lr)))
                grads_cpu = to_cpu(grads)
                del grads

                aux = dict(loss_batch_train=np.array(loss_step_train),grads=grads_cpu)
                # t=time.time()
                if save == True:
                    fname_cp = os.path.join(path_model_save,'step-%03d'%ctr_step)
                    save_epoch(mdl_state,config,weights_output,fname_cp,aux=aux)
                    loss_step_train = []
                    
                # elap = time.time()-t
                # print('File saving time: %f mins',elap/60)

            t_dl = time.time()


        # assert jnp.sum(grads['Conv_0']['kernel']) != 0, 'Gradients are Zero'
        
        # print('Finished training on batch')
        # print('Gonna start ealuating the batch')
        
        # For validation, update the new state with weights from the idx_valdset task
        mdl_state_val = mdl_state
        if APPROACH == 'metal':
            mdl_state_val.params['output']['kernel'] = weights_output[0][idx_valdset]
            mdl_state_val.params['output']['bias'] = weights_output[1][idx_valdset]
            
    
        loss_batch_val = []
        # batch = next(iter(dataloader_val))
        # t = time.time()
        # for batch_val in dataloader_val:
        loss_batch_val,y_pred,y,y_pred_val_units,y_val_units = eval_step(mdl_state_val,dataloader_val,dinf_batch_val)
        
        # elap = time.time()-t
        # print('Val time: %f',elap)

        # t = time.time()
        loss_batch_train_test,y_pred_train,y_train,y_pred_train_units,y_train_units = eval_step(mdl_state_val,(batch_train[0][idx_valdset],batch_train[1][idx_valdset]),dinf_batch_valtr)
        
        loss_currEpoch_master = np.mean(loss_batch_train)
        loss_currEpoch_train = np.mean(loss_batch_train_test)
        loss_currEpoch_val = np.mean(loss_batch_val)
    
        loss_epoch_train.append(np.mean(loss_currEpoch_train))
        loss_epoch_val.append(np.mean(loss_currEpoch_val))
        
        current_lr = lr_schedule(mdl_state.step)
        
        temporal_width_eval = batch_train[0].shape[1]
        fev_val,_,predCorr_val,_ = model_evaluate_new(y_val_units,y_pred_val_units,temporal_width_eval,lag=0,obs_noise=0)
        fev_val_med,predCorr_val_med = np.median(fev_val),np.median(predCorr_val)
        fev_train,_,predCorr_train,_ = model_evaluate_new(y_train_units,y_pred_train_units,temporal_width_eval,lag=0,obs_noise=0)
        fev_train_med,predCorr_train_med = np.median(fev_train),np.median(predCorr_train)
        
        fev_epoch_train.append(fev_train_med)
        fev_epoch_val.append(fev_val_med)

        print('Epoch: %d, global_loss: %.2f || local_train_loss: %.2f, fev: %.2f, corr: %.2f || local_val_loss: %.2f, fev: %.2f, corr: %.2f || lr: %.2e'\
              %(epoch+1,loss_currEpoch_master,loss_currEpoch_train,fev_train_med,predCorr_train_med,loss_currEpoch_val,fev_val_med,predCorr_val_med,current_lr))
        
            
        # fig,axs = plt.subplots(2,1,figsize=(20,10));axs=np.ravel(axs);fig.suptitle('Epoch: %d'%(epoch+1))
        # axs[0].plot(y_train_units[:200,2]);axs[0].plot(y_pred_train_units[:200,10]);axs[0].set_title('Train')
        # axs[1].plot(y_units[:,10]);axs[1].plot(y_pred_units[:,10]);axs[1].set_title('Validation')
        # plt.show()
        # plt.close()
        

    return loss_currEpoch_master,loss_epoch_train,loss_epoch_val,mdl_state,weights_output,fev_epoch_train,fev_epoch_val


# %% Finetuning
MODE = 'MAP'
LOSS_FUN_FT = 'madactivity'

@jax.jit
def ft_calc_loss(y_pred,y,coords,segment_size,N_tr,mask_tr):
    y_pred_units = pred_psfavg(y_pred,coords,segment_size)
    if jnp.ndim(y) == jnp.ndim(y_pred): # That is both are in terms of MAPS
        y_units = pred_psfavg(y,coords,segment_size)      # This is just going to be the actual value at a single pixel
    else:
        y_units=y

    y_pred_units = jnp.where(y_pred_units == 0, 1e-6, y_pred_units)

    # y_pred_units = jnp.where(y_pred_units == 0, 1e-6, y_pred_units)
    if LOSS_FUN_FT=='poisson':
        loss = y_pred_units-y_units*jax.lax.log(y_pred_units)
    elif LOSS_FUN_FT=='poissonreg':
        poisson_loss = y_pred_units-y_units*jax.lax.log(y_pred_units)
        reg_loss = 1e-1*y_pred_units
        loss = poisson_loss+reg_loss
    elif LOSS_FUN_FT=='madpoissonreg':
        poisson_loss = y_pred_units-y_units*jax.lax.log(y_pred_units)
        mad_loss = jnp.abs(y_units-y_pred_units)
        reg_loss = 1e-1*y_pred_units
        loss = poisson_loss+mad_loss+reg_loss
    elif LOSS_FUN_FT=='mad':
        loss = jnp.abs(y_units-y_pred_units)
    elif LOSS_FUN_FT=='madreg':
        mad_loss = jnp.abs(y_units-y_pred_units)
        reg_loss = 1e-2*y_pred_units
        loss = mad_loss+reg_loss
    elif LOSS_FUN_FT=='mse':
        loss = (y_units-y_pred_units)**2
    elif LOSS_FUN_FT=='wmad':
        weight_factor=10
        threshold=0.1
        weights = jnp.where(y_units > threshold, weight_factor, 1.0)
        loss = weights * jnp.abs(y_units - y_pred_units)
    elif LOSS_FUN_FT=='rmad':
        eps=1e-3
        loss = jnp.abs(y_units - y_pred_units)/(jnp.abs(y_units)+eps)
    elif LOSS_FUN_FT=='madactivity':
        min_activity = 0.1
        activity_loss = jnp.square(y_pred_units - min_activity)
        loss = jnp.abs(y_units-y_pred_units)
        loss = loss+activity_loss

    else:
        raise Exception('Loss Function Not Found')


    loss = (loss*mask_tr[None,:])
    loss=jnp.nansum(loss)/(N_tr*loss.shape[0])
    
    return loss,y_units,y_pred_units

@jax.jit
def ft_task_loss(state,trainable_params,fixed_params,batch,coords,N_tr,segment_size,mask_tr):
    """
    fixed_params = ft_params_fixed
    trainable_params = ft_mdl_state.params
    batch=batch_train
    state=ft_mdl_state
    coords=dinf_tr['umaskcoords_trtr']
    N_tr = dinf_tr['N_trtr']
    segment_size = dinf_tr['segment_size']
    mask_tr = dinf_tr['maskunits_trtr']
    """
    X,y = batch
    y_pred,state = state.apply_fn({'params': {**fixed_params, **trainable_params}},X,training=True,mutable=['intermediates'])
    intermediates = state['intermediates']
    dense_activations = intermediates['dense_activations'][0]

    # if training==True:
    loss,y_units,y_pred_units = ft_calc_loss(y_pred,y,coords,segment_size,N_tr,mask_tr)
    reg = 0#0.001*jnp.sum(jnp.abs(dense_activations-0.5))
    loss = loss + reg#weight_regularizer(trainable_params,alpha=1e-3)
    return loss,y_pred

@jax.jit
def ft_train_step(state,fixed_params,batch,coords,N_tr,segment_size,mask_tr):
    
    grad_fn = jax.value_and_grad(ft_task_loss,argnums=1,has_aux=True)
    (loss,y_pred),grads = grad_fn(state,state.params,fixed_params,batch,coords,N_tr,segment_size,mask_tr)
    grads = clip_grads(grads)
    state = state.apply_gradients(grads=grads)
    
    return state,loss,grads

def ft_eval_step(state,ft_params_fixed,batch_val,dinf_batch_val,n_batches=1e5):
    """
    idx_task = idx_valdset
    N_val = dinf_batch_val['N_val']
    mask_val = dinf_batch_val['maskunits_val']
    coords = dinf_batch_val['umaskcoords_val']
    segment_size =  dinf_batch_val['segment_size']
    
    """
    N_val = dinf_batch_val['N_val']
    mask_val = dinf_batch_val['maskunits_val']
    coords = dinf_batch_val['umaskcoords_val']
    segment_size =  dinf_batch_val['segment_size']

    if type(batch_val) is tuple:
        X,y = batch_val
        y_pred = state.apply_fn({'params': {**state.params,**ft_params_fixed}},X,training=True)
        # loss,y_pred = task_loss_eval(state,state.params,data)
        loss,y_units,y_pred_units = ft_calc_loss(y_pred,y,coords,segment_size,N_val,mask_val)
        y_units  = y_units[:,:N_val]
        y_pred_units = y_pred_units[:,:N_val]
        
        
        # return loss,y_pred,y,y_pred_units,y_units
    
    else:       # if the data is in dataloader format
        batch = next(iter(batch_val))
        y_shape = (*batch[0].shape[-2:],2)
        y_pred_units = jnp.empty((0,N_val))
        y_units = jnp.empty((0,N_val))
        y_pred = jnp.empty((0,*y_shape))
        y = jnp.empty((0,*y_shape))

        loss = []
        count_batch = 0
        # batch = next(iter(batch_val))
        for batch in batch_val:
            if count_batch<n_batches:
                X_batch,y_batch = batch
                y_pred_batch = state.apply_fn({'params': {**state.params,**ft_params_fixed}},X_batch,training=True)
                loss_batch,y_units_b,y_pred_units_b = ft_calc_loss(y_pred_batch,y_batch,coords,segment_size,N_val,mask_val)
                y_units_b = y_units_b[:,:N_val]
                y_pred_units_b = y_pred_units_b[:,:N_val]
                
                loss.append(loss_batch)
                y_pred = jnp.concatenate((y_pred,y_pred_batch),axis=0)
                if jnp.ndim(y_batch) != jnp.ndim(y_pred_batch):     # MEaning that resp format is individual units
                    y_batch = generate_activity_map(coords,y_batch,N_val,frame_size=y_pred_batch.shape[1:3])
                y = jnp.concatenate((y,y_batch),axis=0)
                y_pred_units = jnp.concatenate((y_pred_units,y_pred_units_b),axis=0)
                y_units = jnp.concatenate((y_units,y_units_b),axis=0)

                count_batch+=1
            else:
                break
    return loss,y_pred,y,y_pred_units,y_units


def ft_train(ft_mdl_state,ft_params_fixed,config,training_params,dataloader_train,dataloader_val,dinf_tr,dinf_val,nb_epochs,ft_path_model_save,save=False,lr_schedule=None,step_start=0,idx_valdset=0):
    """
    RESP_FORMAT='MAPS'
    RESP_FORMAT='UNITS'

    """
    
    print('Training scheme: Finetuning')
    save = True
    step_start=0
    
    
    # learning_rate_fn = create_learning_rate_scheduler(lr_schedule)
    
    loss_epoch_train = []
    loss_epoch_val = []
    
    loss_batch_train = []
    loss_batch_val = []
    
    fev_epoch_train = []
    fev_epoch_val = []

    dinf_batch_valtr = dict(N_val=dinf_tr['N_trtr'],
                            maskunits_val=dinf_tr['maskunits_trtr'],
                            umaskcoords_val=dinf_tr['umaskcoords_trtr'],
                            segment_size=dinf_tr['segment_size'])

    n_batches = len(dataloader_train)
    print('Total batches: %d'%n_batches)
    epoch=0
    steps_total = nb_epochs*n_batches
    pbar = tqdm(total=steps_total, desc="Processing Steps")  # Initialize progress bar
    
    
    # cp_interval=training_params['cp_interval']

    ctr_step = training_params['cp_interval']*step_start
    step=0
    for epoch in range(step_start,nb_epochs):
        _ = gc.collect()
        loss_batch_train=[]
        # batch_train = next(iter(dataloader_train)); batch=batch_train; 
        ctr_batch=-1
        ctr_batch_master = -1
        t_dl = time.time()
        loss_step_train=[]
        for batch_train in dataloader_train:
            ctr_step = ctr_step+1

            ctr_batch = ctr_batch+1
            ctr_batch_master=ctr_batch_master+1
            
            current_lr = lr_schedule(ft_mdl_state.step)     
            
            t_tr=time.time()
            ft_mdl_state,loss,grads = ft_train_step(ft_mdl_state,ft_params_fixed,batch_train,
                                                                dinf_tr['umaskcoords_trtr'],dinf_tr['N_trtr'],dinf_tr['segment_size'],dinf_tr['maskunits_trtr']
                                                                )
                
            loss_batch_train.append(loss)
            loss_step_train.append(loss)
            pbar.set_postfix({"Epoch": epoch, "Loss": f"{loss:.2f}", "LR": f"{np.array(current_lr):.3E}"})
            pbar.update(1)

            # print('Epoch %d | Loss: %0.2f | LR: %0.3E'%(epoch,loss,np.array(current_lr)))

            
            
            if ctr_step%training_params['cp_interval']==0 or ctr_step%n_batches==0 or ctr_step==1:         # Save if either cp interval reached or end of epoch reached
            
                # print('Epoch %d | Loss: %0.2f | LR: %0.3E'%(epoch,loss,np.array(current_lr)))
                grads_cpu = to_cpu(grads)
                del grads

                aux = dict(loss_batch_train=np.array(loss_step_train),grads=grads_cpu)
                # t=time.time()
                if save == True:
                    weights_output=[]
                    fname_cp = os.path.join(ft_path_model_save,'step-%03d'%ctr_step)
                    save_epoch(ft_mdl_state,config,weights_output,fname_cp,aux=aux)
                    loss_step_train = []
                    
                # elap = time.time()-t
                # print('File saving time: %f mins',elap/60)



        # assert jnp.sum(grads['Conv_0']['kernel']) != 0, 'Gradients are Zero'
        
        _=gc.collect()
        # For validation, update the new state with weights from the idx_valdset task            
    
        loss_batch_val = []
        # batch = next(iter(dataloader_val))
        # t = time.time()
        loss_batch_val,y_pred,y,y_pred_val_units,y_val_units = ft_eval_step(ft_mdl_state,ft_params_fixed,dataloader_val,dinf_val)
        
        # elap = time.time()-t
        # print('Val time: %f',elap)

        # t = time.time()
        loss_batch_train_test,y_pred_train,y_train,y_pred_train_units,y_train_units = ft_eval_step(ft_mdl_state,ft_params_fixed,(batch_train[0],batch_train[1]),dinf_batch_valtr)
        
        loss_currEpoch_master = np.mean(loss_batch_train)
        loss_currEpoch_train = np.mean(loss_batch_train_test)
        loss_currEpoch_val = np.mean(loss_batch_val)
    
        loss_epoch_train.append(np.mean(loss_currEpoch_train))
        loss_epoch_val.append(np.mean(loss_currEpoch_val))
        
        current_lr = lr_schedule(ft_mdl_state.step)
        
        temporal_width_eval = batch_train[0].shape[1]
        fev_val,_,predCorr_val,_ = model_evaluate_new(y_val_units,y_pred_val_units,temporal_width_eval,lag=0,obs_noise=0)
        fev_val_med,predCorr_val_med = np.median(fev_val),np.median(predCorr_val)
        fev_train,_,predCorr_train,_ = model_evaluate_new(y_train_units,y_pred_train_units,temporal_width_eval,lag=0,obs_noise=0)
        fev_train_med,predCorr_train_med = np.median(fev_train),np.median(predCorr_train)
        
        fev_epoch_train.append(fev_train_med)
        fev_epoch_val.append(fev_val_med)

        print('Epoch: %d, train_loss: %.2f, fev: %.2f, corr: %.2f || val_loss: %.2f, fev: %.2f, corr: %.2f || lr: %.2e'\
              %(epoch+1,loss_currEpoch_master,fev_train_med,predCorr_train_med,loss_currEpoch_val,fev_val_med,predCorr_val_med,current_lr))
        
            
        fig,axs = plt.subplots(2,1,figsize=(20,10));axs=np.ravel(axs);fig.suptitle('Epoch: %d'%(epoch+1))
        axs[0].plot(y_train_units[:600,0]);axs[0].plot(y_pred_train_units[:600,0]);axs[0].set_title('Train')
        axs[1].plot(y_val_units[:,0]);axs[1].plot(y_pred_val_units[:,0]);axs[1].set_title('Validation')
        plt.show()
        plt.close()
        

    return loss_currEpoch_master,loss_epoch_train,loss_epoch_val,ft_mdl_state,fev_epoch_train,fev_epoch_val

# %% Recycle

def train(mdl_state,weights_output,config,training_params,dataloader_train,dataloader_val,dinf_tr,dinf_val,nb_epochs,path_model_save,save=False,lr_schedule=None,step_start=0,APPROACH='metal',idx_valdset=0):
    """
    RESP_FORMAT='MAPS'
    RESP_FORMAT='UNITS'

    """
    print('Training scheme: %s'%APPROACH)
    save = True
    step_start=0
    
    
    # learning_rate_fn = create_learning_rate_scheduler(lr_schedule)
    
    loss_epoch_train = []
    loss_epoch_val = []
    
    loss_batch_train = []
    loss_batch_val = []
    
    fev_epoch_train = []
    fev_epoch_val = []

    idx_valdset = 2#7
    dinf_batch_val = jax.tree_map(lambda x: x[idx_valdset] if isinstance(x, np.ndarray) else x, dinf_val)
    dinf_batch_valtr = dict(N_val=dinf_tr['N_trtr'][idx_valdset],
                            maskunits_val=dinf_tr['maskunits_trtr'][idx_valdset],
                            umaskcoords_val=dinf_tr['umaskcoords_trtr'][idx_valdset],
                            segment_size=dinf_tr['segment_size'])


    print('Total batches: %d'%len(dataloader_train))
    epoch=0
    for epoch in tqdm(range(step_start,nb_epochs)):
        _ = gc.collect()
        loss_batch_train=[]
        # batch_train = next(iter(dataloader_train)); batch=batch_train; 
        grads_batches = []
        ctr_batch=-1
        ctr_batch_master = -1
        t_dl = time.time()
        for batch_train in dataloader_train:
            t_dl = time.time()-t_dl
            ctr_batch = ctr_batch+1
            ctr_batch_master=ctr_batch_master+1
            
            current_lr = lr_schedule(mdl_state.step)     
            
            t_tr=time.time()
            if APPROACH == 'metalzero':
                loss,mdl_state,weights_output,grads = train_step_metalzero(mdl_state,batch_train,weights_output,current_lr,dinf_tr)
            elif APPROACH == 'metalzero1step':
                loss,mdl_state,weights_output,grads = train_step_metalzero1step(mdl_state,batch_train,weights_output,current_lr,dinf_tr)
                
            t_tr = time.time()-t_tr
            t_other = time.time()
            loss_batch_train.append(loss)
            if ctr_batch_master==0 or ctr_batch==2000:
                ctr_batch=0
                grads_cpu = to_cpu(grads)
                grads_batches.append(grads_cpu)
            else:
                del grads
                
            # gc.collect()
            t_other = time.time()-t_other
            print('Epoch %d, Batch %d of %d | Loss: 0.3%f | DL: %0.2f s, TR: %0.2f s, Other: %0.2f s'%(epoch,ctr_batch_master,len(dataloader_train),loss,t_dl,t_tr,t_other))
            t_dl = time.time()
            

        # assert jnp.sum(grads['Conv_0']['kernel']) != 0, 'Gradients are Zero'
        
        print('Finished training on batch')
        print('Gonna start ealuating the batch')
        
        # For validation, update the new state with weights from the idx_valdset task
        mdl_state_val = mdl_state
        if APPROACH == 'metal':
            mdl_state_val.params['output']['kernel'] = weights_output[0][idx_valdset]
            mdl_state_val.params['output']['bias'] = weights_output[1][idx_valdset]
            
    
        loss_batch_val = []
        # batch = next(iter(dataloader_val))
        # t = time.time()
        # for batch_val in dataloader_val:
        loss_batch_val,y_pred,y,y_pred_val_units,y_val_units = eval_step(mdl_state_val,dataloader_val,dinf_batch_val)
        
        # elap = time.time()-t
        # print('Val time: %f',elap)

        # t = time.time()
        loss_batch_train_test,y_pred_train,y_train,y_pred_train_units,y_train_units = eval_step(mdl_state_val,(batch_train[0][idx_valdset],batch_train[1][idx_valdset]),dinf_batch_valtr)
        
        loss_currEpoch_master = np.mean(loss_batch_train)
        loss_currEpoch_train = np.mean(loss_batch_train_test)
        loss_currEpoch_val = np.mean(loss_batch_val)
    
        loss_epoch_train.append(np.mean(loss_currEpoch_train))
        loss_epoch_val.append(np.mean(loss_currEpoch_val))
        
        current_lr = lr_schedule(mdl_state.step)
        
        temporal_width_eval = batch_train[0].shape[1]
        fev_val,_,predCorr_val,_ = model_evaluate_new(y_val_units,y_pred_val_units,temporal_width_eval,lag=0,obs_noise=0)
        fev_val_med,predCorr_val_med = np.median(fev_val),np.median(predCorr_val)
        fev_train,_,predCorr_train,_ = model_evaluate_new(y_train_units,y_pred_train_units,temporal_width_eval,lag=0,obs_noise=0)
        fev_train_med,predCorr_train_med = np.median(fev_train),np.median(predCorr_train)
        
        fev_epoch_train.append(fev_train_med)
        fev_epoch_val.append(fev_val_med)

        print('Epoch: %d, global_loss: %.2f || local_train_loss: %.2f, fev: %.2f, corr: %.2f || local_val_loss: %.2f, fev: %.2f, corr: %.2f || lr: %.2e'\
              %(epoch+1,loss_currEpoch_master,loss_currEpoch_train,fev_train_med,predCorr_train_med,loss_currEpoch_val,fev_val_med,predCorr_val_med,current_lr))
        
            
        # fig,axs = plt.subplots(2,1,figsize=(20,10));axs=np.ravel(axs);fig.suptitle('Epoch: %d'%(epoch+1))
        # axs[0].plot(y_train_units[:200,2]);axs[0].plot(y_pred_train_units[:200,10]);axs[0].set_title('Train')
        # axs[1].plot(y_units[:,10]);axs[1].plot(y_pred_units[:,10]);axs[1].set_title('Validation')
        # plt.show()
        # plt.close()

        aux = dict(loss_batch_train=np.array(loss_batch_train),grads_batches=grads_batches)
        t=time.time()
        if save == True:
            fname_cp = os.path.join(path_model_save,'epoch-%03d'%epoch)
            save_epoch(mdl_state,config,weights_output,fname_cp,aux=aux)
            
        elap = time.time()-t
        print('File saving time: %f mins',elap/60)
            
    return loss_currEpoch_master,loss_epoch_train,loss_epoch_val,mdl_state,weights_output,fev_epoch_train,fev_epoch_val

# @jax.jit
# def train_step_metalzero(mdl_state,batch,weights_output,lr,dinf_tr):        # Make unit vectors then scale by num of RGCs
#     """
#     State is the grand model state that actually gets updated
#     state_task is the "state" after gradients are applied for a specific task
#         task_idx = 1
#         conv_kern = conv_kern_all[task_idx]
#         conv_bias = conv_bias_all[task_idx]
#         train_x = train_x[task_idx]
#         train_y_tr = train_y_tr[task_idx]
#         train_y_val = train_y_val[task_idx]
#         coords_tr = umaskcoords_trtr[task_idx]
#         coords_val = umaskcoords_trval[task_idx]
#         N_tr = N_trtr[task_idx]
#         N_val = N_trval[task_idx]
#         mask_tr = mask_trtr[task_idx]
#         mask_val = mask_trval[task_idx]
        
#         loss,mdl_state,weights_output,grads = train_step_metal(mdl_state,batch_train,weights_output,current_lr,dinf_tr)
#     """
#     @jax.jit
#     def metalzero_grads(mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size,train_x,train_y_tr,train_y_val,coords_tr,coords_val,N_tr,N_val,mask_tr,mask_val):

#         # Split the batch into inner and outer training sets
#         # PARAMETERIZE this
#         frac_s_train = 0.5
#         len_data = train_x.shape[0]
#         len_s_train = int(len_data*frac_s_train)
        
#         batch_train = (train_x[:len_s_train],train_y_tr[:len_s_train])
#         batch_val = (train_x[len_s_train:],train_y_val[len_s_train:])
#         # N_points_val = N_task_val*segment_size

#         # Make local model by using global params but local dense layer weights
#         local_params = global_params.copy()
#         # local_params['output']['kernel'] = conv_kern
#         # local_params['output']['bias'] = conv_bias
#         local_mdl_state = mdl_state.replace(params=local_params)

#         # Calculate gradients of the local model wrt to local params    
#         grad_fn = jax.value_and_grad(task_loss,argnums=1,has_aux=True)
#         (local_loss_train,y_pred_train),local_grads = grad_fn(local_mdl_state,local_params,batch_train,coords_tr,N_tr,segment_size,mask_tr)
        
#         # scale the local gradients according to ADAM's first step. Helps to stabilize
#         # And update the parameters
#         local_params = jax.tree_map(lambda p, g: p - lr*(g/(jnp.abs(g)+1e-8)), local_params, local_grads)

#         # Calculate gradients of the loss of the resulting local model but using the validation set
#         # local_mdl_state = mdl_state.replace(params=local_params)
#         (local_loss_val,y_pred_val),local_grads_val = grad_fn(local_mdl_state,local_params,batch_val,coords_val,N_val,segment_size,mask_val)
        
#         # Update only the Dense layer weights since we retain it
#         # local_params_val = jax.tree_map(lambda p, g: p - lr*(g/(jnp.abs(g)+1e-8)), local_params, local_grads_val)
        
#         # Get the direction of generalization
#         # local_grads_total = jax.tree_map(lambda g_1, g_2: g_1+g_2, local_grads,local_grads_val)
        
#         # Normalize the grads to unit vector
#         # local_grads_total = jax.tree_map(lambda g: g/jnp.linalg.norm(g), local_grads_total)
        
#         # Scale vectors by num of RGCs
#         # scaleFac = (N_tr+N_val)/MAX_RGCS
#         # local_grads_total = jax.tree_map(lambda g: g*scaleFac, local_grads_total)



#         # Record dense layer weights
#         # conv_kern = local_params_val['output']['kernel']
#         # conv_bias = local_params_val['output']['bias']
        
#         return local_loss_val,y_pred_val,local_mdl_state
    
    
#     @jax.jit
#     def metalzero_loss(mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size,train_x,train_y_tr,train_y_val,umaskcoords_trtr,umaskcoords_trval,N_trtr,N_trval,mask_trtr,mask_trval):
        
#         local_losses,local_y_preds,local_mdl_states = jax.vmap(Partial(metalzero_grads,mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size))\
#                                                                                                                   (train_x,train_y_tr,train_y_val,
#                                                                                                                    umaskcoords_trtr,umaskcoords_trval,N_trtr,N_trval,mask_trtr,mask_trval)

#         losses = local_losses.sum()
        
#         return losses,(local_y_preds)
    
#     """
#     batch_train = next(iter(dataloader_train)); batch=batch_train; 
#     """
#     global_params = mdl_state.params
    
#     train_x,train_y_tr,train_y_val = batch
#     conv_kern_all,conv_bias_all = weights_output
#     umaskcoords_trtr = dinf_tr['umaskcoords_trtr']
#     umaskcoords_trval = dinf_tr['umaskcoords_trval']
#     N_trtr = dinf_tr['N_trtr']
#     N_trval = dinf_tr['N_trval']
#     mask_trtr = dinf_tr['maskunits_trtr']
#     mask_trval = dinf_tr['maskunits_trval']

#     MAX_RGCS = dinf_tr['MAX_RGCS']
#     cell_types_unique = dinf_tr['cell_types_unique']
#     segment_size = dinf_tr['segment_size']

#     # maxRGCs = mask_unitsToTake_all.shape[-1] #jnp.sum(mask_unitsToTake_all)
    
#     grad_fn = jax.value_and_grad(metalzero_loss,argnums=1,has_aux=True)
#     (losses,rgb),grads = grad_fn(mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size,train_x,train_y_tr,train_y_val,umaskcoords_trtr,umaskcoords_trval,N_trtr,N_trval,mask_trtr,mask_trval)

    
#     mdl_state = mdl_state.apply_gradients(grads=grads)
#     conv_kern = mdl_state.params['output']['kernel']
#     conv_bias = mdl_state.params['output']['bias']
#     weights_output = (conv_kern,conv_bias)

           
#     # print(local_losses_summed)   
        
    
#     """
#     for key in local_grads_summed.keys():
#         try:
#             print('%s kernel: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['kernel']))))
#         except:
#             print('%s bias: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['bias']))))
    
#     """

#     return losses,mdl_state,weights_output,grads




# @jax.jit
# def train_step_metal(mdl_state,batch,weights_output,lr,dinf_tr):        # Make unit vectors then scale by num of RGCs
#     """
#     State is the grand model state that actually gets updated
#     state_task is the "state" after gradients are applied for a specific task
#         task_idx = 1
#         conv_kern = conv_kern_all[task_idx]
#         conv_bias = conv_bias_all[task_idx]
#         train_x_tr = train_x_tr[task_idx]
#         train_y_tr = train_y_tr[task_idx]
#         train_x_val = train_x_val[task_idx]
#         train_y_val = train_y_val[task_idx]
#         coords_tr = umaskcoords_trtr[task_idx]
#         coords_val = umaskcoords_trval[task_idx]
#         N_tr = N_trtr[task_idx]
#         N_val = N_trval[task_idx]
#         mask_tr = mask_trtr[task_idx]
#         mask_val = mask_trval[task_idx]
        
#         loss,mdl_state,weights_output,grads = train_step_metal(mdl_state,batch_train,weights_output,current_lr,dinf_tr)
#     """
#     @jax.jit
#     def metal_grads(mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size,train_x_tr,train_y_tr,train_x_val,train_y_val,coords_tr,coords_val,N_tr,N_val,mask_tr,mask_val,conv_kern,conv_bias):

#         batch_train = (train_x_tr,train_y_tr)
#         batch_val = (train_x_val,train_y_val)


#         # Make local model by using global params but local dense layer weights
#         local_params = {**global_params, 'output': {'kernel': conv_kern, 'bias': conv_bias}}
#         local_mdl_state = mdl_state.replace(params=local_params)

#         # Calculate gradients of the local model wrt to local params    
#         grad_fn = jax.value_and_grad(task_loss,argnums=1,has_aux=True)
#         (local_loss_train,_),local_grads = grad_fn(local_mdl_state,local_params,batch_train,coords_tr,N_tr,segment_size,mask_tr)
        
#         # scale the local gradients according to ADAM's first step. Helps to stabilize
#         # And update the parameters
#         update_fn = lambda p, g: p - lr * (g / (jnp.abs(g) + 1e-8))
#         local_params = jax.tree_map(update_fn, local_params, local_grads)


#         # Calculate gradients of the loss of the resulting local model but using the validation set
#         # local_mdl_state = mdl_state.replace(params=local_params)
#         (local_loss_val,y_pred_val),local_grads_val = grad_fn(local_mdl_state,local_params,batch_val,coords_val,N_val,segment_size,mask_val)
        
#         # Update only the Dense layer weights since we retain it
#         local_params_val = jax.tree_map(update_fn, local_params, local_grads_val)

        
#         # Get the direction of generalization
#         local_grads_total = jax.tree_map(jnp.add, local_grads, local_grads_val)

        
#         # Normalize the grads to unit vector
#         local_grads_total = jax.tree_map(lambda g: g/jnp.linalg.norm(g), local_grads_total)
        
#         # Scale vectors by num of RGCs
#         scaleFac = (N_tr+N_val)/MAX_RGCS
#         local_grads_total = jax.tree_map(lambda g: g*scaleFac, local_grads_total)



#         # Record dense layer weights
#         conv_kern = local_params_val['output']['kernel']
#         conv_bias = local_params_val['output']['bias']
        
#         return local_loss_val,y_pred_val,local_mdl_state,local_grads_total,conv_kern,conv_bias
    

#     """
#     batch_train = next(iter(dataloader_train)); batch=batch_train; 
#     """
#     NUM_SPLITS = 0  # Try different values like 3 if needed

#     global_params = mdl_state.params
    
#     train_x_tr,train_y_tr,train_x_val,train_y_val = batch
#     conv_kern_all,conv_bias_all = weights_output
#     umaskcoords_trtr = dinf_tr['umaskcoords_trtr']
#     umaskcoords_trval = dinf_tr['umaskcoords_trval']
#     N_trtr = dinf_tr['N_trtr']
#     N_trval = dinf_tr['N_trval']
#     mask_trtr = dinf_tr['maskunits_trtr']
#     mask_trval = dinf_tr['maskunits_trval']

#     MAX_RGCS = dinf_tr['MAX_RGCS']
#     cell_types_unique = dinf_tr['cell_types_unique']
#     segment_size = dinf_tr['segment_size']


#     if NUM_SPLITS==0:       # That is don't split the vmap batches. Run all retinas in parallel
#         local_losses,local_y_preds,local_mdl_states,local_grads_all,local_kerns,local_biases = jax.vmap(Partial(metal_grads,\
#                                                                                                                   mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size))\
#                                                                                                                   (train_x_tr,train_y_tr,train_x_val,train_y_val,
#                                                                                                                    umaskcoords_trtr,umaskcoords_trval,N_trtr,N_trval,mask_trtr,mask_trval,
#                                                                                                                    conv_kern_all,conv_bias_all)
    
                    
#     else:       # Otherwise split the vmap. This avoids running out of GPU memory when we have many retinas and large batch size
#         (local_losses, local_y_preds, local_kerns, local_biases),local_grads_all = batched_metal_grads(metal_grads,
#         mdl_state, global_params,MAX_RGCS,cell_types_unique,segment_size, train_x_tr, train_y_tr, train_x_val, train_y_val,
#         umaskcoords_trtr, umaskcoords_trval, N_trtr, N_trval, mask_trtr, mask_trval, conv_kern_all, conv_bias_all, 
#         NUM_SPLITS=NUM_SPLITS)
                
                                        
#     local_losses_summed = jnp.sum(local_losses)
#     local_grads_summed = jax.tree_map(lambda g: jnp.sum(g,axis=0), local_grads_all)
#     local_grads_summed = clip_grads(local_grads_summed)
    
#     weights_output = (local_kerns,local_biases)
    
#     mdl_state = mdl_state.apply_gradients(grads=local_grads_summed)
    
           
#     # print(local_losses_summed)   
        
    
#     """
#     for key in local_grads_summed.keys():
#         try:
#             print('%s kernel: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['kernel']))))
#         except:
#             print('%s bias: %e\n'%(key,jnp.sum(abs(local_grads_summed[key]['bias']))))
    
#     """

#     return local_losses_summed,mdl_state,weights_output,local_grads_summed





# LOSS FUNC TESTS
# y = .5
# y_pred = np.arange(0.1,1.1,.1)

# poisson_loss = y_pred-y*np.log(y_pred)
# alt_loss = np.abs(y-y_pred)/(0.5*(np.abs(y)+np.abs(y_pred)))
# mape_loss = np.abs((y-y_pred)/(y+1e-6))
# mad_loss = np.abs(y-y_pred)
# reg = 2e-1*y_pred

# lamb = 1
# comb_loss = lamb*poisson_loss + (1-lamb)*mad_loss+reg

# plt.plot(y_pred,poisson_loss)
# plt.plot(y_pred,mad_loss)
# plt.plot(y_pred,comb_loss)

