
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
MAX_RGCS = 800

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
    
    
    # unique_ids, new_ids = jnp.unique(u_id, return_inverse=True,size=N_units+1,fill_value=-1)
    # N_units = jnp.max(u_id) + 1

    
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
    # y_units = jnp.where(y_units == 0, 1e-6, y_units)
    # mask = jnp.ones(y_units.shape)
    # mask = mask.at[:,N_tr:].set(0)
    loss = y_pred_units-y_units*jax.lax.log(y_pred_units)
    loss = (loss*mask_tr[None,:])/N_tr
    loss=jnp.nansum(loss)
    
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
    # if jnp.ndim(y)==2:   # That is the format is units
    #     rgb = jnp.zeros((X.shape[0],X.shape[2],X.shape[3],2),dtype=X.dtype)
    
        
    y_pred,state = state.apply_fn({'params': params},X,training=True,mutable=['intermediates'])
    
    # intermediates = state['intermediates']
    # dense_activations = intermediates['dense_activations'][0]
    
    loss,y_units,y_pred_units = calc_loss(y_pred,y,coords,segment_size,N_tr,mask_tr)

    loss = loss + weight_regularizer(params,alpha=1e-4)
    # loss = loss + activity_regularizer(dense_activations,alpha=1e-4)

    return loss,y_pred

@jax.jit
def train_step_metal(mdl_state,batch,weights_output,lr,dinf_tr):        # Make unit vectors then scale by num of RGCs
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
    def metal_grads(mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size,train_x_tr,train_y_tr,train_x_val,train_y_val,coords_tr,coords_val,N_tr,N_val,mask_tr,mask_val,conv_kern,conv_bias):

        # Split the batch into inner and outer training sets
        # PARAMETERIZE this
        # frac_s_train = 0.5
        # len_data = train_x.shape[0]
        # len_s_train = int(len_data*frac_s_train)
        
        # batch_train = (train_x[:len_s_train],train_y_tr[:len_s_train])
        # batch_val = (train_x[len_s_train:],train_y_val[len_s_train:])
        
        batch_train = (train_x_tr,train_y_tr)
        batch_val = (train_x_val,train_y_val)

        # N_points_val = N_task_val*segment_size

        # Make local model by using global params but local dense layer weights
        local_params = global_params
        local_params['output']['kernel'] = conv_kern
        local_params['output']['bias'] = conv_bias
        local_mdl_state = mdl_state.replace(params=local_params)

        # Calculate gradients of the local model wrt to local params    
        grad_fn = jax.value_and_grad(task_loss,argnums=1,has_aux=True)
        (local_loss_train,y_pred_train),local_grads = grad_fn(local_mdl_state,local_params,batch_train,coords_tr,N_tr,segment_size,mask_tr)
        
        # scale the local gradients according to ADAM's first step. Helps to stabilize
        # And update the parameters
        local_params = jax.tree_map(lambda p, g: p - lr*(g/(jnp.abs(g)+1e-8)), local_params, local_grads)

        # Calculate gradients of the loss of the resulting local model but using the validation set
        # local_mdl_state = mdl_state.replace(params=local_params)
        (local_loss_val,y_pred_val),local_grads_val = grad_fn(local_mdl_state,local_params,batch_val,coords_val,N_val,segment_size,mask_val)
        
        # Update only the Dense layer weights since we retain it
        local_params_val = jax.tree_map(lambda p, g: p - lr*(g/(jnp.abs(g)+1e-8)), local_params, local_grads_val)
        
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

    # maxRGCs = mask_unitsToTake_all.shape[-1] #jnp.sum(mask_unitsToTake_all)
    local_losses,local_y_preds,local_mdl_states,local_grads_all,local_kerns,local_biases = jax.vmap(Partial(metal_grads,\
                                                                                                              mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size))\
                                                                                                              (train_x_tr,train_y_tr,train_x_val,train_y_val,
                                                                                                               umaskcoords_trtr,umaskcoords_trval,N_trtr,N_trval,mask_trtr,mask_trval,
                                                                                                               conv_kern_all,conv_bias_all)
                  
    local_losses_summed = jnp.sum(local_losses)
    local_grads_summed = jax.tree_map(lambda g: jnp.sum(g,axis=0), local_grads_all)
    
    
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
def train_step_maml(mdl_state,batch,weights_output,lr,dinf_tr):        # Make unit vectors then scale by num of RGCs
    """
    State is the grand model state that actually gets updated
    state_task is the "state" after gradients are applied for a specific task
        task_idx = 1
        conv_kern = conv_kern_all[task_idx]
        conv_bias = conv_bias_all[task_idx]
        train_x = train_x[task_idx]
        train_y_tr = train_y_tr[task_idx]
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
    def maml_grads(mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size,train_x_tr,train_y_tr,train_x_val,train_y_val,coords_tr,coords_val,N_tr,N_val,mask_tr,mask_val,conv_kern,conv_bias):

        # Split the batch into inner and outer training sets
        # PARAMETERIZE this
        # frac_s_train = 0.5
        # len_data = train_x.shape[0]
        # len_s_train = int(len_data*frac_s_train)
        
        batch_train = (train_x_tr,train_y_tr)
        batch_val = (train_x_val,train_y_val)
        # N_points_val = N_task_val*segment_size

        # Make local model by using global params but local dense layer weights
        local_params = global_params
        # local_params['output']['kernel'] = conv_kern
        # local_params['output']['bias'] = conv_bias
        local_mdl_state = mdl_state.replace(params=local_params)

        # Calculate gradients of the local model wrt to local params    
        grad_fn = jax.value_and_grad(task_loss,argnums=1,has_aux=True)
        (local_loss_train,y_pred_train),local_grads = grad_fn(local_mdl_state,local_params,batch_train,coords_tr,N_tr,segment_size,mask_tr)
        
        # scale the local gradients according to ADAM's first step. Helps to stabilize
        # And update the parameters
        local_params = jax.tree_map(lambda p, g: p - lr*(g/(jnp.abs(g)+1e-8)), local_params, local_grads)

        # Calculate gradients of the loss of the resulting local model but using the validation set
        # local_mdl_state = mdl_state.replace(params=local_params)
        (local_loss_val,y_pred_val),local_grads_val = grad_fn(local_mdl_state,local_params,batch_val,coords_val,N_val,segment_size,mask_val)
        
        # Update only the Dense layer weights since we retain it
        local_params_val = jax.tree_map(lambda p, g: p - lr*(g/(jnp.abs(g)+1e-8)), local_params, local_grads_val)
        
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

    # maxRGCs = mask_unitsToTake_all.shape[-1] #jnp.sum(mask_unitsToTake_all)
    local_losses,local_y_preds,local_mdl_states,local_grads_all,local_kerns,local_biases = jax.vmap(Partial(maml_grads,\
                                                                                                              mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size))\
                                                                                                              (train_x_tr,train_y_tr,train_x_val,train_y_val,
                                                                                                               umaskcoords_trtr,umaskcoords_trval,N_trtr,N_trval,mask_trtr,mask_trval,
                                                                                                               conv_kern_all,conv_bias_all)
                  
    local_losses_summed = jnp.sum(local_losses)
    local_grads_summed = jax.tree_map(lambda g: jnp.sum(g,axis=0), local_grads_all)
    
    
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
    idx_task = idx_master
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
        y_shape = batch[1].shape[1:]
        y_pred_units = jnp.empty((0,N_val))
        y_units = jnp.empty((0,N_val))
        y_pred = jnp.empty((0,*y_shape))
        y = jnp.empty((0,*y_shape))

        loss = []
        count_batch = 0
        for batch in batch_val:
            if count_batch<n_batches:
                X_batch,y_batch = batch
                y_pred_batch = state.apply_fn({'params': state.params},X_batch,training=True)
                loss_batch,y_units_b,y_pred_units_b = calc_loss(y_pred_batch,y_batch,coords,segment_size,N_val,mask_val)
                y_units_b = y_units_b[:,:N_val]
                y_pred_units_b = y_pred_units_b[:,:N_val]
                
                loss.append(loss_batch)
                y_pred = jnp.concatenate((y_pred,y_pred_batch),axis=0)
                y = jnp.concatenate((y,y_batch),axis=0)
                y_pred_units = jnp.concatenate((y_pred_units,y_pred_units_b),axis=0)
                y_units = jnp.concatenate((y_units,y_units_b),axis=0)

                count_batch+=1
            else:
                break
    return loss,y_pred,y,y_pred_units,y_units



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

def save_epoch(state,config,weights_output,fname_cp,weights_all=None):
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
    fname_weights_output = os.path.join(fname_cp,'weights_output.h5')
    with h5py.File(fname_weights_output,'w') as f:
        f.create_dataset('weights_output_kernel',data=np.array(weights_output[0],dtype='float32'),compression='gzip')
        f.create_dataset('weights_output_bias',data=np.array(weights_output[1],dtype='float32'),compression='gzip')

    if weights_all!=None:
        fname_weights_all = os.path.join(fname_cp,'weights_all.h5')
        with h5py.File(fname_weights_all,'w') as f:
            save_nested_dict_to_h5(f,weights_all)
    


    
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

# %% Training func

def train(mdl_state,weights_output,config,dataloader_train,dataloader_val,dinf_tr,dinf_val,nb_epochs,path_model_save,save=False,lr_schedule=None,step_start=0,APPROACH='metal'):
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

    idx_master = 2#7
    dinf_batch_val = jax.tree_map(lambda x: x[idx_master] if isinstance(x, np.ndarray) else x, dinf_val)
    dinf_batch_valtr = dict(N_val=dinf_tr['N_trtr'][idx_master],
                            maskunits_val=dinf_tr['maskunits_trtr'][idx_master],
                            umaskcoords_val=dinf_tr['umaskcoords_trtr'][idx_master],
                            segment_size=dinf_tr['segment_size'])


    
    epoch=0
    for epoch in tqdm(range(step_start,nb_epochs)):
        _ = gc.collect()
        loss_batch_train=[]
        t = time.time()
        # batch_train = next(iter(dataloader_train)); batch=batch_train; 
        for batch_train in dataloader_train:
            current_lr = lr_schedule(mdl_state.step)               
            
            if APPROACH == 'metal':
                loss,mdl_state,weights_output,grads = train_step_metal(mdl_state,batch_train,weights_output,current_lr,dinf_tr)
            elif APPROACH == 'maml':
                loss,mdl_state,weights_output,grads = train_step_maml(mdl_state,batch_train,weights_output,current_lr,dinf_tr)
            else:
                print('Invalid APPROACH')
                break
            elap = time.time()-t
            print(elap)

            # print(loss)
            loss_batch_train.append(loss)
            # _,y,_,_ = batch_train
            # print(mdl_state.step)
            # print(mdl_state.opt_state[0][0])
            

        assert jnp.sum(grads['Conv_0']['kernel']) != 0, 'Gradients are Zero'
            
        
        # For validation, update the new state with weights from the idx_master task
        mdl_state_val = mdl_state
        if APPROACH != 'maml':
            mdl_state_val.params['output']['kernel'] = weights_output[0][idx_master]
            mdl_state_val.params['output']['bias'] = weights_output[1][idx_master]
            
    
        loss_batch_val = []
        batch = next(iter(dataloader_val))
        y_val_units = np.zeros((0,dinf_batch_val['N_val']))
        y_pred_val_units = np.zeros((0,dinf_batch_val['N_val']))
        for batch in dataloader_val:
            batch_val = (batch[0][idx_master],batch[1][idx_master])
            loss,y_pred,y,y_pred_units,y_units = eval_step(mdl_state_val,batch_val,dinf_batch_val)
            y_val_units = np.concatenate((y_val_units,y_units),axis=0)
            y_pred_val_units = np.concatenate((y_pred_val_units,y_pred_units),axis=0)
            loss_batch_val.append(loss)

        loss_batch_train_test,y_pred_train,y_train,y_pred_train_units,y_train_units = eval_step(mdl_state_val,(batch_train[0][idx_master],batch_train[1][idx_master]),dinf_batch_valtr)
        
        loss_currEpoch_master = np.mean(loss_batch_train)
        loss_currEpoch_train = np.mean(loss_batch_train_test)
        loss_currEpoch_val = np.mean(loss_batch_val)
    
        loss_epoch_train.append(np.mean(loss_currEpoch_train))
        loss_epoch_val.append(np.mean(loss_currEpoch_val))
        
        current_lr = lr_schedule(mdl_state.step)
        
        temporal_width_eval = batch_val[0][idx_master].shape[0]
        fev_val,_,predCorr_val,_ = model_evaluate_new(y_units,y_pred_units,temporal_width_eval,lag=0,obs_noise=0)
        fev_val_med,predCorr_val_med = np.median(fev_val),np.median(predCorr_val)
        fev_train,_,predCorr_train,_ = model_evaluate_new(y_train_units,y_pred_train_units,temporal_width_eval,lag=0,obs_noise=0)
        fev_train_med,predCorr_train_med = np.median(fev_train),np.median(predCorr_train)
        
        fev_epoch_train.append(fev_train_med)
        fev_epoch_val.append(fev_val_med)

        print('Epoch: %d, global_loss: %.2f || local_train_loss: %.2f, fev: %.2f, corr: %.2f || local_val_loss: %.2f, fev: %.2f, corr: %.2f || lr: %.2e'\
              %(epoch+1,loss_currEpoch_master,loss_currEpoch_train,fev_train_med,predCorr_train_med,loss_currEpoch_val,fev_val_med,predCorr_val_med,current_lr))
        
            
        fig,axs = plt.subplots(2,1,figsize=(20,10));axs=np.ravel(axs);fig.suptitle('Epoch: %d'%(epoch+1))
        axs[0].plot(y_train_units[:200,2]);axs[0].plot(y_pred_train_units[:200,10]);axs[0].set_title('Train')
        axs[1].plot(y_units[:,10]);axs[1].plot(y_pred_units[:,10]);axs[1].set_title('Validation')
        plt.show()
        plt.close()


        if save == True:
            fname_cp = os.path.join(path_model_save,'epoch-%03d'%epoch)
            save_epoch(mdl_state,config,weights_output,fname_cp)
            
    return loss_currEpoch_master,loss_epoch_train,loss_epoch_val,mdl_state,weights_output,fev_epoch_train,fev_epoch_val


# %% Finetuning


@jax.jit
def create_mask(params,layers_finetune):
    def unfreeze_layer(layer_name_unfreeze, sub_dict):
        return {k: jax.tree_map(lambda _: True, v) if layer_name_unfreeze in k else v for k, v in sub_dict.items()}
    
    mask = jax.tree_util.tree_map(lambda _: False, params)      # Freeze all layers
    
    layers_finetune_exact = models_jax.get_exactLayers(mask,layers_finetune)
    for layer in layers_finetune_exact:
        mask = unfreeze_layer(layer, mask)  # unfreeze selected layer
        
    return mask

@jax.jit
def create_learning_rate_scheduler(lr_schedule):
    print(lr_schedule)

    if lr_schedule['name'] == 'exponential_decay':
        learning_rate_fn = optax.exponential_decay(
                            init_value=lr_schedule['lr_init'], 
                            transition_steps=lr_schedule['transition_steps'], 
                            decay_rate=lr_schedule['decay_rate'], 
                            staircase=lr_schedule['staircase'],
                            transition_begin=lr_schedule['transition_begin'],
                            )
    else:
        learning_rate_fn = optax.constant_schedule(value=lr_schedule['lr_init'])
        
    return learning_rate_fn

    
@jax.jit
def update_optimizer(ft_mdl_state,mask=None,lr_schedule=None):
    params = ft_mdl_state.params
    
    scheduler_fn = create_learning_rate_scheduler(lr_schedule)
    optimizer = optax.adam(learning_rate=scheduler_fn)
    if mask is not None:
        optimizer = optax.chain(optax.masked(optax.adam(learning_rate=scheduler_fn),mask))
        
    ft_optim_state = optimizer.init(params=params)
    ft_mdl_state = ft_mdl_state.replace(tx=optimizer,opt_state=ft_optim_state)

        
    return ft_mdl_state


@jax.jit
def ft_loss_fn(state,trainable_params,fixed_params,batch):
    X,y = batch
    y_pred,state = state.apply_fn({'params': {**fixed_params, **trainable_params}},X,training=True,mutable=['intermediates'])
    intermediates = state['intermediates']
    dense_activations = intermediates['dense_activations'][0]
    loss = (y_pred - y*jax.lax.log(y_pred)).mean()

    # if training==True:
    loss = loss + weight_regularizer(trainable_params,alpha=1e-5)
    loss = loss + activity_regularizer(dense_activations,alpha=1e-5)

    return loss,y_pred

@jax.jit
def ft_train_step(state,fixed_params,batch):
    grad_fn = jax.value_and_grad(ft_loss_fn,argnums=1,has_aux=True)
    (loss,y_pred),grads = grad_fn(state,state.params,fixed_params,batch)
    grads = clip_grads(grads)
    state = state.apply_gradients(grads=grads)
    
    return state,loss

def ft_eval_step(state,fixed_params,data,n_batches=1e5):
    if type(data) is tuple:
        X,y = data
        y_pred = state.apply_fn({'params': {**fixed_params,**state.params}},X,training=True)
        loss = (y_pred - y*jax.lax.log(y_pred)).mean()
        return loss,y_pred,y
    
    else:       # if the data is in dataloader format
        # rgb = {**fixed_params,**state.params}
        _,y_batch = next(iter(data))
        n_units = y_batch.shape[-1]
        y_pred = jnp.empty((0,n_units))
        y = jnp.empty((0,n_units))
        loss = []
        count_batch = 0
        for batch in data:
            if count_batch<n_batches:
                X_batch,y_batch = batch
                y_pred_batch = state.apply_fn({'params': {**fixed_params,**state.params}},X_batch,training=True)
                loss_batch = (y_pred_batch - y_batch*jax.lax.log(y_pred_batch)).mean()
                # loss_batch,(y_pred_batch,batch_updates) = forward_pass(state,state.params,state.batch_stats,batch,training=False)
                loss.append(loss_batch)
                y_pred = jnp.concatenate((y_pred,y_pred_batch),axis=0)
                y = jnp.concatenate((y,y_batch),axis=0)
                count_batch+=1
            else:
                break
    return loss,y_pred,y

@jax.jit
def ft_eval_step_jit(state,fixed_params,batch_val):
    X,y = batch_val
    y_pred = state.apply_fn({'params': {**fixed_params,**state.params}},X,training=True)
    loss = (y_pred - y*jax.lax.log(y_pred)).mean()
    
    return loss,y_pred,y




def ft_train(ft_mdl_state,ft_params_fixed,config,ft_data_train,ft_data_val,ft_data_test,obs_noise,batch_size,ft_nb_epochs,path_model_save,save=True,ft_lr_schedule=None,epoch_start=0):
    # loss_fn = optax.losses.kl_divergence
    n_batches = np.ceil(len(ft_data_train.X)/batch_size)

    RetinaDataset_train = RetinaDataset(ft_data_train.X,ft_data_train.y,transform=None)
    dataloader_train = DataLoader(RetinaDataset_train,batch_size=batch_size,collate_fn=jnp_collate,shuffle=False)
    

    RetinaDataset_val = RetinaDataset(ft_data_val.X,ft_data_val.y,transform=None)
    dataloader_val = DataLoader(RetinaDataset_val,batch_size=batch_size,collate_fn=jnp_collate)
    
    RetinaDataset_test = RetinaDataset(ft_data_test.X,ft_data_test.y,transform=None)
    dataloader_test = DataLoader(RetinaDataset_test,batch_size=batch_size,collate_fn=jnp_collate)

    # learning_rate_fn = create_learning_rate_scheduler(ft_lr_schedule)
    learning_rate_fn = ft_lr_schedule


    loss_epoch_train = []
    loss_epoch_val = []

    loss_batch_train = []
    loss_batch_val = []
    
    fev_epoch_train = []
    fev_epoch_val = []
    fev_epoch_test = []
    
    corr_epoch_train = []
    corr_epoch_val = []
    corr_epoch_test = []
    
    lr_epoch = []
    lr_step = []

    epoch=0
    # epoch_start=0
    step=0
    # batch_train = next(iter(dataloader_train))
    for epoch in tqdm(range(epoch_start,ft_nb_epochs)):
        # t = time.time()
        loss_batch_train=[]
        for batch_train in dataloader_train:
            step = step+1
            ft_mdl_state, loss = ft_train_step(ft_mdl_state,ft_params_fixed,batch_train)
            loss_batch_train.append(loss)
            lr_step.append(learning_rate_fn(ft_mdl_state.step))
        # elap = time.time()-t
        # print(elap)

            # print(loss)

        loss_batch_val,y_pred,y = ft_eval_step(ft_mdl_state,ft_params_fixed,dataloader_val)
        loss_batch_test,y_pred_test,y_test = ft_eval_step(ft_mdl_state,ft_params_fixed,dataloader_test)
        loss_batch_train_test,y_pred_train_test,y_train_test = ft_eval_step(ft_mdl_state,ft_params_fixed,batch_train)
        
        loss_currEpoch_train = np.nanmean(loss_batch_train)
        loss_currEpoch_val = np.nanmean(loss_batch_val)

        loss_epoch_train.append(np.mean(loss_currEpoch_train))
        loss_epoch_val.append(np.mean(loss_currEpoch_val))
        
        # print(ft_mdl_state.step)
        # print(step)
        current_lr = learning_rate_fn(ft_mdl_state.step)
        lr_epoch.append(current_lr)
        
        temporal_width_eval = ft_data_train.X[0].shape[0]
        fev_val,_,corr_val,_ = model_evaluate_new(y,y_pred,temporal_width_eval,lag=0,obs_noise=obs_noise)
        fev_val_med,corr_val_med = np.median(fev_val),np.median(corr_val)
        fev_test,_,corr_test,_ = model_evaluate_new(y_test,y_pred_test,temporal_width_eval,lag=0,obs_noise=obs_noise)
        fev_test_med,corr_test_med = np.median(fev_test),np.median(corr_test)
        fev_train,_,corr_train,_ = model_evaluate_new(y_train_test,y_pred_train_test,temporal_width_eval,lag=0,obs_noise=obs_noise)
        fev_train_med,corr_train_med = np.median(fev_train),np.median(corr_train)

        fev_epoch_train.append(fev_train_med)
        # fev_epoch_val.append(fev_val_med)
        # fev_epoch_test.append(fev_test_med)
        fev_epoch_val.append(fev_val)
        fev_epoch_test.append(fev_test)

        
        corr_epoch_train.append(corr_train_med)
        # corr_epoch_val.append(corr_val_med)
        # corr_epoch_test.append(corr_test_med)
        corr_epoch_val.append(corr_val)
        corr_epoch_test.append(corr_test)


        print('Epoch: %d, train_loss: %.2f, fev: %.2f, corr: %.2f || val_loss: %.2f, fev: %.2f, corr: %.2f || lr: %.2e'\
              %(epoch+1,loss_currEpoch_train,fev_train_med,corr_train_med,loss_currEpoch_val,fev_val_med,corr_val_med,current_lr))

        # fig,axs = plt.subplots(2,1,figsize=(20,10));axs=np.ravel(axs);fig.suptitle('Finetuning | Epoch: %d'%(epoch+1))
        # axs[0].plot(y_train_test[:200,10]);axs[0].plot(y_pred_train_test[:200,10]);axs[0].set_title('Train')
        # axs[1].plot(y[:,10]);axs[1].plot(y_pred[:,10]);axs[1].set_title('Validation')
        # plt.show()
        # plt.close()
        
        weights_all = {**ft_params_fixed,**ft_mdl_state.params}
        weights_output = (weights_all['Dense_0']['kernel'],weights_all['Dense_0']['bias'])
        if save==True:
            fname_cp = os.path.join(path_model_save,'epoch-%03d'%epoch)
            save_epoch(ft_mdl_state,config,weights_output,fname_cp,weights_all)
            
        perf = (fev_epoch_train,corr_epoch_train,fev_epoch_val,corr_epoch_val,fev_epoch_test,corr_epoch_test)

    return loss_epoch_train,loss_epoch_val,ft_mdl_state,perf,lr_epoch,lr_step


# %% Recycle
# @jax.jit
# def train_step_maml(mdl_state,batch,weights_output,lr,dinf_tr):        # Make unit vectors then scale by num of RGCs
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
#     def maml_grads(mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size,train_x,train_y_tr,train_y_val,coords_tr,coords_val,N_tr,N_val,mask_tr,mask_val):

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
#     def maml_loss(mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size,train_x,train_y_tr,train_y_val,umaskcoords_trtr,umaskcoords_trval,N_trtr,N_trval,mask_trtr,mask_trval):
        
#         local_losses,local_y_preds,local_mdl_states = jax.vmap(Partial(maml_grads,mdl_state,global_params,MAX_RGCS,cell_types_unique,segment_size))\
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
    
#     grad_fn = jax.value_and_grad(maml_loss,argnums=1,has_aux=True)
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



