#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:04:06 2025

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

import numpy as np
import jax
import re
from collections import namedtuple
from model import utils_si
from model.data_handler import isintuple
Exptdata = namedtuple('Exptdata', ['X', 'y'])
Exptdata_spikes = namedtuple('Exptdata', ['X', 'y','spikes'])

def change_dtype(data,dtype='float32'):
    X=data.X
    X = [arr.astype(dtype) for arr in X]
    
    y=data.y
    y=[arr.astype(dtype) for arr in y]
    return Exptdata(X,y)

def get_unit_types(unames):
    pattern = re.compile(r'type(\d+)')
    cell_types = np.array([pattern.match(entry).group(1) for entry in unames]).astype('int')
    cell_types_unique = np.unique(cell_types)
    
    return cell_types

def remove_boundary_units(dinf,data_train,data_val):
    unit_locs =  dinf['unit_locs']
    pixel_neigh = 1
    frame_shape = data_train.X.shape[-2:]
    idx_keep = []
    for u in range(len(unit_locs)):
        if np.all(unit_locs[u]>0) and np.all(unit_locs[u]-pixel_neigh)>0 and np.all((unit_locs[u]+pixel_neigh+1)<[frame_shape[1],frame_shape[0]]):
            idx_keep.append(u)
    
    if len(idx_keep)<len(unit_locs):
        n_discard = len(unit_locs)-len(idx_keep)
        print('Discarded %d boundary units'%n_discard)
        
        dinf['unit_locs'] = dinf['unit_locs'][idx_keep]
        dinf['unit_types'] = dinf['unit_types'][idx_keep]
        dinf['unames'] = dinf['unames'][idx_keep]
        
        data_train = Exptdata_spikes(data_train.X,data_train.y[:,idx_keep],data_train.spikes)
        data_val = Exptdata_spikes(data_val.X,data_val.y[:,idx_keep],data_val.spikes)

    return dinf,data_train,data_val

    
def bytestostring(arr):
    rgb = np.array([a.decode('utf-8') for a in arr])
    return rgb

def normalize_responses(data,norm_val):
    y = data.y
    norm_val = np.maximum(norm_val, 1e-6)
    y = y/norm_val[None,:]
    
    return Exptdata_spikes(data.X,y,data.spikes)

def arrange_data_formaps(exp,data_train,data_val,parameters,frac_train_units,psf_params,info_unitSplit=None,BUILD_MAPS=False,MODE='training'):
    """
    mode can be either training in whic case we split units across training and validation sets. In validation mode we keep all units for validation
    """
    dinf = {}
    dinf['unit_locs'] = parameters['unit_locs']
    dinf['unit_types'] = parameters['unit_types']
    dinf['unames'] = bytestostring(parameters['unames'])
    
    # dinf,data_train,data_val = remove_boundary_units(dinf,data_train,data_val)  # This step is already done in dataset creation stage
    
    rgb = np.concatenate((data_train.y,data_val.y),axis=0)
    max_resp = np.max(rgb,axis=0)
    data_train = normalize_responses(data_train,max_resp)
    data_val = normalize_responses(data_val,max_resp)

    
    if (info_unitSplit==None and frac_train_units==1) or MODE=='validation':
        # data_train,_,_ = buildRespMap(data_train.X,data_train.y,data_train.spikes,parameters['unit_locs'],parameters['unit_types'])
        # data_val,_,_ = buildRespMap(data_val.X,data_val.y,data_val.spikes,parameters['unit_locs'],parameters['unit_types'])

        idx_units_train = np.arange(len(dinf['unames']))
        idx_units_val = idx_units_train

    
    else:
        if info_unitSplit==None:
            print('Generating unit split info')
            info_unitSplit = units_validation_split(dinf['unames'],dinf['unit_locs'],frac_train_units)
        else:
            print('Using unit split info from dataset file')

        
        data_train,data_val = maps_validation_split(data_train,data_val,info_unitSplit['idx_train'],
                                                                   info_unitSplit['idx_val'],parameters['unit_locs'],parameters['unit_types'],BUILD_MAPS=BUILD_MAPS)
        
        idx_units_train = info_unitSplit['idx_train']
        idx_units_val = info_unitSplit['idx_val']
        
        print('N_RGCS training: %d\nN_RGCs testing: %d'%(len(idx_units_train),len(idx_units_val)))
        

    dinf['idx_units_train'] = idx_units_train
    dinf['N_units_train'] = len(idx_units_train)

    dinf['idx_units_val'] = idx_units_val
    dinf['N_units_val'] = len(idx_units_val)


    data_train,dinf['umasks_train'],dinf['umaskcoords_train'],dinf['maskunitloc_train'],dinf['segment_size'] = unit_psf(data_train,
                                                                                                                dinf['unit_locs'][idx_units_train],
                                                                                                                dinf['unit_types'][idx_units_train],
                                                                                                                psf_params,BUILD_MAPS=BUILD_MAPS)
  
    data_val,dinf['umasks_val'],dinf['umaskcoords_val'],dinf['maskunitloc_val'],_ = unit_psf(data_val,
                                                                                    dinf['unit_locs'][idx_units_val],
                                                                                    dinf['unit_types'][idx_units_val],
                                                                                    psf_params,BUILD_MAPS=BUILD_MAPS)
    
    
    if MODE=='training':     # Only do this thing if the mode is not validation
        intersection = check_psf_overlap(dinf['umaskcoords_train'],dinf['umaskcoords_val'])
        for t in range(len(intersection)):
            print('%s: Number of overlapping pixels in Cell Type %d: %d'%(exp,t+1,len(intersection[t])))
    
        all_empty = np.all([arr.size == 0 for arr in intersection])
        if all_empty==False:
            data_train,dinf['umasks_train'],dinf['umaskcoords_train'] = remove_overlaps_across(data_train,dinf['umaskcoords_train'],dinf['umasks_train'],
                                                                                                          dinf['unit_types'][idx_units_train],intersection,BUILD_MAPS=BUILD_MAPS)
    assert len(dinf['umaskcoords_train'])/dinf['segment_size'] ==  dinf['N_units_train']
    assert len(dinf['umaskcoords_val'])/dinf['segment_size'] ==  dinf['N_units_val']

    # intersection2 = check_psf_overlap(dinf['umaskcoords_train'],dinf['umaskcoords_val'])
    # all_empty = np.all([arr.size == 0 for arr in intersection2])
    # assert all_empty==True,'%s Train/Val datasets still have overlap'
    
    return data_train,data_val,dinf




def units_validation_split(unames,unit_locs,frac_train_units=0.9,min_dist=5):
    """
    unames = data_quality['uname_selectedUnits']
    u_idx = data_quality['idx_unitsToTake']    
    frac_train_units=0.9
    """
    pattern = re.compile(r'type(\d+)')
    cell_types = np.array([pattern.match(entry).group(1) for entry in unames]).astype('int')
    cell_types_unique = np.unique(cell_types)
    
    n_perType = np.zeros((len(cell_types_unique)))
    for t in range(len(cell_types_unique)):
        n_perType[t] = (cell_types==cell_types_unique[t]).sum()

    nTrain_perType = (frac_train_units*n_perType).astype('int')
    nVal_perType = (n_perType-nTrain_perType).astype('int')
    
    idx_train_perType = np.zeros((len(cell_types_unique)),dtype='object')
    idx_val_perType = np.zeros((len(cell_types_unique)),dtype='object')
    idx_train = np.zeros((0),dtype='int')
    idx_val = np.zeros((0),dtype='int')
    
    unames_train_perType = np.zeros((len(cell_types_unique)),dtype='object')
    unames_val_perType = np.zeros((len(cell_types_unique)),dtype='object')
    unames_train = np.zeros((0),dtype='object')
    unames_val = np.zeros((0),dtype='object')


    for t in range(len(cell_types_unique)):
        idx_type = np.where(cell_types==cell_types_unique[t])[0]
        
        
        center_of_mass = unit_locs[idx_type].mean(axis=0)
        distances = np.linalg.norm(unit_locs[idx_type] - center_of_mass, axis=1)
        
        # Sort by distance to center
        sorted_indices = np.argsort(distances)
        
        # Select points ensuring at least 5 pixels apart
        selected_indices = []
        for idx in sorted_indices:
            if len(selected_indices) == nVal_perType[t]:
                break  # Stop when we've selected enough points
            
            # Check distance from previously selected points
            if all(np.linalg.norm(unit_locs[idx_type][idx] - unit_locs[idx_type][sel_idx]) >= min_dist for sel_idx in selected_indices):
                selected_indices.append(idx)
        
        selected_indices = np.array(selected_indices)

        idx_val_perType[t] = idx_type[selected_indices]
        idx_train_perType[t] = np.setdiff1d(idx_type,idx_val_perType[t])

        
        idx_train = np.concatenate((idx_train,idx_train_perType[t]),axis=0)
        idx_val = np.concatenate((idx_val,idx_val_perType[t]),axis=0)
        
        
        unames_train_perType[t] = unames[idx_train_perType[t]]
        unames_val_perType[t] = unames[idx_val_perType[t]]
        
    unames_train = unames[idx_train]
    unames_val = unames[idx_val]

          
    
    dict_units_split = dict(
        cell_types=cell_types,
        cell_types_unique=cell_types_unique,
        idx_train_perType=idx_train_perType,
        idx_val_perType=idx_val_perType,
        idx_train=idx_train,
        idx_val=idx_val,
        
        unames_train_perType=unames_train_perType,
        unames_val_perType=unames_val_perType,
        unames_train=unames_train,
        unames_val=unames_val
    )
    
    return dict_units_split



def mapToUnits(fr_map,unit_locs,unit_types):
    samp_idx = np.arange(fr_map.shape[0])[:, None]  # Shape: (N, 1)
    spatial_y = unit_locs[:, 1]  # Shape: (M,)
    spatial_x = unit_locs[:, 0]  # Shape: (M,)
    type_idx = unit_types - 1  # Shape: (M,)
    samp_idx = np.broadcast_to(samp_idx, (fr_map.shape[0], len(unit_types)))
    y_units = fr_map[samp_idx, spatial_y, spatial_x, type_idx]  # Shape: (N, M)
    return y_units


def buildRespMap(X,y,spikes,rf_centers,unit_types):
    
    """
    y = y_units_val
    rf_centers = unit_locs_val
    unit_types=unit_types_val
    """
    
    unit_types_unique = np.unique(unit_types)
    
    fr_map = np.zeros((X.shape[0],X.shape[1],X.shape[2],len(unit_types_unique)),dtype='float32')
    
    u=0
    rf_center_int = []
    for u in range(len(unit_types)): 
        cell_type_idx = unit_types[u]-1 # -1 because array indexing starts from 0
        rf_center_cell = np.floor(rf_centers[u]).astype('int')
        if rf_center_cell[1]>X.shape[1]-1:
            rf_center_cell[1] = X.shape[1]-1
            
        if rf_center_cell[0]>X.shape[2]-1:
            rf_center_cell[0] = X.shape[2]-1

        fr_map[:,rf_center_cell[1],rf_center_cell[0],cell_type_idx] = y[:,u]
        rf_center_int.append(rf_center_cell)
        # fr_map[:,rf_masks[u],cell_type_idx] = y[:,u,None]


    data = Exptdata_spikes(X,fr_map,spikes)
    return data,rf_center_int,unit_types



def maps_validation_split(data_train,data_val,idx_train,idx_val,unit_locs,unit_types,nsamps_val=5000,BUILD_MAPS=True):
    
    """
    data=data_train
    
    # unit_masks = parameters['unit_masks']
    unit_locs = parameters['unit_locs']
    unit_types = parameters['unit_types']
    idx_train = info_unitSplit['idx_train']
    idx_val = info_unitSplit['idx_val']   
    """
    
    X_tr=data_train.X
    y_tr=data_train.y
    spikes_tr=data_train.spikes
    
    y_units_tr = y_tr# mapToUnits(y_tr,unit_locs,unit_types)
    
    y_units_train = y_units_tr[:,idx_train]
    unit_types_train = unit_types[idx_train]
    unit_locs_train = unit_locs[idx_train]
    spikes_train = spikes_tr[:,idx_train]
    # unit_masks_train = unit_masks[idx_train]

    X_val=data_val.X
    y_val=data_val.y
    spikes_val=data_val.spikes
    y_units_val = y_val#mapToUnits(y_val,unit_locs,unit_types)

    y_units_val = y_units_val[:,idx_val]
    unit_types_val = unit_types[idx_val]
    unit_locs_val = unit_locs[idx_val]
    spikes_val = spikes_val[:,idx_val]
    # unit_masks_val = unit_masks[idx_val]

    if BUILD_MAPS==True:
        X_y_train,_,_ = buildRespMap(X_tr,y_units_train,spikes_train,unit_locs_train,unit_types_train)
        X_y_val,_,_ = buildRespMap(X_val,y_units_val,spikes_val,unit_locs_val,unit_types_val)
    else:
        X_y_train = Exptdata_spikes(X_tr,y_units_train,spikes_train)
        X_y_val = Exptdata_spikes(X_val,y_units_val,spikes_val)
       
    
    return X_y_train,X_y_val


def unit_psf(data,unit_locs,unit_types,psf_params,BUILD_MAPS=True):
    """
    data = data_train
    unit_locs=dinf['unit_locs'][idx_units_train]
    unit_types = dinf['unit_types'][idx_units_train]
    pixel_neigh=psf_params['pixel_neigh']
    method = psf_params['method']
    """
    
    # X = data.X
    y = data.y
    spikes = data.spikes
    pixel_neigh=psf_params['pixel_neigh']
    method = psf_params['method']

    frame_shape = [data.X.shape[1],data.X.shape[2]]

    
    # Fix for boundary cells
    # unit_locs[unit_locs[:,0]>=X.shape[2],0] = X.shape[2]-1
    # unit_locs[unit_locs[:,1]>=X.shape[1],1] = X.shape[1]-1
    # unit_locs[unit_locs<0]=0
    
    u=0
    test=[]
    unit_masks = []
    unit_masks_coords = np.zeros((0,4),dtype='int')
    for u in range(len(unit_types)):
        u_type_idx = unit_types[u]-1
        loc_mask = np.zeros((frame_shape[0],frame_shape[1]),dtype=bool)

        if method=='square':
            # if np.all(unit_locs[u]-pixel_neigh)>0 and np.all((unit_locs[u]+pixel_neigh+1)<[X.shape[2],X.shape[1]]):
            loc_mask[unit_locs[u,1]-pixel_neigh:unit_locs[u,1]+pixel_neigh+1,unit_locs[u,0]-pixel_neigh:unit_locs[u,0]+pixel_neigh+1] = True
            segment_size = int((pixel_neigh+1+pixel_neigh)*(pixel_neigh+1+pixel_neigh))
            
        elif method=='cross':
            # if np.all(unit_locs[u]-pixel_neigh)>0 and np.all((unit_locs[u]+pixel_neigh+1)<[X.shape[2],X.shape[1]]):
            loc_mask[unit_locs[u,1]-pixel_neigh:unit_locs[u,1]+pixel_neigh+1,unit_locs[u,0]] = True
            loc_mask[unit_locs[u,1],unit_locs[u,0]-pixel_neigh:unit_locs[u,0]+pixel_neigh+1] = True
            segment_size = int((4*pixel_neigh)+1)

        test.append(loc_mask.sum())
        
        if BUILD_MAPS==True:
            rgb = y[:,unit_locs[u,1],unit_locs[u,0],u_type_idx]
            y[:,loc_mask,u_type_idx] = rgb[:,None]
        
        
        a = np.where(loc_mask)
        b = np.array((u*np.ones(len(a[1]),dtype='int'),unit_types[u]*np.ones(len(a[1]),dtype='int'),a[1],a[0])).T
        unit_masks_coords = np.concatenate((unit_masks_coords,b),axis=0)

        
        unit_masks.append(loc_mask)
        
                
    unit_masks = np.asarray(unit_masks)
   
    cell_types_unique = np.unique(unit_types)
    mask_unitloc = get_maskunitloc(unit_locs,unit_types,cell_types_unique,frame_shape)
    
    data = Exptdata_spikes(data.X,y,spikes)
    
    return data,unit_masks,unit_masks_coords,mask_unitloc,segment_size

    
    
def check_psf_overlap(umaskcoords_train,umaskcoords_val):
    
    coords_train = umaskcoords_train[:,2:4]
    type_train = umaskcoords_train[:,1]
    coords_val = umaskcoords_val[:,2:4]
    type_val = umaskcoords_val[:,1]
    
    unique_types = np.unique(type_train)
    
    t=0
    intersection = []
    for t in range(len(unique_types)):
        idx_coords_train_t = type_train==unique_types[t]
        coords_train_t = coords_train[idx_coords_train_t]
        
        idx_coords_val_t = type_val==unique_types[t]
        coords_val_t = coords_val[idx_coords_val_t]

        intersection_t = np.array([point for point in coords_train_t if any(np.all(point == coords_val_t, axis=1))])
        intersection.append(intersection_t)     # Lists represent cell types

    return intersection

"""
def remove_overlaps_across(data,umaskcoords_train,umasks_train,cell_types,intersection):
    # data=data_train
    # umaskcoords_train = dinf['umaskcoords_train']
    # umasks_train = dinf['umasks_train']
    # cell_types = dinf['unit_types'][dinf['idx_units_train']]

    X_tr = data.X
    y_tr = data.y
    spikes_tr = data.spikes
    cell_types_unique = np.unique(umaskcoords_train[:,1])
    umasks_new = umasks_train
    
    t=0
    idx_coords_remove = np.zeros((0),dtype='int')
    for t in range(len(intersection)):
        coords_t = intersection[t]
        if len(coords_t)>0:
            y_tr[:,coords_t[:,1],coords_t[:,0],t] = 0
            a = umaskcoords_train[:,2:4]
            idx_remove = np.nonzero((a[:, None] == coords_t).all(axis=2))[0]
            idx_coords_remove = np.concatenate((idx_coords_remove,idx_remove),axis=0)
            
            idx_cells_t = cell_types==cell_types_unique[t]
            umasks_new[idx_cells_t][:,coords_t[:,1],coords_t[:,0]] = False
        
    idx_coords_keep = np.setdiff1d(np.arange(len(umaskcoords_train)),idx_coords_remove)
    umaskcoords_train_new = umaskcoords_train[idx_coords_keep]
    
    data_train_new = Exptdata_spikes(X_tr,y_tr,spikes_tr)
    
    return data_train_new,umasks_new,umaskcoords_train_new
"""

def remove_overlaps_across(data,umaskcoords_train,umasks_train,cell_types,intersection,BUILD_MAPS=True):
    """
    data=data_train
    umaskcoords_train = dinf['umaskcoords_train']
    umasks_train = dinf['umasks_train']
    cell_types = dinf['unit_types'][dinf['idx_units_train']]
    """
    X_tr = data.X
    y_tr = data.y
    spikes_tr = data.spikes
    cell_types_unique = np.unique(umaskcoords_train[:,1])
    umasks_new = umasks_train
    
    t=0
    idx_coords_remove = np.zeros((0),dtype='int')
    for t in range(len(intersection)):
        coords_t = intersection[t]
        if len(coords_t)>0:
            if BUILD_MAPS==True:
                y_tr[:,coords_t[:,1],coords_t[:,0],t] = 0
            a = umaskcoords_train[:,2:4]
            idx_remove = np.nonzero((a[:, None] == coords_t).all(axis=2))[0]
            idx_coords_remove = np.concatenate((idx_coords_remove,idx_remove),axis=0)
            
            idx_cells_t = cell_types==cell_types_unique[t]
            umasks_new[idx_cells_t][:,coords_t[:,1],coords_t[:,0]] = False
        
    idx_coords_keep = np.setdiff1d(np.arange(len(umaskcoords_train)),idx_coords_remove)
    # umaskcoords_train_new = umaskcoords_train[idx_coords_keep]
    umaskcoords_train_new = umaskcoords_train

    data_train_new = Exptdata_spikes(X_tr,y_tr,spikes_tr)
    
    return data_train_new,umasks_new,umaskcoords_train_new



def mapToUnitsEval(pred_frmap,unit_locs,unit_types,umaskcoords,method='avg'):
    """
    umaskcoords = umaskcoords_val
    """
    spatial_y = umaskcoords[:,3] 
    spatial_x = umaskcoords[:,2] 
    type_idx = umaskcoords[:,1] - 1 
    u_id = umaskcoords[:,0]
    N_units = len(np.unique(u_id))
    
    y_pred_allpixs = pred_frmap[:,spatial_y,spatial_x,type_idx].T
    # y_pred_units = np.nanmean(y_pred_allpixs,axis=-1) 
    y_pred_sumsegs = jax.ops.segment_sum(y_pred_allpixs,u_id,N_units)
    # count_per_cell = jax.ops.segment_sum(jnp.ones(len(type_idx)), u_id, N_units)
    y_pred_units = y_pred_sumsegs# / count_per_cell[:, None]  # Broadcast over batch_size
    
    return y_pred_units.T


def scale_preds(pred_rate):
    
    med = np.nanmedian(pred_rate,axis=0)
    a = (pred_rate-np.nanmin(pred_rate,axis=0)) / (np.nanmax(pred_rate,axis=0) - np.nanmin(pred_rate,axis=0))
    a = a*med
    
    return a
    
def set_background_fr(data,unit_locs,unit_types,bgr_val='med'):
    """
    data=data_train
    unit_locs = dinf['unit_locs']
    unit_types = dinf['unit_types']
    """
    if bgr_val == 'med':
        X = data.X
        y = data.y
        spikes = data.spikes
        
        y_units = mapToUnits(y,unit_locs,unit_types)
        unique_types = np.unique(unit_types)
        med_resp = []
        for t in range(len(unique_types)):
            med_resp.append(np.nanmedian(y_units[:,unit_types==unique_types[t]]))
        
        x_mesh,y_mesh = np.meshgrid(np.arange(y.shape[2]),np.arange(y.shape[1]))
        x_mesh = x_mesh.flatten()
        y_mesh = y_mesh.flatten()
        coords_allpix = np.array([x_mesh,y_mesh]).T
        
        coords_nonzeropixs = np.array([row for row in coords_allpix if not any(np.all(row == b) for b in unit_locs)])
        
        y_new = y
        for t in range(len(unique_types)):
            y_new[:,coords_nonzeropixs[:,1],coords_nonzeropixs[:,0],t] = med_resp[t]
            
        data = Exptdata_spikes(X,y_new,spikes)
        
        print('Background firing rate set to median')
        
    return data

def get_maskunitloc(unitlocs_train,unittypes_train,cell_types_unique,frame_shape):
    """
    frame_shape = [40,80]
    idx_units_train =  dinf['idx_units_train']
    unitlocs_train = dinf['unit_locs'][idx_units_train]
    unittypes_train = dinf['unit_types'][idx_units_train]
    
    """
    # unitlocs_train[unitlocs_train[:,0]>=frame_shape[1],0] = frame_shape[1]-1
    # unitlocs_train[unitlocs_train[:,1]>=frame_shape[0],1] = frame_shape[0]-1
    # unitlocs_train[unitlocs_train<0] = 0
    
    maskunitloc = np.zeros((frame_shape[0],frame_shape[1],len(cell_types_unique)),dtype='bool')
    for t in range(len(cell_types_unique)):
        idx_t = unittypes_train==cell_types_unique[t]
        unitlocs_t = unitlocs_train[idx_t]
        maskunitloc[unitlocs_t[:,1],unitlocs_t[:,0],t] = True
    
    return maskunitloc
    
def remap_unit_ids(coords):
    unique_ids, new_ids = np.unique(coords[:, 0], return_inverse=True)
    remapped_coords = coords.copy()
    remapped_coords[:,0] = new_ids
    return remapped_coords

def umask_metal_split(umaskcoords,FRAC_U_TRTR=0.95):
    """
    unames = data_quality['uname_selectedUnits']
    u_idx = data_quality['idx_unitsToTake']    
    FRAC_U_TRTR=0.75
    umaskcoords = dinf['umaskcoords_train']
    """
    
    cell_types = umaskcoords[:,1]
    cell_types_unique = np.unique(cell_types)
    
    umaskcoords_tr_subtr = np.zeros((0,4),dtype='int32')
    umaskcoords_tr_subval = np.zeros((0,4),dtype='int32')

    t=1
    for t in range(len(cell_types_unique)):
        umaskcoords_t = umaskcoords[umaskcoords[:,1]==cell_types_unique[t],:]
        
        unitids = np.unique(umaskcoords_t[:,0])
        n_tr = int((FRAC_U_TRTR*len(unitids)))
        unitids_tr = np.sort(np.random.choice(unitids,n_tr,replace=False))
        unitids_val = np.setdiff1d(unitids,unitids_tr)
                
        a = np.where(np.isin(umaskcoords_t[:,0], unitids_tr))[0]
        b = np.where(np.isin(umaskcoords_t[:,0], unitids_val))[0]

        umaskcoords_subtr = umaskcoords_t[a,:]
        umaskcoords_subval = umaskcoords_t[b,:]

        umaskcoords_tr_subtr = np.concatenate((umaskcoords_tr_subtr,umaskcoords_subtr),axis=0)
        umaskcoords_tr_subval = np.concatenate((umaskcoords_tr_subval,umaskcoords_subval),axis=0)
        
    umaskcoords_tr_subtr_remap = remap_unit_ids(umaskcoords_tr_subtr)
    umaskcoords_tr_subval_remap = remap_unit_ids(umaskcoords_tr_subval)


    return umaskcoords_tr_subtr,umaskcoords_tr_subval,umaskcoords_tr_subtr_remap,umaskcoords_tr_subval_remap


def prepare_metaldataset(data_train,umaskcoords_tr_tr,umaskcoords_tr_val,frac_stim_train=0.5,bgr=0,BUILD_MAPS=True):
    """
    1. Set to 0 the units in training set we want to use for validation gradients during metal and vice versa
    2. Select first half of stimuli to train and second half for validation gradients
    
    X = data_train.X
    y = data_train.y
    umaskcoords_tr_tr = dinf['umaskcoords_trtr']
    umaskcoords_tr_val = dinf['umaskcoords_trval']

    """
    # bgr = 0
    cell_types_unique = np.unique(umaskcoords_tr_tr[:,1])
    
    
    nsamps_tr = int(np.floor(frac_stim_train*len(np.arange(data_train.X.shape[0]))))
    train_y_tr = data_train.y[:nsamps_tr].copy()
    train_y_val = data_train.y[-nsamps_tr:].copy()

    assert train_y_tr.shape[0] == train_y_val.shape[0],'trtr and trval lengths not the same'

    if BUILD_MAPS==True:
        t=0
        for t in range(len(cell_types_unique)):
            
            # Set validation units to 0 in training set
            mask = umaskcoords_tr_val[:, 1] == cell_types_unique[t]
            b = umaskcoords_tr_val[mask,2:4]
            train_y_tr[:,b[:,1],b[:,0],t] = bgr
    
            # Set training units to 0 in validaion set
            mask = umaskcoords_tr_tr[:, 1] == cell_types_unique[t]
            b = umaskcoords_tr_tr[mask,2:4]
            train_y_val[:,b[:,1],b[:,0],t] = bgr
            
    else:
        idx_trtr = np.unique(umaskcoords_tr_tr[:,0])
        train_y_tr=train_y_tr[:,idx_trtr]
        
        idx_trval = np.unique(umaskcoords_tr_val[:,0])
        train_y_val=train_y_val[:,idx_trval]



    data_tr_tr = Exptdata(data_train.X[:nsamps_tr],train_y_tr)
    data_tr_val = Exptdata(data_train.X[-nsamps_tr:],train_y_val)

    return data_tr_tr,data_tr_val


def expand_dataset(data,nsamps_max,temporal_width_prepData):
    
    diff_dsetLen = (nsamps_max-temporal_width_prepData) - len(data.X)
    a = diff_dsetLen/len(data.X)
    if a > 1.1:
        a = int(np.ceil(a))
    else:
        a = 1
    expanded_X = data.X
    expanded_y = data.y
    if isintuple(data,'spikes'):
        expanded_spikes = data.spikes
        for j in range(a):
            expanded_X = expanded_X+data.X
            expanded_y = expanded_y+data.y
            expanded_spikes = expanded_spikes+data.spikes
        data = Exptdata_spikes(expanded_X,expanded_y,expanded_spikes)
    else:
        for j in range(a):
            expanded_X = expanded_X+data.X
            expanded_y = expanded_y+data.y
        data = Exptdata(expanded_X,expanded_y)

    
    return data

def get_expandedRGClist(data,MAX_RGCS):
    
    rgb = np.arange(data.y.shape[-1])
    num_rgcs_curr = rgb.shape[0]
    if num_rgcs_curr<MAX_RGCS:
        rgb = np.concatenate((rgb,np.tile(rgb[-1],MAX_RGCS-num_rgcs_curr)))
    idx_unitsToTake = rgb[:MAX_RGCS]
    mask_unitsToTake = np.ones_like(idx_unitsToTake)
    mask_unitsToTake[num_rgcs_curr:] = 0

    return idx_unitsToTake,mask_unitsToTake

# %%
"""
t=2
a = dinf['umaskcoords_train'][:,2:4]
a = a[dinf['umaskcoords_train'][:,1]==t]
rgb_tr = np.zeros((40,80))
rgb_tr[a[:,1],a[:,0]]=1
# plt.imshow(rgb_tr,cmap='gray');plt.show()

a = dinf['umaskcoords_val'][:,2:4]
a = a[dinf['umaskcoords_val'][:,1]==t]
rgb_val = np.zeros((40,80))
rgb_val[a[:,1],a[:,0]]=1
# plt.imshow(rgb_val,cmap='gray');plt.show()

rgb_tr_val = np.concatenate((rgb_tr[:,:,None],rgb_val[:,:,None],np.zeros_like(rgb_tr)[:,:,None]),axis=2).astype('int')
plt.imshow(rgb_tr_val*255)

"""