#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:33:32 2024

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

import h5py
import numpy as np
import os
import tensorflow as tf
import gc

# def get_grads(mdl, inp):
#     with tf.GradientTape() as tape:
#         out = mdl(inp)
#         None(None, None, None)

# get_grads = tf.function(get_grads)

def save_gradDataset(fname_gradsFile, mdl_select, data_alldsets):
    f = h5py.File(fname_gradsFile, 'a')
    lightLevels = list(data_alldsets.keys())
    if '/' + mdl_select not in f:
        grp = f.create_group('/' + mdl_select)


def init_GradDataset(f, select_mdl, d, grads_shape, stim_shape, batchsize = (4,)):
    if '/' + select_mdl not in f:
        grp = f.create_group('/' + select_mdl)
    grp_path = '/' + select_mdl + '/' + d
    if grp_path not in f:
        grp = f.create_group(grp_path)
    dset_grads = grp.create_dataset('grads', shape=(grads_shape[0], 0) + grads_shape[2:], dtype='float16', maxshape=grads_shape, chunks=(1, batchsize) + grads_shape[2:])#, **('shape', 'dtype', 'maxshape', 'chunks'))
    return grp


def append_GradDataset(f, grp, grads_chunk, stim_chunk):
    grp['grads'].resize(grp['grads'].shape[1] + grads_chunk.shape[1], axis=1)
    grp['grads'][:, -grads_chunk.shape[1]:] = grads_chunk


def load_gradDataset(fnames_gradsFiles):
    dict_grads = { }
    fname_select = fnames_gradsFiles[0]
    for fname_select in fnames_gradsFiles:
        f = h5py.File(fname_select, 'r')
        modelsInGradFile = list(f.keys())
        for m in modelsInGradFile:
            dict_grads[m] = { }
            lightLevelsInFile = list(f[m].keys())
            for l in lightLevelsInFile:
                dict_grads[m][l] = { }
                keys_lightLevel = f[m][l].keys()
                for k in keys_lightLevel:
                    dict_grads[m][l][k] = np.array(f[m][l][k])
    f.close()
    return dict_grads
