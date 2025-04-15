#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:29:42 2023

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

def fr_cones_gammafixed():
    pr_params = {}
    pr_params['sigma'] = 2.2
    pr_params['sigma_scaleFac'] = 10.
    pr_params['sigma_trainable'] = False
    
    pr_params['phi'] = 2.2
    pr_params['phi_scaleFac'] = 10.
    pr_params['phi_trainable'] = False

    pr_params['eta'] = 2.
    pr_params['eta_scaleFac'] = 1000.
    pr_params['eta_trainable'] = False

    pr_params['beta'] = 0.9
    pr_params['beta_scaleFac'] = 10.
    pr_params['beta_trainable'] = False

    pr_params['cgmp2cur'] = 0.01
    pr_params['cgmp2cur_scaleFac'] = 1.
    pr_params['cgmp2cur_trainable'] = False

    pr_params['cgmphill'] = 3.
    pr_params['cgmphill_scaleFac'] = 1.
    pr_params['cgmphill_trainable'] = False

    pr_params['cdark'] = 1.
    pr_params['cdark_scaleFac'] = 1.
    pr_params['cdark_trainable'] = False

    pr_params['betaSlow'] = 0.
    pr_params['betaSlow_scaleFac'] = 1.
    pr_params['betaSlow_trainable'] = False

    pr_params['hillcoef'] = 4.
    pr_params['hillcoef_scaleFac'] = 1.
    pr_params['hillcoef_trainable'] = False

    pr_params['hillaffinity'] = 0.5
    pr_params['hillaffinity_scaleFac'] = 1.
    pr_params['hillaffinity_trainable'] = False

    pr_params['gamma'] = 2.
    pr_params['gamma_scaleFac'] = 10.
    pr_params['gamma_trainable'] = False

    pr_params['gdark'] = 0.205#0.35
    pr_params['gdark_scaleFac'] = 100.
    pr_params['gdark_trainable'] = False

    pr_params['timeBin'] = 8
    
    return pr_params

def fr_cones_fixed():
    pr_params = {}
    pr_params['sigma'] = 2.2
    pr_params['sigma_scaleFac'] = 10.
    pr_params['sigma_trainable'] = False
    
    pr_params['phi'] = 2.2
    pr_params['phi_scaleFac'] = 10.
    pr_params['phi_trainable'] = False

    pr_params['eta'] = 2.
    pr_params['eta_scaleFac'] = 1000.
    pr_params['eta_trainable'] = False

    pr_params['beta'] = 0.9
    pr_params['beta_scaleFac'] = 10.
    pr_params['beta_trainable'] = False

    pr_params['cgmp2cur'] = 0.01
    pr_params['cgmp2cur_scaleFac'] = 1.
    pr_params['cgmp2cur_trainable'] = False

    pr_params['cgmphill'] = 3.
    pr_params['cgmphill_scaleFac'] = 1.
    pr_params['cgmphill_trainable'] = False

    pr_params['cdark'] = 1.
    pr_params['cdark_scaleFac'] = 1.
    pr_params['cdark_trainable'] = False

    pr_params['betaSlow'] = 0.
    pr_params['betaSlow_scaleFac'] = 1.
    pr_params['betaSlow_trainable'] = False

    pr_params['hillcoef'] = 4.
    pr_params['hillcoef_scaleFac'] = 1.
    pr_params['hillcoef_trainable'] = False

    pr_params['hillaffinity'] = 0.5
    pr_params['hillaffinity_scaleFac'] = 1.
    pr_params['hillaffinity_trainable'] = False

    pr_params['gamma'] = 1.
    pr_params['gamma_scaleFac'] = 10.
    pr_params['gamma_trainable'] = True

    pr_params['gdark'] = 0.205
    pr_params['gdark_scaleFac'] = 100.
    pr_params['gdark_trainable'] = False

    pr_params['timeBin'] = 8
    
    return pr_params


def fr_cones_trainable():
    pr_params = {}
    pr_params['sigma'] = 2.2
    pr_params['sigma_scaleFac'] = 10.
    pr_params['sigma_trainable'] = True
    
    pr_params['phi'] = 2.2
    pr_params['phi_scaleFac'] = 10.
    pr_params['phi_trainable'] = True

    pr_params['eta'] = 2.
    pr_params['eta_scaleFac'] = 1000.
    pr_params['eta_trainable'] = True

    pr_params['beta'] = 0.9
    pr_params['beta_scaleFac'] = 10.
    pr_params['beta_trainable'] = True

    pr_params['cgmp2cur'] = 0.01
    pr_params['cgmp2cur_scaleFac'] = 1.
    pr_params['cgmp2cur_trainable'] = False

    pr_params['cgmphill'] = 3.
    pr_params['cgmphill_scaleFac'] = 1.
    pr_params['cgmphill_trainable'] = False

    pr_params['cdark'] = 1.
    pr_params['cdark_scaleFac'] = 1.
    pr_params['cdark_trainable'] = False

    pr_params['betaSlow'] = 0.
    pr_params['betaSlow_scaleFac'] = 1.
    pr_params['betaSlow_trainable'] = False

    pr_params['hillcoef'] = 4.
    pr_params['hillcoef_scaleFac'] = 1.
    pr_params['hillcoef_trainable'] = False

    pr_params['hillaffinity'] = 0.5
    pr_params['hillaffinity_scaleFac'] = 1.
    pr_params['hillaffinity_trainable'] = False

    pr_params['gamma'] = 1.
    pr_params['gamma_scaleFac'] = 10.
    pr_params['gamma_trainable'] = True

    pr_params['gdark'] = 0.35
    pr_params['gdark_scaleFac'] = 100.
    pr_params['gdark_trainable'] = False

    pr_params['timeBin'] = 8
    
    return pr_params


def fr_cones_trainable_gdark0():
    pr_params = {}
    pr_params['sigma'] = 2.2
    pr_params['sigma_scaleFac'] = 10.
    pr_params['sigma_trainable'] = True
    
    pr_params['phi'] = 2.2
    pr_params['phi_scaleFac'] = 10.
    pr_params['phi_trainable'] = True

    pr_params['eta'] = 2.
    pr_params['eta_scaleFac'] = 1000.
    pr_params['eta_trainable'] = True

    pr_params['beta'] = 0.9
    pr_params['beta_scaleFac'] = 10.
    pr_params['beta_trainable'] = True

    pr_params['cgmp2cur'] = 0.01
    pr_params['cgmp2cur_scaleFac'] = 1.
    pr_params['cgmp2cur_trainable'] = False

    pr_params['cgmphill'] = 3.
    pr_params['cgmphill_scaleFac'] = 1.
    pr_params['cgmphill_trainable'] = False

    pr_params['cdark'] = 1.
    pr_params['cdark_scaleFac'] = 1.
    pr_params['cdark_trainable'] = False

    pr_params['betaSlow'] = 0.
    pr_params['betaSlow_scaleFac'] = 1.
    pr_params['betaSlow_trainable'] = False

    pr_params['hillcoef'] = 4.
    pr_params['hillcoef_scaleFac'] = 1.
    pr_params['hillcoef_trainable'] = False

    pr_params['hillaffinity'] = 0.5
    pr_params['hillaffinity_scaleFac'] = 1.
    pr_params['hillaffinity_trainable'] = False

    pr_params['gamma'] = 1.
    pr_params['gamma_scaleFac'] = 10.
    pr_params['gamma_trainable'] = True

    pr_params['gdark'] = 0.0001
    pr_params['gdark_scaleFac'] = 1.
    pr_params['gdark_trainable'] = False

    pr_params['timeBin'] = 8
    
    return pr_params

def fr_cones_trainable_beta0():
    pr_params = {}
    pr_params['sigma'] = 2.2
    pr_params['sigma_scaleFac'] = 10.
    pr_params['sigma_trainable'] = True
    
    pr_params['phi'] = 2.2
    pr_params['phi_scaleFac'] = 10.
    pr_params['phi_trainable'] = True

    pr_params['eta'] = 2.
    pr_params['eta_scaleFac'] = 1000.
    pr_params['eta_trainable'] = True

    pr_params['beta'] = 1e-6
    pr_params['beta_scaleFac'] = 1.
    pr_params['beta_trainable'] = False

    pr_params['cgmp2cur'] = 0.01
    pr_params['cgmp2cur_scaleFac'] = 1.
    pr_params['cgmp2cur_trainable'] = False

    pr_params['cgmphill'] = 3.
    pr_params['cgmphill_scaleFac'] = 1.
    pr_params['cgmphill_trainable'] = False

    pr_params['cdark'] = 1.
    pr_params['cdark_scaleFac'] = 1.
    pr_params['cdark_trainable'] = False

    pr_params['betaSlow'] = 0.
    pr_params['betaSlow_scaleFac'] = 1.
    pr_params['betaSlow_trainable'] = False

    pr_params['hillcoef'] = 4.
    pr_params['hillcoef_scaleFac'] = 1.
    pr_params['hillcoef_trainable'] = False

    pr_params['hillaffinity'] = 0.5
    pr_params['hillaffinity_scaleFac'] = 1.
    pr_params['hillaffinity_trainable'] = False

    pr_params['gamma'] = 1.
    pr_params['gamma_scaleFac'] = 10.
    pr_params['gamma_trainable'] = True

    pr_params['gdark'] = 0.35
    pr_params['gdark_scaleFac'] = 100.
    pr_params['gdark_trainable'] = False

    pr_params['timeBin'] = 8
    
    return pr_params



def fr_cones_beta0():
    pr_params = {}
    pr_params['sigma'] = 2.2
    pr_params['sigma_scaleFac'] = 10.
    pr_params['sigma_trainable'] = False
    
    pr_params['phi'] = 2.2
    pr_params['phi_scaleFac'] = 10.
    pr_params['phi_trainable'] = False

    pr_params['eta'] = 2.
    pr_params['eta_scaleFac'] = 1000.
    pr_params['eta_trainable'] = False

    pr_params['beta'] = 0.
    pr_params['beta_scaleFac'] = 1.
    pr_params['beta_trainable'] = False

    pr_params['cgmp2cur'] = 0.01
    pr_params['cgmp2cur_scaleFac'] = 1.
    pr_params['cgmp2cur_trainable'] = False

    pr_params['cgmphill'] = 3.
    pr_params['cgmphill_scaleFac'] = 1.
    pr_params['cgmphill_trainable'] = False

    pr_params['cdark'] = 1.
    pr_params['cdark_scaleFac'] = 1.
    pr_params['cdark_trainable'] = False

    pr_params['betaSlow'] = 0.
    pr_params['betaSlow_scaleFac'] = 1.
    pr_params['betaSlow_trainable'] = False

    pr_params['hillcoef'] = 4.
    pr_params['hillcoef_scaleFac'] = 1.
    pr_params['hillcoef_trainable'] = False

    pr_params['hillaffinity'] = 0.5
    pr_params['hillaffinity_scaleFac'] = 1.
    pr_params['hillaffinity_trainable'] = False

    pr_params['gamma'] = 1.
    pr_params['gamma_scaleFac'] = 10.
    pr_params['gamma_trainable'] = True

    pr_params['gdark'] = 0.35
    pr_params['gdark_scaleFac'] = 100.
    pr_params['gdark_trainable'] = False

    pr_params['timeBin'] = 8
    
    return pr_params


def fr_cones_trainable_all():
    pr_params = {}
    pr_params['sigma'] = .22
    pr_params['sigma_scaleFac'] = 100.
    pr_params['sigma_trainable'] = True
    
    pr_params['phi'] = .22
    pr_params['phi_scaleFac'] = 100.
    pr_params['phi_trainable'] = True

    pr_params['eta'] = 2.
    pr_params['eta_scaleFac'] = 1000.
    pr_params['eta_trainable'] = True

    pr_params['beta'] = 0.09
    pr_params['beta_scaleFac'] = 100.
    pr_params['beta_trainable'] = True

    pr_params['cgmp2cur'] = 0.01
    pr_params['cgmp2cur_scaleFac'] = 1.
    pr_params['cgmp2cur_trainable'] = True

    pr_params['cgmphill'] = 3.
    pr_params['cgmphill_scaleFac'] = 1.
    pr_params['cgmphill_trainable'] = True

    pr_params['cdark'] = 1.
    pr_params['cdark_scaleFac'] = 1.
    pr_params['cdark_trainable'] = True

    pr_params['betaSlow'] = 0.
    pr_params['betaSlow_scaleFac'] = 1.
    pr_params['betaSlow_trainable'] = True

    pr_params['hillcoef'] = 4.
    pr_params['hillcoef_scaleFac'] = 1.
    pr_params['hillcoef_trainable'] = True

    pr_params['hillaffinity'] = 0.5
    pr_params['hillaffinity_scaleFac'] = 1.
    pr_params['hillaffinity_trainable'] = True

    pr_params['gamma'] = 10.
    pr_params['gamma_scaleFac'] = 100.
    pr_params['gamma_trainable'] = True

    pr_params['gdark'] = 0.35
    pr_params['gdark_scaleFac'] = 100.
    pr_params['gdark_trainable'] = True

    pr_params['timeBin'] = 8
    
    return pr_params


# %% RODS

def fr_rods_fixed():
    pr_params = {}
    pr_params['sigma'] = 0.707
    pr_params['sigma_scaleFac'] = 10.
    pr_params['sigma_trainable'] = False
    
    pr_params['phi'] = 0.707
    pr_params['phi_scaleFac'] = 10.
    pr_params['phi_trainable'] = False

    pr_params['eta'] = 0.0253
    pr_params['eta_scaleFac'] = 100.
    pr_params['eta_trainable'] = False

    pr_params['beta'] = 0.25
    pr_params['beta_scaleFac'] = 100.
    pr_params['beta_trainable'] = False

    pr_params['cgmp2cur'] = 0.01
    pr_params['cgmp2cur_scaleFac'] = 1.
    pr_params['cgmp2cur_trainable'] = False

    pr_params['cgmphill'] = 3.
    pr_params['cgmphill_scaleFac'] = 1.
    pr_params['cgmphill_trainable'] = False

    pr_params['cdark'] = 1.
    pr_params['cdark_scaleFac'] = 1.
    pr_params['cdark_trainable'] = False

    pr_params['betaSlow'] = 0.
    pr_params['betaSlow_scaleFac'] = 1.
    pr_params['betaSlow_trainable'] = False

    pr_params['hillcoef'] = 4.
    pr_params['hillcoef_scaleFac'] = 1.
    pr_params['hillcoef_trainable'] = False

    pr_params['hillaffinity'] = 0.5
    pr_params['hillaffinity_scaleFac'] = 1.
    pr_params['hillaffinity_trainable'] = False

    pr_params['gamma'] = 1.
    pr_params['gamma_scaleFac'] = 10.
    pr_params['gamma_trainable'] = True

    pr_params['gdark'] = 0.155
    pr_params['gdark_scaleFac'] = 100.
    pr_params['gdark_trainable'] = False

    pr_params['timeBin'] = 8
    
    return pr_params

def fr_rods_fixed_beta0():
    pr_params = {}
    pr_params['sigma'] = 0.707
    pr_params['sigma_scaleFac'] = 10.
    pr_params['sigma_trainable'] = False
    
    pr_params['phi'] = 0.707
    pr_params['phi_scaleFac'] = 10.
    pr_params['phi_trainable'] = False

    pr_params['eta'] = 0.0253
    pr_params['eta_scaleFac'] = 100.
    pr_params['eta_trainable'] = False

    pr_params['beta'] = 0.
    pr_params['beta_scaleFac'] = 1.
    pr_params['beta_trainable'] = False

    pr_params['cgmp2cur'] = 0.01
    pr_params['cgmp2cur_scaleFac'] = 1.
    pr_params['cgmp2cur_trainable'] = False

    pr_params['cgmphill'] = 3.
    pr_params['cgmphill_scaleFac'] = 1.
    pr_params['cgmphill_trainable'] = False

    pr_params['cdark'] = 1.
    pr_params['cdark_scaleFac'] = 1.
    pr_params['cdark_trainable'] = False

    pr_params['betaSlow'] = 0.
    pr_params['betaSlow_scaleFac'] = 1.
    pr_params['betaSlow_trainable'] = False

    pr_params['hillcoef'] = 4.
    pr_params['hillcoef_scaleFac'] = 1.
    pr_params['hillcoef_trainable'] = False

    pr_params['hillaffinity'] = 0.5
    pr_params['hillaffinity_scaleFac'] = 1.
    pr_params['hillaffinity_trainable'] = False

    pr_params['gamma'] = 1.
    pr_params['gamma_scaleFac'] = 10.
    pr_params['gamma_trainable'] = True

    pr_params['gdark'] = 0.155
    pr_params['gdark_scaleFac'] = 100.
    pr_params['gdark_trainable'] = False

    pr_params['timeBin'] = 8
    
    return pr_params

def fr_rods_trainable():
    pr_params = {}
    pr_params['sigma'] = 0.707
    pr_params['sigma_scaleFac'] = 10.
    pr_params['sigma_trainable'] = True
    
    pr_params['phi'] = 0.707
    pr_params['phi_scaleFac'] = 10.
    pr_params['phi_trainable'] = True

    pr_params['eta'] = 0.0253
    pr_params['eta_scaleFac'] = 100.
    pr_params['eta_trainable'] = True

    pr_params['beta'] = 0.25
    pr_params['beta_scaleFac'] = 100.
    pr_params['beta_trainable'] = True

    pr_params['cgmp2cur'] = 0.01
    pr_params['cgmp2cur_scaleFac'] = 1.
    pr_params['cgmp2cur_trainable'] = False

    pr_params['cgmphill'] = 3.
    pr_params['cgmphill_scaleFac'] = 1.
    pr_params['cgmphill_trainable'] = False

    pr_params['cdark'] = 1.
    pr_params['cdark_scaleFac'] = 1.
    pr_params['cdark_trainable'] = False

    pr_params['betaSlow'] = 0.
    pr_params['betaSlow_scaleFac'] = 1.
    pr_params['betaSlow_trainable'] = False

    pr_params['hillcoef'] = 4.
    pr_params['hillcoef_scaleFac'] = 1.
    pr_params['hillcoef_trainable'] = False

    pr_params['hillaffinity'] = 0.5
    pr_params['hillaffinity_scaleFac'] = 1.
    pr_params['hillaffinity_trainable'] = False

    pr_params['gamma'] = 1.
    pr_params['gamma_scaleFac'] = 10.
    pr_params['gamma_trainable'] = True

    pr_params['gdark'] = 0.155
    pr_params['gdark_scaleFac'] = 100.
    pr_params['gdark_trainable'] = False

    pr_params['timeBin'] = 8
    
    return pr_params


def fr_rods_trainable_beta0():
    pr_params = {}
    pr_params['sigma'] = 0.707
    pr_params['sigma_scaleFac'] = 10.
    pr_params['sigma_trainable'] = True
    
    pr_params['phi'] = 0.707
    pr_params['phi_scaleFac'] = 10.
    pr_params['phi_trainable'] = True

    pr_params['eta'] = 0.0253
    pr_params['eta_scaleFac'] = 100.
    pr_params['eta_trainable'] = True

    pr_params['beta'] = 0.
    pr_params['beta_scaleFac'] = 1.
    pr_params['beta_trainable'] = False

    pr_params['cgmp2cur'] = 0.01
    pr_params['cgmp2cur_scaleFac'] = 1.
    pr_params['cgmp2cur_trainable'] = False

    pr_params['cgmphill'] = 3.
    pr_params['cgmphill_scaleFac'] = 1.
    pr_params['cgmphill_trainable'] = False

    pr_params['cdark'] = 1.
    pr_params['cdark_scaleFac'] = 1.
    pr_params['cdark_trainable'] = False

    pr_params['betaSlow'] = 0.
    pr_params['betaSlow_scaleFac'] = 1.
    pr_params['betaSlow_trainable'] = False

    pr_params['hillcoef'] = 4.
    pr_params['hillcoef_scaleFac'] = 1.
    pr_params['hillcoef_trainable'] = False

    pr_params['hillaffinity'] = 0.5
    pr_params['hillaffinity_scaleFac'] = 1.
    pr_params['hillaffinity_trainable'] = False

    pr_params['gamma'] = 1.
    pr_params['gamma_scaleFac'] = 10.
    pr_params['gamma_trainable'] = True

    pr_params['gdark'] = 0.155
    pr_params['gdark_scaleFac'] = 100.
    pr_params['gdark_trainable'] = False

    pr_params['timeBin'] = 8
    
    return pr_params
# %% LinearCone Params
def prln_cones_trainable():
    pr_params = {}
    pr_params['alpha'] = 4.0
    pr_params['alpha_scaleFac'] = 10.
    pr_params['alpha_trainable'] = True
    
    pr_params['tau_r'] = 2.8
    pr_params['tau_r_scaleFac'] = 10.
    pr_params['tau_r_trainable'] = True

    pr_params['tau_d'] = 2.4
    pr_params['tau_d_scaleFac'] = 10.
    pr_params['tau_d_trainable'] = True

    pr_params['tau_o'] = 200
    pr_params['tau_o_scaleFac'] = 10.
    pr_params['tau_o_trainable'] = True

    pr_params['omega'] = 9
    pr_params['omega_scaleFac'] = 10.
    pr_params['omega_trainable'] = True

    return pr_params


