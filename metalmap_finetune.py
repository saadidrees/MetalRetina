#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 08:16:18 2025

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""



from model.parser import parser_finetune


def run_finetune(ft_expDate,path_pretrained,ft_fname_data_train_val_test,ft_mdl_name,ft_path_model_base,ft_trainingSamps_dur=-1,
                 validationSamps_dur=0.5,nb_epochs=5,ft_lr=0.001,batch_size=256,job_id=0,saveToCSV=1,CONTINUE_TRAINING=1):

# %%
    import numpy as np
    import os
    import time
    import csv
    import h5py
    import glob
    import re
    import matplotlib.pyplot as plt
    import cloudpickle
    import jax.numpy as jnp
    import jax
    import optax
    import orbax
    import gc
    import datetime
    from collections import namedtuple
    from tqdm.auto import tqdm
    import shutil
    
    from model.data_handler import prepare_data_cnn2d_maps,isintuple
    from model import data_handler_ej
    from model.data_handler_mike import load_h5Dataset
    from model.performance import model_evaluate_new, estimate_noise
    import model.paramsLogger
    import model.utils_si
    
    from model import models_jax, train_singleretunits, dataloaders,handler_maps
    from model import train_metalmaps
    from torch.utils.data import DataLoader
    
    from model.performance import getModelParams
    import model.prfr_params

        
    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    Exptdata_spikes = namedtuple('Exptdata', ['X', 'y','spikes'
                                              ])
     
    fname_pretrained = os.path.split(path_pretrained[:-1])[-1]
    pretrained_params = model.performance.getModelParams(path_pretrained)

    
    # % load train val and test datasets from saved h5 file
    """
        load_h5dataset is a function to load training and validation data from h5 dataset. We can extract all data or a subset using the nsamps arguments.
        data_train, val and test are named tuples. data_train.X contains the stimulus with dimensions [samples,y pixels, x pixels]
        and data_train.y contains the spikerate normalized by median [samples,numOfCells]
    """
    BUILD_MAPS = False
    MAX_RGCS = model.train_metalmaps.MAX_RGCS
    
    
    # Load datasets
    idx_train_start = 0    # mins to chop off in the begining.
    psf_params = dict(pixel_neigh=1,method='cross')
    FRAC_U_TRTR = 1
    d=0
    dict_train = {};dict_trtr={};dict_trval={}
    dict_val = {}
    dict_test = {}
    unames_allDsets = []
    data_info = {}
    dict_dinf = {}
    nsamps_alldsets_loaded = []
    
    rgb = load_h5Dataset(ft_fname_data_train_val_test,nsamps_val=validationSamps_dur,nsamps_train=ft_trainingSamps_dur,nsamps_test=0,  # THIS NEEDS TO BE TIDIED UP
                         idx_train_start=idx_train_start)
    data_train=rgb[0]
    data_val = rgb[1]
    data_test = rgb[2]
    data_quality = rgb[3]
    dataset_rr = rgb[4]
    parameters = rgb[5]
    if len(rgb)>7:
        data_info = rgb[7]
        
    info_unitSplit = data_handler_ej.load_info_unitSplit(ft_fname_data_train_val_test)
    info_unitSplit['unames_train'] = parameters['unames'][info_unitSplit['idx_train']]
    info_unitSplit['unames_val'] = parameters['unames'][info_unitSplit['idx_val']]
    
    
    t_frame = parameters['t_frame']     # time in ms of one frame/sample 
    
    frac_train_units = parameters['frac_train_units']
    
    # Setting MODE to validation ensures that Only stim is different in Train/Val sets and same RGCs are used
    data_train,data_val,dinf = handler_maps.arrange_data_formaps(ft_expDate,data_train,data_val,parameters,frac_train_units,psf_params=psf_params,info_unitSplit=info_unitSplit,
                                                                 BUILD_MAPS=BUILD_MAPS,MODE='training',NORMALIZE_RESP=1)       
    
    dinf['unit_locs_train'] = dinf['unit_locs'][dinf['idx_units_train']]
    dinf['unit_types_train'] = dinf['unit_types'][dinf['idx_units_train']]
    
    dinf['unit_locs_val'] = dinf['unit_locs'][dinf['idx_units_val']]
    dinf['unit_types_val'] = dinf['unit_types'][dinf['idx_units_val']]
    
    
    # FRAC_U_TRTR of 1 means dont hold out any cells for TRVAL
    dinf['umaskcoords_trtr'],dinf['umaskcoords_trval'],dinf['umaskcoords_trtr_remap'],dinf['umaskcoords_trval_remap'] = handler_maps.umask_metal_split(dinf['umaskcoords_train'],
                                                                                                                                                       FRAC_U_TRTR=FRAC_U_TRTR)     
    
    data_trtr,data_trval = handler_maps.prepare_metaldataset(data_train,dinf['umaskcoords_trtr'],dinf['umaskcoords_trval'],bgr=0,frac_stim_train=0.5,BUILD_MAPS=BUILD_MAPS)
    
    del data_train, data_test
    
    
    ft_fname_data_train_val_test = os.path.split(ft_fname_data_train_val_test)[-1]
    dict_trtr[ft_fname_data_train_val_test] = data_trtr   
    dict_val[ft_fname_data_train_val_test] = data_val
    dict_dinf[ft_fname_data_train_val_test] = dinf
    unames_allDsets.append(parameters['unames'])
    nsamps_alldsets_loaded.append(len(data_trtr.X))
        
    n_tasks = len(dict_val)
    cell_types_unique = np.unique(dinf['umaskcoords_train'][:,1])
        
    nrgcs_trtr = dict_trtr[ft_fname_data_train_val_test].y.shape[-1]
    nrgcs_val = dict_val[ft_fname_data_train_val_test].y.shape[-1]
    
    
    # %
    if BUILD_MAPS==False:   
        idx_unitsToTake,mask_unitsToTake = handler_maps.get_expandedRGClist(dict_trtr[ft_fname_data_train_val_test],MAX_RGCS)
        idx_unitsToTake_all_trtr = idx_unitsToTake
        mask_unitsToTake_all_trtr = mask_unitsToTake
        
        idx_unitsToTake,mask_unitsToTake = handler_maps.get_expandedRGClist(dict_val[ft_fname_data_train_val_test],MAX_RGCS)
        idx_unitsToTake_all_val = idx_unitsToTake
        mask_unitsToTake_all_val = mask_unitsToTake
    
    # Get unit names
    uname_train = [];uname_val = [];
    c_tr = np.zeros(len(cell_types_unique),dtype='int'); c_val = np.zeros(len(cell_types_unique),dtype='int');
    
    dinf = dict_dinf[ft_fname_data_train_val_test]
    rgb = dinf['unames'][dinf['idx_units_train']]
    exp_uname = ft_expDate
    uname_train.append(exp_uname)
    
    for t in range(len(cell_types_unique)):
        c_tr[t]=c_tr[t]+(dinf['unit_types'][dinf['idx_units_train']]==cell_types_unique[t]).sum()
    
    rgb = dinf['unames'][dinf['idx_units_val']]
    uname_val.append(exp_uname)
    
    for t in range(len(cell_types_unique)):
        c_val[t]=c_val[t]+(dinf['unit_types'][dinf['idx_units_val']]==cell_types_unique[t]).sum()
    
    print('Total number of datasets: %d'%n_tasks)
    for t in range(len(cell_types_unique)):
        print('Trainining set | Cell type %d: %d RGCs'%(cell_types_unique[t],c_tr[t]))
        print('Validation set | Cell type %d: %d RGCs'%(cell_types_unique[t],c_val[t]))
    
    
    # Get pre-trained model params
    pretrained_params = model.performance.getModelParams(path_pretrained)

    
    # Data will be rolled so that each sample has a temporal width. Like N frames of movie in one sample. The duration of each frame is in t_frame
    # if the model has a photoreceptor layer, then the PR layer has a termporal width of pr_temporal_width, which before convs will be chopped off to temporal width
    # this is done to get rid of boundary effects. pr_temporal_width > temporal width
    if ft_mdl_name[:2] == 'PR':    # in this case the rolling width should be that of PR
        temporal_width_prepData = pretrained_params['P']
        temporal_width_eval = pretrained_params['P']
        
    else:   # in all other cases its same as temporal width
        temporal_width_prepData = pretrained_params['T']
        temporal_width_eval = pretrained_params['T']    # termporal width of each sample. Like how many frames of movie in one sample
        pr_temporal_width = 0
    
    
    modelNames_all = models_jax.model_definitions()    # get all model names
    modelNames_2D = modelNames_all[0]
    modelNames_3D = modelNames_all[1]

    # Expand dataset if needed for vectorization, and roll it for temporal dimension
    print(ft_fname_data_train_val_test)
    data_trtr = dict_trtr[ft_fname_data_train_val_test]
    data_val = dict_val[ft_fname_data_train_val_test]
    
    if ft_mdl_name in modelNames_2D:
        if BUILD_MAPS==True:
            idx_unitsToTake_trtr = []
            idx_unitsToTake_trval = []
            idx_unitsToTake_val = []
        else:
            idx_unitsToTake_trtr = idx_unitsToTake_all_trtr
            idx_unitsToTake_val = idx_unitsToTake_all_val

        data_trtr = prepare_data_cnn2d_maps(data_trtr,temporal_width_prepData,MAKE_LISTS=True,idx_unitsToTake=idx_unitsToTake_trtr)     # [samples,temporal_width,rows,columns]
        data_val = prepare_data_cnn2d_maps(data_val,temporal_width_prepData,MAKE_LISTS=True,idx_unitsToTake=idx_unitsToTake_val)   
            
    else:
        raise ValueError('model not found')

    dict_trtr[ft_fname_data_train_val_test] = data_trtr
    dict_val[ft_fname_data_train_val_test] = data_val
   
    # Shuffle just the training dataset
    dict_trtr = dataloaders.shuffle_dataset(dict_trtr)    
       
    maxLen_umaskcoords_tr_subtr = max([len(value['umaskcoords_trtr_remap']) for value in dict_dinf.values()])
    maxLen_umaskcoords_val = max([len(value['umaskcoords_val']) for value in dict_dinf.values()])
    
    
    # ---- PACK MASKS AND COORDINATES FOR EACH RGC
    b_umaskcoords_trtr=-1*np.ones((maxLen_umaskcoords_tr_subtr,dinf['umaskcoords_trtr_remap'].shape[1]),dtype='int32');
    b_umaskcoords_val=-1*np.ones((maxLen_umaskcoords_val,dinf['umaskcoords_val'].shape[1]),dtype='int32')
    
    maskunits_trtr = np.zeros((MAX_RGCS),dtype='int')
    maskunits_val = np.zeros((MAX_RGCS),dtype='int')
    
    d=0
    rgb = dict_dinf[ft_fname_data_train_val_test]['umaskcoords_trtr_remap']
    b_umaskcoords_trtr[:len(rgb),:] = rgb
    N_trtr = len(np.unique(rgb[:,0]))
    maskunits_trtr[:N_trtr] = 1
    
    rgb = dict_dinf[ft_fname_data_train_val_test]['umaskcoords_val']
    b_umaskcoords_val[:len(rgb),:] = rgb
    N_val = len(np.unique(rgb[:,0]))
    maskunits_val[:N_val] = 1
    
    segment_size = dict_dinf[ft_fname_data_train_val_test]['segment_size']
        
        
    dinf_tr = dict(umaskcoords_trtr=b_umaskcoords_trtr,
                      N_trtr=N_trtr,
                      maskunits_trtr=maskunits_trtr,
                      cell_types_unique=cell_types_unique,
                      segment_size=segment_size,
                      MAX_RGCS=MAX_RGCS)
    
    dinf_val = dict(umaskcoords_val=b_umaskcoords_val,
                      N_val=N_val,
                      maskunits_val=maskunits_val,
                      segment_size=segment_size,
                      MAX_RGCS=MAX_RGCS)
    
    
    # PRINT INFO
    nsamps_train = 0;dset_names=[]
    dset = ft_fname_data_train_val_test
    rgb = re.split('_',os.path.split(dset)[-1])[0]
    dset_names.append(rgb)
    nsamps_train = nsamps_train+len(dict_val[dset].X)+len(dict_val[dset].X)
    
    run_info = dict(
                    nsamps_train=nsamps_train,
                    nsamps_unique_alldsets=nsamps_train,
                    nunits_train=c_tr.sum(),
                    nunits_val=c_val.sum(),
                    nexps = n_tasks,
                    exps = ft_expDate
                    )
      
    # ---- Dataloaders  
    assert MAX_RGCS > c_tr.sum().max(), 'MAX_RGCS limit lower than maximum RGCs in a dataset'
    
    n_tasks = 1
    assert n_tasks==1, 'Finetuning pipeline only takes 1 dataset'
    batch_size_train = batch_size
    
    dset = ft_fname_data_train_val_test
    data_trtr = dict_trtr[dset]
    RetinaDataset_train = dataloaders.RetinaDataset(data_trtr.X,data_trtr.y,transform=None)
    dataloader_train = DataLoader(RetinaDataset_train,batch_size=batch_size_train,collate_fn=dataloaders.jnp_collate);
    
    # batch = next(iter(dataloader_train));a,b,c,d=batch
    
    data_val = dict_val[dset]
    RetinaDataset_val = dataloaders.RetinaDataset(data_val.X,data_val.y,transform=None)
    dataloader_val = DataLoader(RetinaDataset_val,batch_size=128,collate_fn=dataloaders.jnp_collate);
    # batch = next(iter(dataloader_val));a,b=batch
      
    n_batches = len(dataloader_train)#np.ceil(len(data_train.X)/bz)
    
    # %% Load Pre-Trained Model 
    temporal_width = pretrained_params['T']
    dict_params = {}
    dict_params['filt_temporal_width'] = temporal_width
    dict_params['nout'] = len(cell_types_unique)
    
    if not os.path.isabs(path_pretrained):
        path_pretrained = os.path.join(os.getcwd(),path_pretrained)
    
    allCps = glob.glob(path_pretrained+'/step*')
    assert  len(allCps)!=0, 'No checkpoints found'
    
    step_numbers = np.sort(np.asarray([int(re.search(r'step-(\d+)', s).group(1)) for s in allCps]))
    nb_cps = len(allCps)
    
    with open(os.path.join(path_pretrained,'model_architecture.pkl'), 'rb') as f:
        mdl,config = cloudpickle.load(f)
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    
    # This loop helps see the valid checkpoints. For some models my loss goes into Nan. Need to figure out still
    bias_allSteps=[]; weights_allSteps=[]
    for i in tqdm(range(nb_cps)):
        fname_latestWeights = os.path.join(path_pretrained,'step-%03d' % step_numbers[i])
        raw_restored = orbax_checkpointer.restore(fname_latestWeights)
        mdl_state = train_metalmaps.load(mdl,raw_restored['model'],pretrained_params['LR'])
        
        weights = mdl_state.params
        output_bias = np.array(weights['output']['bias'])
        bias_allSteps.append(output_bias.sum())
        weights_allSteps.append(weights)
    
    if sum(np.isnan(bias_allSteps))>0:
        last_cp = np.where(np.isnan(bias_allSteps))[0][0]-1     # Where the weights are not nan
    else:
        last_cp = nb_cps-1
    assert np.isnan(weights_allSteps[last_cp]['output']['kernel']).sum()==0, 'Model checkpoint has NaN values'
    
    # last_cp = step_numbers[last_cp]
    last_cp = step_numbers[7]

    
    fname_latestWeights = os.path.join(path_pretrained,'step-%03d' % last_cp)
    
    raw_restored = orbax_checkpointer.restore(fname_latestWeights)
    pre_mdl_state = train_metalmaps.load(mdl,raw_restored['model'],pretrained_params['LR'])
    
    print('Loaded pre-trained model')
        
    # %% Create FT Model
    initial_epoch=0
    CONTINUE_TRAINING=0

    LOSS_FUN = model.train_metalmaps.LOSS_FUN
    ft_fname_model,ft_model_params = model.utils_si.modelFileName(U=nrgcs_trtr,P=pretrained_params['P'],T=temporal_width,CB_n=0,
                                                        C1_n=pretrained_params['C1_n'],C1_s=pretrained_params['C1_s'],C1_3d=pretrained_params['C1_3d'],
                                                        C2_n=pretrained_params['C2_n'],C2_s=pretrained_params['C2_s'],C2_3d=pretrained_params['C2_3d'],
                                                        C3_n=pretrained_params['C3_n'],C3_s=pretrained_params['C3_s'],C3_3d=pretrained_params['C3_3d'],
                                                        C4_n=pretrained_params['C4_n'],C4_s=pretrained_params['C4_s'],C4_3d=pretrained_params['C4_3d'],
                                                        BN=pretrained_params['BN'],MP=pretrained_params['MP'],LR=ft_lr,
                                                        TR=pretrained_params['TR'],TRSAMPS=ft_trainingSamps_dur)

    ft_model_params['filt_temporal_width'] = temporal_width
    ft_model_params['nout'] = len(cell_types_unique)        

    ft_path_model_save = os.path.join(ft_path_model_base,ft_expDate,ft_fname_model)   # the model save directory is the fname_model appened to save path

    if not os.path.isabs(ft_path_model_save):
        ft_path_model_save = os.path.join(os.getcwd(),ft_path_model_save)

    if CONTINUE_TRAINING==0 and os.path.exists(ft_path_model_save):
        shutil.rmtree(ft_path_model_save)  # Remove any existing checkpoints from the last notebook run.
        os.makedirs(ft_path_model_save)
    elif not os.path.exists(ft_path_model_save):
        os.makedirs(ft_path_model_save)


    ft_path_save_model_performance = os.path.join(ft_path_model_save,'performance')
    if not os.path.exists(ft_path_save_model_performance):
        os.makedirs(ft_path_save_model_performance)
        

    training_params = dict(LOSS_FUN=LOSS_FUN)


    fname_excel = 'performance_'+ft_fname_model+'.csv'

    transition_steps = int(n_batches*1)# 20000
    # lr_schedule = optax.exponential_decay(init_value=ft_lr,transition_steps=transition_steps,decay_rate=0.5,staircase=True,transition_begin=0,end_value=ft_lr/100)
    lr_schedule = optax.constant_schedule(ft_lr)

    total_steps = n_batches*nb_epochs
    rgb_lrs = [lr_schedule(i) for i in range(total_steps)]
    rgb_lrs = np.array(rgb_lrs)
    plt.plot(rgb_lrs);plt.show()
    print(np.array(rgb_lrs))


    # All pr params have a field '..._trainable' which if we set true will be updated during finetuning
    if 'PR' in ft_mdl_name:
        ft_pr_params_name = 'fr_cones_gammalarge'
        pr_params_fun = getattr(model.prfr_params,ft_pr_params_name)
        ft_pr_params = pr_params_fun()
        ft_pr_params['sigma_trainable']=True
        ft_pr_params['phi_trainable']=True
        ft_pr_params['eta_trainable']=True
        ft_pr_params['beta_trainable']=True
        ft_pr_params['gamma_trainable']=False
        ft_pr_params['cgmp2cur_trainable'] = False
        ft_pr_params['cgmphill_trainable'] = False
        ft_pr_params['cdark_trainable'] = False
        ft_pr_params['hillcoef_trainable'] = False
        ft_pr_params['hillaffinity_trainable'] = False
        ft_pr_params['gdark_trainable'] = False

        ft_model_params['pr_params'] = ft_pr_params

    # We initialize a new instance of the model of type ft_mdl_name
    inp_shape = data_val.X[0].shape
    model_func = getattr(models_jax,ft_mdl_name) 
    ft_mdl = model_func
    ft_mdl_state,ft_mdl,ft_config = train_metalmaps.initialize_model(ft_mdl,ft_model_params,inp_shape,ft_lr,save_model=True,lr_schedule=lr_schedule)
    models_jax.model_summary(ft_mdl,inp_shape,console_kwargs={'width':180})


    # % Select layers to finetune
    layers_finetune = [key for key in pre_mdl_state.params]  # This selects all layers in the pretrained model
    # layers_finetune = ['PRFR_0','LayerNorm_0','output','LayerNorm_4','TrainableAF_3']        # Or you can select manually
    ft_params_fixed,ft_params_trainable = train_metalmaps.split_dict(pre_mdl_state.params,layers_finetune)  # And then see which layers will be fixed and which will be finetuned

    # Do this if we want the PR layer to also be trainable
    if 'PR' in ft_mdl_name:
        ft_params_trainable['PRFR_0'] = ft_mdl_state.params['PRFR_0']

    print('Fixed layers:');print(ft_params_fixed.keys())
    print('Trainable layers:');print(ft_params_trainable.keys())

          
    # Copy all weights from pre-trained model (For every layer)
    for key in pre_mdl_state.params:
        ft_mdl_state.params[key] = pre_mdl_state.params[key]

    # If you want to initialize a layer to random weights do it here

    # Compile the model with the trainable param list
    optimizer = optax.adam(learning_rate=lr_schedule) #,weight_decay=1e-4)
    ft_mdl_state = train_metalmaps.TrainState.create(
                apply_fn=ft_mdl.apply,
                params=ft_params_trainable,
                tx=optimizer)

    if CONTINUE_TRAINING==1:
        allCps = glob.glob(ft_path_model_save+'/step*')
        
        if len(allCps)>0:
            step_numbers = np.sort(np.asarray([int(re.search(r'step-(\d+)', s).group(1)) for s in allCps]))
            nb_cps = len(allCps)
            last_cp = step_numbers[nb_cps-1]
            
            initial_epoch = nb_cps-1           
        else:
            initial_epoch = 0


    # Get the performance of dataset on the pretrained model. So sort of the initial performance
    val_loss,pred_rate,y,pred_rate_units,y_units = train_metalmaps.ft_eval_step(ft_mdl_state,ft_params_fixed,dataloader_val,dinf_val)
    val_loss = np.mean(val_loss)
    fev, fracExVar, predCorr, rrCorr = model_evaluate_new(y_units,pred_rate_units,temporal_width_eval,lag=0,obs_noise=0)
    print('Pre-trained perf | loss: %f, fev: %0.2f, Corr: %0.2f'%(np.mean(val_loss),np.median(fev),np.median(predCorr)))
    pre_fev_val_allUnits_lastEpoch = fev
    pre_corr_val_allUnits_lastEpoch = predCorr
        

    # %% Fine-tune the model
    cp_interval = n_batches
    training_params['cp_interval'] = cp_interval
    
    t_elapsed = 0
    t = time.time()
    if initial_epoch < nb_epochs:
        print('-----RUNNING MODEL-----')
        
        loss_currEpoch_master,loss_epoch_train,loss_epoch_val,ft_mdl_state,fev_epoch_train,fev_epoch_val = train_metalmaps.ft_train(ft_mdl_state,ft_params_fixed,config,training_params,\
                                                                                      dataloader_train,dataloader_val,dinf_tr,dinf_val,nb_epochs,ft_path_model_save,save=True,\
                                                                                      lr_schedule=lr_schedule,step_start=initial_epoch+1)
        _ = gc.collect()
            
    t_elapsed = time.time()-t
    print('time elapsed: '+str(t_elapsed)+' seconds')
    
    # %% Evaluate performance
    
    #  Load checkpoint infos
    allCps = glob.glob(ft_path_model_save+'/step*')
    assert  len(allCps)!=0, 'No checkpoints found'

    step_numbers = np.sort(np.asarray([int(re.search(r'step-(\d+)', s).group(1)) for s in allCps]))
    nb_cps = len(allCps)
    last_cp = step_numbers[nb_cps-1]
       
    n_cells = dinf_val['N_val']
    cps_sel = np.arange(0,len(step_numbers)).astype('int')
    nb_cps_sel = len(cps_sel)

    val_loss_allEpochs = np.empty(nb_cps_sel+1)
    val_loss_allEpochs[:] = np.nan

    fev_medianUnits_allEpochs = np.empty(nb_cps_sel+1)
    fev_medianUnits_allEpochs[:] = np.nan
    fev_allUnits_allEpochs = np.zeros((nb_cps_sel+1,n_cells))
    fev_allUnits_allEpochs[:] = np.nan

    predCorr_medianUnits_allEpochs = np.empty(nb_cps_sel+1)
    predCorr_medianUnits_allEpochs[:] = np.nan
    predCorr_allUnits_allEpochs = np.zeros((nb_cps_sel+1,n_cells))
    predCorr_allUnits_allEpochs[:] = np.nan

    data_val = dict_val[ft_fname_data_train_val_test]

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    RetinaDataset_val = dataloaders.RetinaDataset(data_val.X,data_val.y,transform=None)
    dataloader_val = DataLoader(RetinaDataset_val,batch_size=batch_size,collate_fn=dataloaders.jnp_collate)


    # %
    print('-----EVALUATING PERFORMANCE-----')
    i=nb_cps_sel-1
    for i in tqdm(range(0,nb_cps_sel+1)): #[nb_cps_sel-1]: #tqdm(range(0,nb_cps_sel-1)):
        
        if i==0:    # That is randomly initialized
            mdl_state_eval = pre_mdl_state
        else:
            mdl_state_eval = ft_mdl_state
            weight_fold = 'step-%03d' %(step_numbers[cps_sel[i-1]])  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
            weight_fold = os.path.join(ft_path_model_save,weight_fold)
        
            assert os.path.isdir(weight_fold)==True, 'Checkpoint %d does not exist'%step_numbers[cps_sel[i-1]]
            raw_restored = orbax_checkpointer.restore(weight_fold)
            mdl_state_eval = train_metalmaps.load(ft_mdl,raw_restored['model'],ft_lr)

                
        val_loss,pred_rate,y,pred_rate_units,y_units = train_metalmaps.ft_eval_step(mdl_state_eval,ft_params_fixed,dataloader_val,dinf_val)
        val_loss = np.mean(val_loss)

        val_loss_allEpochs[i] = val_loss
        

        fev, fracExVar, predCorr, rrCorr = model_evaluate_new(y_units,pred_rate_units,temporal_width_eval,lag=0,obs_noise=0)
                
        fev_allUnits_allEpochs[i,:] = fev
        fev_medianUnits_allEpochs[i] = np.nanmedian(fev)      
        
        predCorr_allUnits_allEpochs[i,:] = predCorr
        predCorr_medianUnits_allEpochs[i] = np.nanmedian(predCorr)

        _ = gc.collect()
        
    fig,axs = plt.subplots(1,2,figsize=(14,5)); fig.suptitle(dset_names)
    axs[0].plot(step_numbers[cps_sel],predCorr_medianUnits_allEpochs[:len(cps_sel)])
    axs[0].set_xlabel('Training steps');axs[0].set_ylabel('Corr'); 
    axs[1].plot(step_numbers[cps_sel],fev_medianUnits_allEpochs[:len(cps_sel)])
    axs[1].set_xlabel('Training steps');axs[1].set_ylabel('FEV'); 

    idx_best_step = len(cps_sel)-1
    weight_fold = 'step-%03d' %(step_numbers[cps_sel[idx_best_step]])  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
    weight_fold = os.path.join(ft_path_model_save,weight_fold)
    raw_restored = orbax_checkpointer.restore(weight_fold)
    mdl_state_eval = train_metalmaps.load(ft_mdl,raw_restored['model'],ft_lr)


    params_final = mdl_state_eval.params
    params_orig = pre_mdl_state.params
    # %%
    
    # u=33;plt.plot(y_units[:500,u]);plt.plot(pred_rate_units[:500,u])
    
    
    # %%
    performance_finetuning = {
    'expDate':ft_expDate,
    'ft_mdl_name': ft_mdl_name,
    
    'ft_fev_val_allUnits_allEpochs': np.asarray(fev_allUnits_allEpochs),
    'ft_corr_val_allUnits_allEpochs': np.asarray(predCorr_allUnits_allEpochs),   
    
    'pre_fev_val_allUnits_lastEpoch': np.asarray(pre_fev_val_allUnits_lastEpoch),
    'pre_corr_val_allUnits_lastEpoch': np.asarray(pre_corr_val_allUnits_lastEpoch),   

    'lr_schedule': rgb_lrs,
    'ft_lr': ft_lr,
    'ft_trainingSamps_dur': ft_trainingSamps_dur,   
    }
    
    
    
    meta_info = {
        'pretrained_mdl': path_pretrained,
        'n_rgcs': y_units.shape[-1],
        'ft_idx_best_epoch': idx_best_step
        }
    
    fname_save_performance = os.path.join(ft_path_model_save,'perf_finetuning.pkl')
    
    with open(fname_save_performance, 'wb') as f:       # Save model architecture
        cloudpickle.dump([performance_finetuning,meta_info,params_orig,params_final], f)
    
    
# %%
    print('-----FINISHED FINETUNING-----')
    return performance_finetuning,params_orig,params_final

        
if __name__ == "__main__":
    print('In "Main"')
    args = parser_finetune()
    # Raw print arguments
    print("Arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))       
    run_finetune(**vars(args))    
    
    
    
    
    # %% Parameter changes
"""
    # def compute_relative_changes(original_params, finetuned_params):
    #     param_changes = jax.tree_map(lambda a,b: jnp.linalg.norm(b-a)/jnp.linalg.norm(a), params_orig, params_final)
                
    #     return param_changes
    
    # def compute_relative_changes(original_params, finetuned_params):
        
    #     original_params = jax.tree_map(lambda a: jnp.reshape(a,(-1,a.shape[-1])),original_params)
    #     finetuned_params = jax.tree_map(lambda a: jnp.reshape(a,(-1,a.shape[-1])),finetuned_params)
        
    #     changes = jax.tree_map(lambda a,b: jnp.sum(jnp.abs(a-b)),original_params,finetuned_params)
    #     param_changes = jax.tree_map(lambda a,c: c/jnp.sum(jnp.abs(a)),original_params,changes)
                                       
    #     # param_changes = jax.tree_map(lambda a,b: jnp.linalg.norm(b-a)/jnp.linalg.norm(a), params_orig, params_final)
                
    #     return param_changes
    
    def compute_relative_changes(original_params, finetuned_params):
        
        original_params = jax.tree_map(lambda a: jnp.reshape(a,(-1,a.shape[-1])),original_params)
        finetuned_params = jax.tree_map(lambda a: jnp.reshape(a,(-1,a.shape[-1])),finetuned_params)
        
        param_changes = jax.tree_map(lambda a,b: jnp.sum(jnp.linalg.norm(b-a)/jnp.linalg.norm(a)),original_params,finetuned_params)
        # param_changes = jax.tree_map(lambda a,c: c/jnp.sum(jnp.abs(a)),original_params,changes)
                                       
        # param_changes = jax.tree_map(lambda a,b: jnp.linalg.norm(b-a)/jnp.linalg.norm(a), params_orig, params_final)
                
        return param_changes

    def compute_directional_changes(original_params, finetuned_params):
        
        original_params = jax.tree_map(lambda a: jnp.reshape(a, (-1, a.shape[-1])), original_params)
        finetuned_params = jax.tree_map(lambda a: jnp.reshape(a, (-1, a.shape[-1])), finetuned_params)
        
        def cosine_similarity(a, b):
            delta = b - a
            return jnp.sum(jnp.dot(delta.flatten(), a.flatten())) / (jnp.linalg.norm(delta) * jnp.linalg.norm(a) + 1e-8)
        
        cosine_sims = jax.tree_map(cosine_similarity, original_params, finetuned_params)
        
        return cosine_sims

    
    def get_cpt_mdl(mdl_state,cpt=None):
        if cpt==None:
            allCps = glob.glob(ft_path_model_save+'/step*')
            assert  len(allCps)!=0, 'No checkpoints found'
            step_numbers = np.sort(np.asarray([int(re.search(r'step-(\d+)', s).group(1)) for s in allCps]))
            nb_cps = len(allCps)
            last_cp = step_numbers[nb_cps-1]
        else:
            last_cp = cpt
        
        mdl_state_eval = ft_mdl_state
        weight_fold = 'step-%03d' %(last_cp)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
        weight_fold = os.path.join(ft_path_model_save,weight_fold)
        
        assert os.path.isdir(weight_fold)==True, 'Checkpoint %d does not exist'%step_numbers[cps_sel[i-1]]
        raw_restored = orbax_checkpointer.restore(weight_fold)
        mdl_state_eval = train_metalmaps.load(ft_mdl,raw_restored['model'],ft_lr)
        
        return mdl_state_eval
    
    idx_bestcp = np.argmax(predCorr_medianUnits_allEpochs)
    mdl_state_eval = get_cpt_mdl(ft_mdl_state,step_numbers[idx_bestcp-1])
    params_final = mdl_state_eval.params
    params_orig = pre_mdl_state.params
    
    
    param_changes = compute_relative_changes(params_orig, params_final)
    param_changes_dir = compute_directional_changes(params_orig, params_final)
    
    layer_order = ['Conv_0','LayerNorm_0','TrainableAF_0','Conv_1','LayerNorm_1','TrainableAF_1','Conv_2','LayerNorm_2','TrainableAF_2',
                       'Conv_3','LayerNorm_3','TrainableAF_3',
                       'LayerNorm_4','output','LayerNorm_5','TrainableAF_4']
    
    # %
    def get_flattened_changes(param_changes,layer_order,param_name):
        flattened_changes=[]
        for layer_name in layer_order:
            for param in param_name:
                change = param_changes[layer_name][param]
                rgb = (f'{layer_name}/{param}', change)
                flattened_changes.append(rgb)
        param_names = [change[0] for change in flattened_changes]
        changes = [change[1] for change in flattened_changes]
    
        return param_names,changes
    
    layer_order = ['TrainableAF_0','TrainableAF_1','TrainableAF_2','TrainableAF_3','TrainableAF_4']
    pname_AF_sat,pval_AF_sat = get_flattened_changes(param_changes,layer_order,['sat',])
    pname_AF_gain,pval_AF_gain = get_flattened_changes(param_changes,layer_order,['gain',])
    
    layer_order = ['LayerNorm_0','LayerNorm_1','LayerNorm_2','LayerNorm_3','LayerNorm_4','LayerNorm_5']
    pname_LN_scale,pval_LN_scale = get_flattened_changes(param_changes,layer_order,['scale',])
    pname_LN_bias,pval_LN_bias = get_flattened_changes(param_changes,layer_order,['bias',])
    
    
    layer_order = ['LayerNorm_0','LayerNorm_1','LayerNorm_2','LayerNorm_3','LayerNorm_4','LayerNorm_5']
    param_name = ['bias',]
    
    layer_order = ['Conv_0','Conv_1','Conv_2','Conv_3','output']
    pname_conv_kernel,pval_conv_kernel = get_flattened_changes(param_changes,layer_order,['kernel',])
    pname_conv_bias,pval_conv_bias = get_flattened_changes(param_changes,layer_order,['bias',])
    
    
    fig,axs=plt.subplots(3,2,figsize=(12,10))
    fig.suptitle(ft_expDate)
    axs[0,0].plot(pval_conv_kernel,'-o')
    axs[0,0].set_ylabel('relative change')
    axs[0,0].set_title('Conv kernel')
    axs[0,1].plot(pval_conv_bias,'-o')
    axs[0,1].set_ylabel('relative change')
    axs[0,1].set_title('Conv bias')
    
    
    axs[1,0].plot(pval_LN_scale,'-o')
    axs[1,0].set_ylabel('relative change')
    axs[1,0].set_title('LayerNorm Scale')
    axs[1,1].plot(pval_LN_bias,'-o')
    axs[1,1].set_ylabel('relative change')
    axs[1,1].set_title('LayerNorm Bias')
    
    
    axs[2,0].plot(pval_AF_sat,'-o')
    axs[2,0].set_ylabel('relative change')
    axs[2,0].set_title('ActivationFunction Sat')
    axs[2,1].plot(pval_AF_gain,'-o')
    axs[2,1].set_ylabel('relative change')
    axs[2,1].set_title('ActivationFunction Gain')
    
    
    # %%
    flattened_changes = [(f'{layer_name}/{p}', change) 
                         for layer_name in layer_order
                         for p, change in param_changes[layer_name]]
    
    
    sorted_changes = sorted(flattened_changes, key=lambda x: x[1], reverse=True)
    print(sorted_changes[:10])
    
    param_names = [change[0] for change in sorted_changes]
    param_changes = [change[1] for change in sorted_changes]
    
    # Create a bar chart
    N_top = len(param_names)
    plt.figure(figsize=(10, 6))
    plt.barh(param_names[:N_top], param_changes[:N_top])  # Display top 10 parameters
    plt.xlabel('Relative Change')
    plt.ylabel('Parameter')
    plt.title('Parameter Changes After Fine-tuning')
    plt.gca().invert_yaxis()  # To show the largest changes at the top
    plt.show()
    
    
    
    
    # %% Plot AF model
    
    def TrainableAF(x,sat=0.01,gain=0.95):
        
        a = ((1-sat+1e-6)*jnp.log(1+jnp.exp(gain*x)))/(gain+1e-6)
        b = (sat*(jnp.exp(gain*x)))/(1+jnp.exp(gain*x)+1e-6)
        
        outputs = a+b
        return outputs
    
    x = np.arange(-5,5,0.1)
    
    
    chan=120
    sel_layer = 'TrainableAF_4'
    sat = params_orig[sel_layer]['sat'][chan]
    gain = params_orig[sel_layer]['gain'][chan]
    outputs_orig = TrainableAF(x,sat,gain)
    
    sat = params_final[sel_layer]['sat'][chan]
    gain = params_final[sel_layer]['gain'][chan]
    outputs_final = TrainableAF(x,sat,gain)
    
    
    plt.plot(x,outputs_orig,label='initial');plt.plot(x,outputs_final,label='final');
    plt.legend();plt.title('Layer: %s | Chan: %d'%(sel_layer,chan))
    
"""