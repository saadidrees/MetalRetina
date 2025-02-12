#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 23:29:28 2021

@author: saad
"""



from model.parser import parser_run_model


def run_model(expFold,mdl_name,path_model_save_base,fname_data_train_val_test,
                            path_existing_mdl='',idxStart_fixedLayers=0,idxEnd_fixedLayers=-1,transfer_mode='finetuning',APPROACH='maml',
                            saveToCSV=1,runOnCluster=0,
                            temporal_width=40, thresh_rr=0,
                            chans_bp=1,
                            chan1_n=8, filt1_size=13, filt1_3rdDim=20,
                            chan2_n=0, filt2_size=0, filt2_3rdDim=0,
                            chan3_n=0, filt3_size=0, filt3_3rdDim=0,
                            chan4_n=0, filt4_size=0, filt4_3rdDim=0,
                            pr_temporal_width = 180,pr_params_name='',
                            nb_epochs=100,bz_ms=10000,trainingSamps_dur=0,validationSamps_dur=0.3,testSamps_dur=0.3,idx_unitsToTake=0,frac_train_units=0.95,
                            BatchNorm=1,BatchNorm_train=0,MaxPool=1,c_trial=1,
                            lr=0.01,lr_fac=1,use_lrscheduler=1,lrscheduler='constant',
                            USE_CHUNKER=0,CONTINUE_TRAINING=1,info='',job_id=0,
                            select_rgctype='',USE_WANDB=0,
                            path_dataset_base='/home/saad/data/analyses/data_kiersten'):

# %% prepare data
    print('In main function body')
# import needed modules
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

    from model.data_handler import prepare_data_cnn2d_maps,isintuple
    from model import data_handler_ej
    from model.data_handler_mike import load_h5Dataset
    from model.performance import model_evaluate_new, estimate_noise
    import model.paramsLogger
    import model.utils_si
    
    from model import models_jax, train_singleretunits, dataloaders,handler_maps
    from model import train_metalmaps
    from torch.utils.data import DataLoader

    Exptdata = namedtuple('Exptdata', ['X', 'y'])
    Exptdata_spikes = namedtuple('Exptdata', ['X', 'y','spikes'])
    
    # %
    devices = jax.devices()
    for device in devices:
        if device.device_kind == 'Gpu':
            print(f"GPU: {device.device_kind}, Name: {device.device_kind}")
        else:
            print(f"Device: {device.device_kind}, Name: {device}")

    if runOnCluster==1:
        USE_WANDB=0
    
    if path_existing_mdl==0 or path_existing_mdl=='0':
        path_existing_mdl=''
        
    if pr_params_name==0 or pr_params_name=='0':
        pr_params_name=''
        
    # if only 1 layer cnn then set all parameters for next layers to 0
    if chan2_n == 0:
        filt2_size = 0
        filt2_3rdDim = 0
        
        chan3_n = 0
        filt3_size = 0
        filt3_3rdDim = 0 
        
    if chan3_n == 0:
        filt3_size = 0
        filt3_3rdDim = 0 
        
    if chan4_n == 0:
        filt4_size = 0
        filt4_3rdDim = 0 
    
    if 'BP' not in mdl_name:
        chans_bp=0
        

    # path to save results to - PARAMETERIZE THIS
    if runOnCluster==1:
        path_save_performance = '/home/sidrees/scratch/RetinaPredictors/performance'
    else:
        path_save_performance = '/home/saad/postdoc_db/projects/RetinaPredictors/performance'
    
    
    if not os.path.exists(path_save_performance):
        os.makedirs(path_save_performance)
          
# %% load train val and test datasets from saved h5 file
    """
        load_h5dataset is a function to load training and validation data from h5 dataset. We can extract all data or a subset using the nsamps arguments.
        data_train, val and test are named tuples. data_train.X contains the stimulus with dimensions [samples,y pixels, x pixels]
        and data_train.y contains the spikerate normalized by median [samples,numOfCells]
    """

    trainingSamps_dur_orig = trainingSamps_dur
    if nb_epochs == 0:  # i.e. if only evaluation has to be run then don't load all training data
        trainingSamps_dur = 4
        
    # Check whether the filename has multiple datasets that need to be merged
    if fname_data_train_val_test.endswith('.txt'):
        with open(fname_data_train_val_test, 'r') as f:
            expDates = f.readlines()
        expDates = [line.strip() for line in expDates]
        
        dataset_suffix = expDates[0]
        expDates = expDates[1:]
        
        fname_data_train_val_test_all = []
        i=5
        for i in range(len(expDates)):
            name_datasetFile = expDates[i]+'_dataset_train_val_test_'+dataset_suffix+'.h5'
            fname_data_train_val_test_all.append(os.path.join(path_dataset_base,'datasets',name_datasetFile))


    else:
        fname_data_train_val_test_all = fname_data_train_val_test.split('+')
    
    # Get the total num of samples and RGCs in each dataset
    nsamps_alldsets = []
    num_rgcs_all = []
    for d in range(len(fname_data_train_val_test_all)):
        with h5py.File(fname_data_train_val_test_all[d]) as f:
            nsamps_alldsets.append(f['data_train']['X'].shape[0])
            num_rgcs_all.append(f['data_train']['y'].shape[1])
    nsamps_alldsets = np.asarray(nsamps_alldsets)
    num_rgcs_all = np.asarray(num_rgcs_all)

    # Load datasets
    idx_train_start = 0    # mins to chop off in the begining.
    psf_params = dict(pixel_neigh=1,method='cross')
    FRAC_U_TRTR = 0.75
    d=0
    dict_train = {};dict_trtr={};dict_trval={}
    dict_val = {}
    dict_test = {}
    unames_allDsets = []
    data_info = {}
    dict_dinf = {}
    nsamps_alldsets_loaded = []
    for d in range(len(fname_data_train_val_test_all)):
        exp = expDates[d]
        print('Loading dataset %d of %d'%(d+1,len(fname_data_train_val_test_all)))
        rgb = load_h5Dataset(fname_data_train_val_test_all[d],nsamps_val=validationSamps_dur,nsamps_train=trainingSamps_dur,nsamps_test=testSamps_dur,  # THIS NEEDS TO BE TIDIED UP
                             idx_train_start=idx_train_start)
        data_train=rgb[0]
        data_val = rgb[1]
        data_test = rgb[2]
        data_quality = rgb[3]
        dataset_rr = rgb[4]
        parameters = rgb[5]
        if len(rgb)>7:
            data_info = rgb[7]
            
        info_unitSplit = data_handler_ej.load_info_unitSplit(fname_data_train_val_test_all[d])
        info_unitSplit['unames_train'] = parameters['unames'][info_unitSplit['idx_train']]
        info_unitSplit['unames_val'] = parameters['unames'][info_unitSplit['idx_val']]
        frac_train_units = parameters['frac_train_units']

    
        t_frame = parameters['t_frame']     # time in ms of one frame/sample 
        # parameters['unames'] = data_quality['uname_selectedUnits']

        data_train,data_val,dinf = handler_maps.arrange_data_formaps(exp,data_train,data_val,parameters,frac_train_units,psf_params=psf_params,info_unitSplit=info_unitSplit)
        dinf['unit_locs_train'] = dinf['unit_locs'][dinf['idx_units_train']]
        dinf['unit_types_train'] = dinf['unit_types'][dinf['idx_units_train']]
        
        dinf['unit_locs_val'] = dinf['unit_locs'][dinf['idx_units_val']]
        dinf['unit_types_val'] = dinf['unit_types'][dinf['idx_units_val']]


        dinf['umaskcoords_trtr'],dinf['umaskcoords_trval'],dinf['umaskcoords_trtr_remap'],dinf['umaskcoords_trval_remap'] = handler_maps.umask_metal_split(dinf['umaskcoords_train'],FRAC_U_TRTR=FRAC_U_TRTR)
        data_trtr,data_trval = handler_maps.prepare_metaldataset(data_train, dinf['umaskcoords_trtr'],dinf['umaskcoords_trval'],bgr=0)


        dict_train[fname_data_train_val_test_all[d]] = data_train
        dict_trtr[fname_data_train_val_test_all[d]] = data_trtr
        dict_trval[fname_data_train_val_test_all[d]] = data_trval

        dict_val[fname_data_train_val_test_all[d]] = data_val
        dict_test[fname_data_train_val_test_all[d]] = data_test
        dict_dinf[fname_data_train_val_test_all[d]] = dinf
        unames_allDsets.append(parameters['unames'])
        nsamps_alldsets_loaded.append(data_train.X.shape[0])
        

    # To make datasets equal length (for vectorization)
    nsamps_alldsets_loaded = np.asarray(nsamps_alldsets_loaded)
    nsamps_max = nsamps_alldsets_loaded.max()
    if nsamps_max>400000:  #388958
        nsamps_max = 400000
    
    cell_types_unique = np.unique(dinf['umaskcoords_train'][:,1])
    
# %%

    # Get unit names
    uname_train = [];uname_val = [];
    c_tr = np.zeros(len(cell_types_unique),dtype='int'); c_val = np.zeros(len(cell_types_unique),dtype='int');
    c_exp_tr=np.zeros((len(expDates),len(cell_types_unique)),dtype='int');c_exp_val=np.zeros((len(expDates),len(cell_types_unique)),dtype='int');

    for d in range(len(fname_data_train_val_test_all)):
        dinf = dict_dinf[fname_data_train_val_test_all[d]]
        rgb = dinf['unames'][dinf['idx_units_train']]
        exp_uname = [expDates[d]+'_'+u for u in rgb ]
        uname_train.append(exp_uname)
        
        for t in range(len(cell_types_unique)):
            c_tr[t]=c_tr[t]+(dinf['unit_types'][dinf['idx_units_train']]==cell_types_unique[t]).sum()
            c_exp_tr[d,t] = (dinf['unit_types'][dinf['idx_units_train']]==cell_types_unique[t]).sum()
        
        rgb = dinf['unames'][dinf['idx_units_val']]
        exp_uname = [expDates[d]+'_'+u for u in rgb ]
        uname_val.append(exp_uname)
        
        for t in range(len(cell_types_unique)):
            c_val[t]=c_val[t]+(dinf['unit_types'][dinf['idx_units_val']]==cell_types_unique[t]).sum()
            c_exp_val[d,t] = (dinf['unit_types'][dinf['idx_units_val']]==cell_types_unique[t]).sum()

    print('Total number of datasets: %d'%len(fname_data_train_val_test_all))
    for t in range(len(cell_types_unique)):
        print('Trainining set | Cell type %d: %d RGCs'%(cell_types_unique[t],c_tr[t]))
        print('Validation set | Cell type %d: %d RGCs'%(cell_types_unique[t],c_val[t]))

    
    # Data will be rolled so that each sample has a temporal width. Like N frames of movie in one sample. The duration of each frame is in t_frame
    # if the model has a photoreceptor layer, then the PR layer has a termporal width of pr_temporal_width, which before convs will be chopped off to temporal width
    # this is done to get rid of boundary effects. pr_temporal_width > temporal width
    if mdl_name[:2] == 'PR':    # in this case the rolling width should be that of PR
        temporal_width_prepData = pr_temporal_width
        temporal_width_eval = pr_temporal_width
        
    else:   # in all other cases its same as temporal width
        temporal_width_prepData = temporal_width
        temporal_width_eval = temporal_width    # termporal width of each sample. Like how many frames of movie in one sample
        pr_temporal_width = 0

    
    modelNames_all = models_jax.model_definitions()    # get all model names
    modelNames_2D = modelNames_all[0]
    modelNames_3D = modelNames_all[1]

    # Expand dataset if needed for vectorization, and roll it for temporal dimension
    d=0
    for d in range(len(fname_data_train_val_test_all)):
        print(fname_data_train_val_test_all[d])
        data_train = dict_train[fname_data_train_val_test_all[d]]
        data_trtr = dict_trtr[fname_data_train_val_test_all[d]]
        data_trval = dict_trval[fname_data_train_val_test_all[d]]

        data_test = dict_test[fname_data_train_val_test_all[d]]
        data_val = dict_val[fname_data_train_val_test_all[d]]
        
        if mdl_name in modelNames_2D:
            data_train = prepare_data_cnn2d_maps(data_train,temporal_width_prepData,MAKE_LISTS=True)     # [samples,temporal_width,rows,columns]
            data_trtr = prepare_data_cnn2d_maps(data_trtr,temporal_width_prepData,MAKE_LISTS=True)     # [samples,temporal_width,rows,columns]
            data_trval = prepare_data_cnn2d_maps(data_trval,temporal_width_prepData,MAKE_LISTS=True)     # [samples,temporal_width,rows,columns]

            data_test = prepare_data_cnn2d_maps(data_test,temporal_width_prepData)
            data_val = prepare_data_cnn2d_maps(data_val,temporal_width_prepData,MAKE_LISTS=True)   
            
            # If a dataset is shorter than the max one, then just repeat it so we can still vectorize everything
            if len(data_train.X)<(nsamps_max-temporal_width_prepData):
                
                data_train = handler_maps.expand_dataset(data_train,nsamps_max,temporal_width_prepData)
                data_trtr = handler_maps.expand_dataset(data_trtr,nsamps_max,temporal_width_prepData)
                data_trval = handler_maps.expand_dataset(data_trval,nsamps_max,temporal_width_prepData)

                
            filt1_3rdDim=0
            filt2_3rdDim=0
            filt3_3rdDim=0
    
            
        else:
            raise ValueError('model not found')
    
        dict_train[fname_data_train_val_test_all[d]] = data_train
        dict_trtr[fname_data_train_val_test_all[d]] = data_trtr
        dict_trval[fname_data_train_val_test_all[d]] = data_trval

        dict_test[fname_data_train_val_test_all[d]] = data_test
        dict_val[fname_data_train_val_test_all[d]] = data_val
   
    # Shuffle just the training dataset
    dict_train = dataloaders.shuffle_dataset(dict_train)    
    
    # data_train = dataset_shuffle(data_train,n_train)

 # %% Prepare dataloaders for MAML Training
        
    
    """
    dataloader_temp = dataloader_train# DataLoader(Retinadatasets_train_s,batch_size=1,collate_fn=dataloaders.jnp_collate,shuffle=False)
    
    t = time.time()
    for batch in dataloader_temp:
        elap = time.time()-t
        print(elap)
        
    """

   # ----  Dataloaders  
    MAX_RGCS = 1000
    assert MAX_RGCS > c_exp_tr.sum(axis=1).max(), 'MAX_RGCS limit lower than maximum RGCs in a dataset'
    # MAX_RGCS=int(c_tr.sum())

    n_tasks = len(fname_data_train_val_test_all)    
   
    Retinadatasets_train=[]; Retinadatasets_val=[];
    
    
    d=0
    for d in range(len(fname_data_train_val_test_all)):
        dset = fname_data_train_val_test_all[d]
       
        rgb = dataloaders.RetinaDatasetTRVALMAPS(dict_trtr[dset].X,dict_trtr[dset].y,dict_trval[dset].y,transform=None)
        Retinadatasets_train.append(rgb)
       
        rgb = dataloaders.RetinaDatasetMAML(dict_val[dset].X,dict_val[dset].y,transform=None)
        Retinadatasets_val.append(rgb)
    
       
       
    batch_size_train = bz_ms
    combined_dataset = dataloaders.CombinedDatasetTRVALMAPS(Retinadatasets_train,num_samples=batch_size_train)
    dataloader_train = DataLoader(combined_dataset,batch_size=1,collate_fn=dataloaders.jnp_collate_MAML,shuffle=False)
    batch = next(iter(dataloader_train));a,b,c=batch
       
    batch_size_val = bz_ms
    combined_dataset = dataloaders.CombinedDataset(Retinadatasets_val,datasets_q=None,num_samples=batch_size_val)
    dataloader_val = DataLoader(combined_dataset,batch_size=1,collate_fn=dataloaders.jnp_collate_MAML,shuffle=False)
    batch = next(iter(dataloader_val));a,b=batch

    maxLen_umaskcoords_tr_subtr = max([len(value['umaskcoords_trtr_remap']) for value in dict_dinf.values()])
    maxLen_umaskcoords_tr_subval = max([len(value['umaskcoords_trval_remap']) for value in dict_dinf.values()])
    maxLen_umaskcoords_val = max([len(value['umaskcoords_val']) for value in dict_dinf.values()])


    # ---- PACK MASKS AND COORDINATES FOR EACH RGC
    b_umaskcoords_trtr=-1*np.ones((n_tasks,maxLen_umaskcoords_tr_subtr,dinf['umaskcoords_trtr_remap'].shape[1]),dtype='int32');
    b_umaskcoords_trval=-1*np.ones((n_tasks,maxLen_umaskcoords_tr_subval,dinf['umaskcoords_trval_remap'].shape[1]),dtype='int32')
    b_umaskcoords_val=-1*np.ones((n_tasks,maxLen_umaskcoords_val,dinf['umaskcoords_val'].shape[1]),dtype='int32')
    
    bool_trtr=np.zeros((n_tasks,maxLen_umaskcoords_tr_subtr),dtype='int32')
    bool_trval=np.zeros((n_tasks,maxLen_umaskcoords_tr_subval),dtype='int32')
    bool_val=np.zeros((n_tasks,maxLen_umaskcoords_val),dtype='int32')
    
    N_trtr = np.zeros((n_tasks),dtype='int')
    N_trval = np.zeros((n_tasks),dtype='int')
    N_val = np.zeros((n_tasks),dtype='int')
    
    maskunits_trtr = np.zeros((n_tasks,MAX_RGCS),dtype='int')
    maskunits_trval = np.zeros((n_tasks,MAX_RGCS),dtype='int')
    maskunits_val = np.zeros((n_tasks,MAX_RGCS),dtype='int')

    
    d=0
    for d in range(len(fname_data_train_val_test_all)):
        rgb = dict_dinf[fname_data_train_val_test_all[d]]['umaskcoords_trtr_remap']
        b_umaskcoords_trtr[d,:len(rgb),:] = rgb
        N_trtr[d] = len(np.unique(rgb[:,0]))
        maskunits_trtr[d,:N_trtr[d]] = 1
        # bool_trtr[d,:len(rgb)] = 1

        
        rgb = dict_dinf[fname_data_train_val_test_all[d]]['umaskcoords_trval_remap']
        b_umaskcoords_trval[d,:len(rgb),:] = rgb
        N_trval[d] = len(np.unique(rgb[:,0]))
        maskunits_trval[d,:N_trval[d]] = 1
        # bool_trval[d,:len(rgb)] = 1



        rgb = dict_dinf[fname_data_train_val_test_all[d]]['umaskcoords_val']
        b_umaskcoords_val[d,:len(rgb),:] = rgb
        N_val[d] = len(np.unique(rgb[:,0]))
        maskunits_val[d,:N_val[d]] = 1
        # bool_val[d,:len(rgb)] = 1 

        segment_size = dict_dinf[fname_data_train_val_test_all[d]]['segment_size']
        
        
    dinf_tr = dict(umaskcoords_trtr=b_umaskcoords_trtr,
                      umaskcoords_trval=b_umaskcoords_trval,
                      N_trtr=N_trtr,
                      N_trval=N_trval,
                      maskunits_trtr=maskunits_trtr,
                      maskunits_trval=maskunits_trval,
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
    for d in range(len(fname_data_train_val_test_all)):
        dset = fname_data_train_val_test_all[d]
        rgb = re.split('_',os.path.split(dset)[-1])[0]
        dset_names.append(rgb)
        nsamps_train = nsamps_train+len(dict_train[dset].X)
    
    run_info = dict(
                    nsamps_train=nsamps_train,
                    nsamps_unique_alldsets=nsamps_alldsets.sum(),
                    nunits_train=c_tr.sum(),
                    nunits_val=c_val.sum(),
                    nexps = n_tasks,
                    exps = expDates
                    )
        
    
    print('Total training data duration: %0.2f mins'%(run_info['nsamps_train']*t_frame/1000/60))



    # ---- Batch size and LRs    
    bz = batch_size_train #math.ceil(bz_ms/t_frame)   # input batch size (bz_ms) is in ms. Convert into samples
    n_batches = len(dataloader_train)#np.ceil(len(data_train.X)/bz)
    
    if lrscheduler == 'exponential_decay':
        lr_schedule = optax.exponential_decay(init_value=lr,transition_steps=n_batches*10,decay_rate=0.75,staircase=True,transition_begin=0)
    
    elif lrscheduler == 'warmup_exponential_decay':
        
        max_lr = lr
        min_lr = 0.00001
        
        n_warmup = 5
        warmup_schedule = optax.linear_schedule(init_value=min_lr,end_value=max_lr,transition_steps=n_batches*n_warmup)
        n_decay = 50
        # decay_schedule = optax.linear_schedule(init_value=max_lr,end_value=min_lr,transition_steps=n_batches*n_decay)
        decay_schedule = optax.exponential_decay(init_value=max_lr,transition_steps=n_batches*10,decay_rate=0.5,staircase=True,transition_begin=0)
        lr_schedule = optax.join_schedules(schedules=[warmup_schedule,decay_schedule],boundaries=[n_batches*n_warmup])

    else:
        lr_schedule = optax.constant_schedule(lr)


    epochs = np.arange(0,nb_epochs)
    epochs_steps = np.arange(0,nb_epochs*n_batches,n_batches)
    rgb_lrs = [lr_schedule(i) for i in epochs_steps]
    rgb_lrs = np.array(rgb_lrs)
    # plt.plot(epochs,rgb_lrs);plt.show()
    print(np.array(rgb_lrs))

# %% Select model 
    """
     There are three ways of selecting/building a model
     1. Continue training an existing model whose training was interrupted
     2. Build a new model
     3. Build a new model but transfer some or all weights (In this case the weight transferring layers should be similar)
    """
    
    
    inp_shape = dict_train[dset].X[0].shape
    out_shape = dict_train[dset].y[0].shape[-1]
    DTYPE = dict_train[dset].X[0].dtype

    fname_model,dict_params = model.utils_si.modelFileName(U=len(expDates),P=pr_temporal_width,T=temporal_width,CB_n=chans_bp,
                                                        C1_n=chan1_n,C1_s=filt1_size,C1_3d=filt1_3rdDim,
                                                        C2_n=chan2_n,C2_s=filt2_size,C2_3d=filt2_3rdDim,
                                                        C3_n=chan3_n,C3_s=filt3_size,C3_3d=filt3_3rdDim,
                                                        C4_n=chan4_n,C4_s=filt4_size,C4_3d=filt4_3rdDim,
                                                        BN=BatchNorm,MP=MaxPool,LR=lr,TR=c_trial,TRSAMPS=trainingSamps_dur_orig)
    
    if pr_params_name!='':
        path_model_save = os.path.join(path_model_save_base,mdl_name,pr_params_name,fname_model)   # the model save directory is the fname_model appened to save path
    else:
        path_model_save = os.path.join(path_model_save_base,mdl_name,fname_model)   # the model save directory is the fname_model appened to save path
    if not os.path.exists(path_model_save):
        os.makedirs(path_model_save)


    if mdl_name[:4] == 'PRFR':
        if pr_params_name=='':
            raise ValueError('Invalid PR model parameters')
        pr_params_fun = getattr(model.prfr_params,pr_params_name)
        pr_params = pr_params_fun()
        dict_params['pr_params'] = pr_params
    
    dict_params['filt_temporal_width'] = temporal_width
    # dict_params['dtype'] = DTYPE
    dict_params['nout'] = len(cell_types_unique)        
    
    
    if CONTINUE_TRAINING==1 or nb_epochs==0:       # if to continue a halted or previous training
        allEpochs = glob.glob(path_model_save+'/epoch*')
        allEpochs.sort()
        if len(allEpochs)!=0:
            lastEpochFile = os.path.split(allEpochs[-1])[-1]
            rgb = re.compile(r'epoch-(\d+)')
            initial_epoch = int(rgb.search(lastEpochFile)[1])
        else:
            initial_epoch = 0

        if initial_epoch == 0:
            initial_epoch = len(glob.glob(path_model_save+'/weights_*'))    # This is for backwards compatibility
    else:
        initial_epoch = 0
        

    if (initial_epoch>0 and initial_epoch < nb_epochs) or nb_epochs==0:     # Load existing model if true
        
        with open(os.path.join(path_model_save,'model_architecture.pkl'), 'rb') as f:
            mdl,config = cloudpickle.load(f)

        fname_latestWeights = os.path.join(path_model_save,'epoch-%03d' % initial_epoch)
        
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        raw_restored = orbax_checkpointer.restore(fname_latestWeights)
        mdl_state = train_singleretunits.load(mdl,raw_restored['model'],lr)
        
        # Also load the dense layer weights
        weights_dense_file = os.path.join(path_model_save,fname_latestWeights,'weights_dense.h5')

        with h5py.File(weights_dense_file,'r') as f:
            kern_all = jnp.array(f['weights_dense_kernel'])
            bias_all = jnp.array(f['weights_dense_bias'])
            
        weights_dense = (kern_all,bias_all)
        
        print('Loaded existing model')

    else:
        # create the model
        model_func = getattr(models_jax,mdl_name)
        mdl = model_func
        mdl_state,mdl,config = model.train_metalmaps.initialize_model(mdl,dict_params,inp_shape,lr,save_model=True,lr_schedule=lr_schedule)
        
        archi_name = 'model_architecture.pkl'
        with open(os.path.join(path_model_save,archi_name), 'wb') as f:       # Save model architecture
            cloudpickle.dump([mdl,config], f)
    
        # Initialize seperate output layers for each task
        specific_layer_names = ('output','LayerNorm_5','TrainableAF_4')
        conv_kern_all = np.empty((n_tasks,*mdl_state.params['output']['kernel'].shape))
        conv_bias_all = np.empty((n_tasks,*mdl_state.params['output']['bias'].shape))
        # ln_scale_all = np.empty((n_tasks,*mdl_state.params['LayerNorm_5']['scale'].shape))
        # ln_bias_all = np.empty((n_tasks,*mdl_state.params['LayerNorm_5']['bias'].shape))
        # af_gain_all = np.empty((n_tasks,*mdl_state.params['TrainableAF_4']['gain'].shape))
        # af_sat_all = np.empty((n_tasks,*mdl_state.params['TrainableAF_4']['sat'].shape))

    
        for i in range(n_tasks):
            conv_kern_all[i]=jnp.array(mdl_state.params['output']['kernel'])
            conv_bias_all[i]=jnp.array(mdl_state.params['output']['bias'])
            # ln_scale_all[i]=jnp.array(mdl_state.params['LayerNorm_5']['scale'])
            # ln_bias_all[i]=jnp.array(mdl_state.params['LayerNorm_5']['bias'])
            # af_gain_all[i]=jnp.array(mdl_state.params['TrainableAF_4']['gain'])
            # af_sat_all[i]=jnp.array(mdl_state.params['TrainableAF_4']['sat'])


    
        # kern_all = jnp.array(kern_all)
        # bias_all = jnp.array(bias_all)
    
        # weights_output = (conv_kern_all,conv_bias_all,ln_scale_all,ln_bias_all,af_gain_all,af_sat_all)
        weights_output = (conv_kern_all,conv_bias_all)

    path_save_model_performance = os.path.join(path_model_save,'performance')
    if not os.path.exists(path_save_model_performance):
        os.makedirs(path_save_model_performance)
                
    
    fname_excel = 'performance_'+fname_model+'.csv'
        
    models_jax.model_summary(mdl,inp_shape,console_kwargs={'width':180})
    
        
# %% Log all params and hyperparams
    
    
    params_txt = dict(expFold=expFold,mdl_name=mdl_name,path_model_save_base=path_model_save_base,fname_data_train_val_test=fname_data_train_val_test_all,
                      path_dataset_base=path_dataset_base,path_existing_mdl=path_existing_mdl,nb_epochs=nb_epochs,bz_ms=bz_ms,runOnCluster=runOnCluster,USE_CHUNKER=USE_CHUNKER,
                      trainingSamps_dur=trainingSamps_dur_orig,validationSamps_dur=validationSamps_dur,CONTINUE_TRAINING=CONTINUE_TRAINING,
                      idxStart_fixedLayers=idxStart_fixedLayers,idxEnd_fixedLayers=idxEnd_fixedLayers,
                      info=info,lr=rgb_lrs,lr_fac=lr_fac,use_lrscheduler=use_lrscheduler,lr_schedule=lr_schedule,batch_size=bz,initial_epoch=initial_epoch)
    for key in dict_params.keys():
        params_txt[key] = dict_params[key]
    


    
    fname_paramsTxt = os.path.join(path_model_save,'model_log.txt')
    if os.path.exists(fname_paramsTxt):
        f_mode = 'a'
        fo = open(fname_paramsTxt,f_mode)
        fo.write('\n\n\n\n\n\n')
        fo.close()
    else:
        f_mode = 'w'
        
    model.paramsLogger.dictToTxt(params_txt,fname_paramsTxt,f_mode='a')

    
# %% Train model MAML
    t_elapsed = 0
    t = time.time()
    approach=APPROACH
    if initial_epoch < nb_epochs:
        print('-----RUNNING MODEL-----')
        
        loss_currEpoch_master,loss_epoch_train,loss_epoch_val,mdl_state,weights_dense,fev_epoch_train,fev_epoch_val = train_metalmaps.train(mdl_state,weights_output,config,\
                                                                                      dataloader_train,dataloader_val,dinf_tr,dinf_val,nb_epochs,path_model_save,save=True,lr_schedule=lr_schedule,\
                                                                                          APPROACH=APPROACH,step_start=initial_epoch+1)
        _ = gc.collect()
            
    t_elapsed = time.time()-t
    print('time elapsed: '+str(t_elapsed)+' seconds')


    # %% Model Evaluation
    
    # Select the testing dataset
    d=2

    for d in np.arange(0,len(fname_data_train_val_test_all)):   
        idx_dset = d
        dinf_batch_val = jax.tree_map(lambda x: x[idx_dset] if isinstance(x, np.ndarray) else x, dinf_val)

        
        n_cells = dinf_val['N_val'][idx_dset]
    
        nb_epochs = np.max([initial_epoch,nb_epochs])   # number of epochs. Update this variable based on the epoch at which training ended
        val_loss_allEpochs = np.empty(nb_epochs)
        val_loss_allEpochs[:] = np.nan
        fev_medianUnits_allEpochs = np.empty(nb_epochs)
        fev_medianUnits_allEpochs[:] = np.nan
        fev_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
        fev_allUnits_allEpochs[:] = np.nan
        fracExVar_medianUnits_allEpochs = np.empty(nb_epochs)
        fracExVar_medianUnits_allEpochs[:] = np.nan
        fracExVar_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
        fracExVar_allUnits_allEpochs[:] = np.nan
        
        predCorr_medianUnits_allEpochs = np.empty(nb_epochs)
        predCorr_medianUnits_allEpochs[:] = np.nan
        predCorr_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
        predCorr_allUnits_allEpochs[:] = np.nan
        rrCorr_medianUnits_allEpochs = np.empty(nb_epochs)
        rrCorr_medianUnits_allEpochs[:] = np.nan
        rrCorr_allUnits_allEpochs = np.zeros((nb_epochs,n_cells))
        rrCorr_allUnits_allEpochs[:] = np.nan
        
        data_val = dict_val[fname_data_train_val_test_all[idx_dset]]
        
        if isintuple(data_val,'y_trials'):
            rgb = np.squeeze(np.asarray(data_val.y_trials))
            obs_noise = 0#estimate_noise(rgb)
            # obs_noise = estimate_noise(data_test.y_trials)
            obs_rate_allStimTrials = np.asarray(data_val.y)
            num_iters = 1
            
        elif 'stim_0' in dataset_rr and dataset_rr['stim_0']['val'][:,:,idx_unitsToTake].shape[0]>1:
            obs_rate_allStimTrials = dataset_rr['stim_0']['val'][:,:,idx_unitsToTake]
            obs_noise = None
            num_iters = 10
        else:
            obs_rate_allStimTrials = np.asarray(data_val.y)
            if 'var_noise' in data_quality:
                obs_noise = data_quality['var_noise'][idx_unitsToTake]
            else:
                obs_noise = 0
            num_iters = 1
        
        if isintuple(data_val,'dset_names'):
            rgb = data_val.dset_names
            idx_natstim = [i for i,n in enumerate(rgb) if re.search(r'NATSTIM',n)]
            idx_cb = [i for i,n in enumerate(rgb) if re.search(r'CB',n)]
            
        # if obs_noise.shape[0] == mask_unitsToTake_all[idx_dset].shape[0]:
        # obs_noise = obs_noise[mask_unitsToTake_all[idx_dset]==1]

        samps_shift = 0 # number of samples to shift the response by. This was to correct some timestamp error in gregs data
      
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    
        RetinaDataset_val = dataloaders.RetinaDataset(data_val.X,data_val.y,transform=None)
        dataloader_val = DataLoader(RetinaDataset_val,batch_size=bz,collate_fn=dataloaders.jnp_collate)
    
        # mdl_state,mdl,config = model.jax.train_singleretunits.initialize_model(mdl,dict_params,inp_shape,lr,save_model=False)
    
    
        print('-----EVALUATING PERFORMANCE-----')
        i=0
        for i in range(0,nb_epochs):
            print('evaluating epoch %d of %d'%(i,nb_epochs))
            # weight_file = 'weights_'+fname_model+'_epoch-%03d.h5' % (i+1)
            weight_fold = 'epoch-%03d' % (i)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
            weight_file = os.path.join(path_model_save,weight_fold)
            weights_dense_file = os.path.join(path_model_save,weight_fold,'weights_output.h5')
    
            if os.path.isdir(weight_file):
                raw_restored = orbax_checkpointer.restore(weight_file)
                mdl_state = train_metalmaps.load(mdl,raw_restored['model'],lr)
                
                with h5py.File(weights_dense_file,'r') as f:
                    weights_kern = jnp.array(f['weights_output_kernel'][idx_dset])
                    weights_bias = jnp.array(f['weights_output_bias'][idx_dset])
                    
                # Restore the correct dense weights for this dataset
                mdl_state.params['output']['kernel'] = weights_kern
                mdl_state.params['output']['bias'] = weights_bias
    
                val_loss,pred_rate,y,pred_rate_units,y_units = train_metalmaps.eval_step(mdl_state,dataloader_val,dinf_batch_val)
                val_loss = np.mean(val_loss)
        
                val_loss_allEpochs[i] = val_loss
                
        
                fev, fracExVar, predCorr, rrCorr = model_evaluate_new(y_units,pred_rate_units,temporal_width_eval,lag=int(samps_shift),obs_noise=obs_noise)
                        
                fev_allUnits_allEpochs[i,:] = fev
                fev_medianUnits_allEpochs[i] = np.nanmedian(fev)      
                fracExVar_allUnits_allEpochs[i,:] = fracExVar
                fracExVar_medianUnits_allEpochs[i] = np.nanmedian(fracExVar)
                
                predCorr_allUnits_allEpochs[i,:] = predCorr
                predCorr_medianUnits_allEpochs[i] = np.nanmedian(predCorr)
                rrCorr_allUnits_allEpochs[i,:] = rrCorr
                rrCorr_medianUnits_allEpochs[i] = np.nanmedian(rrCorr)
                
                _ = gc.collect()
        
        fig,axs = plt.subplots(1,1,figsize=(7,5)); axs.plot(predCorr_medianUnits_allEpochs)
        axs.set_xlabel('Epochs');axs.set_ylabel('Corr'); fig.suptitle(dset_names[idx_dset] + ' | '+str(dict_params['nout'])+' RGCs')
        
        fname_fig = os.path.join(path_model_save,'fev_val_%s.png'%dset_names[idx_dset])
        fig.savefig(fname_fig)
        
        
        idx_bestEpoch = nb_epochs-1#np.nanargmax(fev_medianUnits_allEpochs)
        # idx_bestEpoch = np.nanargmax(fev_medianUnits_allEpochs)
        fev_medianUnits_bestEpoch = np.round(fev_medianUnits_allEpochs[idx_bestEpoch],2)
        fev_allUnits_bestEpoch = fev_allUnits_allEpochs[(idx_bestEpoch),:]
        fracExVar_medianUnits = np.round(fracExVar_medianUnits_allEpochs[idx_bestEpoch],2)
        fracExVar_allUnits = fracExVar_allUnits_allEpochs[(idx_bestEpoch),:]
        
        predCorr_medianUnits_bestEpoch = np.round(predCorr_medianUnits_allEpochs[idx_bestEpoch],2)
        predCorr_allUnits_bestEpoch = predCorr_allUnits_allEpochs[(idx_bestEpoch),:]
        rrCorr_medianUnits = np.round(rrCorr_medianUnits_allEpochs[idx_bestEpoch],2)
        rrCorr_allUnits = rrCorr_allUnits_allEpochs[(idx_bestEpoch),:]
    
        
        # Load the best weights to save stuff
        weight_fold = 'epoch-%03d' % (idx_bestEpoch)  # 'file_name_{}_{:.03f}.png'.format(f_nm, val)
        weight_file = os.path.join(path_model_save,weight_fold)
        weights_dense_file = os.path.join(path_model_save,weight_fold,'weights_dense.h5')
    
        raw_restored = orbax_checkpointer.restore(weight_file)
        mdl_state = train_metalmaps.load(mdl,raw_restored['model'],lr)
        
        with h5py.File(weights_dense_file,'r') as f:
            weights_kern = jnp.array(f['weights_output_kernel'][idx_dset])
            weights_bias = jnp.array(f['weights_output_bias'][idx_dset])
            
        # Restore the correct dense weights for this dataset
        mdl_state.params['output']['kernel'] = weights_kern
        mdl_state.params['output']['bias'] = weights_bias
    
        
        val_loss,pred_rate,y,pred_rate_units,y_units = train_metalmaps.eval_step(mdl_state,dataloader_val,dinf_batch_val)
        fname_bestWeight = np.array(weight_file,dtype='bytes')
        fev_val, fracExVar_val, predCorr_val, rrCorr_val = model_evaluate_new(y_units,pred_rate_units,temporal_width_eval,lag=int(samps_shift),obs_noise=obs_noise)
    
        # if len(idx_natstim)>0:
        #     fev_val_natstim, _, predCorr_val_natstim, _ = model_evaluate_new(obs_rate_allStimTrials[idx_natstim],pred_rate[idx_natstim],temporal_width_eval,lag=int(samps_shift),obs_noise=obs_noise)
        #     fev_val_cb, _, predCorr_val_cb, _ = model_evaluate_new(obs_rate_allStimTrials[idx_cb],pred_rate[idx_cb],temporal_width_eval,lag=int(samps_shift),obs_noise=obs_noise)
        #     print('FEV_NATSTIM = %0.2f' %(np.nanmean(fev_val_natstim)*100))
        #     print('FEV_CB = %0.2f' %(np.nanmean(fev_val_cb)*100))

# %% Test
#     # offset = 3000
#     x_train = np.array(data_train.X[-2000:])
#     y_train = np.array(data_train.y[-2000:])
#     # x_val = np.array(data_val.X[-2000:])
#     # y_val = np.array(data_val.y[-2000:])
#     x_test = np.array(data_test.X[-2000:])
#     y_test = np.array(data_test.y[-2000:])

    
# # val_loss,pred_rate,y = train_singleretunits.eval_step(mdl_state,(x_train,y_train))
#     _,pred_train,_ = maml.eval_step(mdl_state,(x_train,y_train),mask_unitsToTake_all[idx_dset])
#     pred_train = pred_train[:,mask_unitsToTake_all[idx_dset]==1]
#     # _,pred_val,_ = train_singleretunits.eval_step(mdl_state,(x_val,y_val))
#     _,pred_test,_ = maml.eval_step(mdl_state,(x_test,y_test),mask_unitsToTake_all[idx_dset])
#     pred_test = pred_test[:,mask_unitsToTake_all[idx_dset]==1]
    
#     # for i in range(100):
#     u = 75  #33# 110 #75
    
#     fig,axs =plt.subplots(2,1,figsize=(20,5))
#     axs=np.ravel(axs)
#     axs[0].plot(y_train[-2000:,u])
#     axs[0].plot(pred_train[-2000:,u])
#     axs[0].set_title(str(u))
#     axs[1].plot(y_test[:2000,u])
#     axs[1].plot(pred_test[:2000,u])
#     axs[1].set_title('Validation')
#     plt.show()

    
# %% Save performance
    # data_test=data_val
    if 't_elapsed' not in locals():
        t_elapsed = np.nan
        

    print('-----SAVING PERFORMANCE STUFF TO H5-----')
    
    
    model_performance = {
        'dset_names':dset_names,
        'loss_currEpoch_master':loss_currEpoch_master,
        'loss_epoch_train':loss_epoch_train,
        'loss_epoch_val':loss_epoch_val,
        'fev_epoch_train':fev_epoch_train,
        'fev_epoch_val':fev_epoch_val,
        
        'idx_dset_eval':idx_dset,
        'dset_name_eval':dset_names[idx_dset],
            
        'fev_medianUnits_allEpochs': fev_medianUnits_allEpochs,
        'fev_allUnits_allEpochs': fev_allUnits_allEpochs,
        'fev_medianUnits_bestEpoch': fev_medianUnits_bestEpoch,
        'fev_allUnits_bestEpoch': fev_allUnits_bestEpoch,
        
        'fracExVar_medianUnits': fracExVar_medianUnits,
        'fracExVar_allUnits': fracExVar_allUnits,
        
        'predCorr_medianUnits_allEpochs': predCorr_medianUnits_allEpochs,
        'predCorr_allUnits_allEpochs': predCorr_allUnits_allEpochs,
        'predCorr_medianUnits_bestEpoch': predCorr_medianUnits_bestEpoch,
        'predCorr_allUnits_bestEpoch': predCorr_allUnits_bestEpoch,
        
        'rrCorr_medianUnits': rrCorr_medianUnits,
        'rrCorr_allUnits': rrCorr_allUnits,          
        
        'fname_bestWeight': np.atleast_1d(fname_bestWeight),
        'idx_bestEpoch': idx_bestEpoch,
        
        'val_loss_allEpochs': val_loss_allEpochs,
        't_elapsed': np.array(t_elapsed),
        # 'val_dataset_name': dataset_rr['stim_0']['dataset_name'],
        }
        

    metaInfo = {
       'mdl_name': mdl_name,
       'existing_mdl': np.array(path_existing_mdl,dtype='bytes'),
       'path_model_save': path_model_save,
       'thresh_rr': thresh_rr,
       'trial_num': c_trial,
       'Date': np.array(datetime.datetime.now(),dtype='bytes'),
       'info': np.array(info,dtype='bytes')
       }
        
    model_params = {
                'chan1_n' : chan1_n,
                'filt1_size' : filt1_size,
                'filt1_3rdDim': filt1_3rdDim,
                'chan2_n' : chan2_n,
                'filt2_size' : filt2_size,
                'filt2_3rdDim': filt2_3rdDim,
                'chan3_n' : chan3_n,
                'filt3_size' : filt3_size,
                'filt3_3rdDim': filt3_3rdDim,   
                'chan4_n' : chan4_n,
                'filt4_size' : filt4_size,
                'filt4_3rdDim': filt4_3rdDim,            
                'batch_size' : batch_size_train,
                'nb_epochs' : nb_epochs,
                'BatchNorm': BatchNorm,
                'MaxPool': MaxPool,
                'pr_temporal_width': pr_temporal_width,
                'lr': lr,
                'lr_schedule':lr_schedule
                }
    
    stim_info = {
         'fname_data_train_val_test':fname_data_train_val_test_all,
          'n_valSamps': -1,
          'temporal_width':temporal_width,
         'pr_temporal_width': pr_temporal_width
         }
    if len(data_info)>0:
        for k in data_info:
            stim_info[k] = data_info[k]
    

    fname_save_performance = os.path.join(path_save_model_performance,(expFold+'_'+fname_model+'.pkl'))

    with open(fname_save_performance, 'wb') as f:       # Save model architecture
        cloudpickle.dump([fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dinf_tr,dinf_val,dict_dinf,run_info], f)

    dataset_rr = None
    # save_modelPerformance(fname_save_performance,fname_model,metaInfo,data_quality,model_performance,model_params,stim_info,dataset_rr,datasets_val,dataset_pred)   # It would really help to have a universal h5 writing function

    print('FEV = %0.2f' %(np.nanmax(model_performance['fev_medianUnits_allEpochs'])*100))

    
# %% Write performance to csv file
    print('-----WRITING TO CSV FILE-----')
    if saveToCSV==1:
        name_dataset = os.path.split(fname_data_train_val_test)
        name_dataset = name_dataset[-1]
        csv_header = ['mdl_name','fname_mdl','expFold','idxStart_fixedLayers','idxEnd_fixedLayers','dataset','RGC_types','thresh_rr','RR','temp_window','pr_temporal_width','pr_params_name','batch_size','epochs','chan1_n','filt1_size','filt1_3rdDim','chan2_n','filt2_size','filt2_3rdDim','chan3_n','filt3_size','filt3_3rdDim','chan4_n','filt4_size','filt4_3rdDim','BatchNorm','MaxPool','c_trial','FEV_median','predCorr_median','rrCorr_median','TRSAMPS','t_elapsed','job_id']
        csv_data = [mdl_name,fname_model,expFold,idxStart_fixedLayers,idxEnd_fixedLayers,name_dataset,select_rgctype,thresh_rr,fracExVar_medianUnits,temporal_width,pr_temporal_width,pr_params_name,bz,nb_epochs,chan1_n, filt1_size, filt1_3rdDim, chan2_n, filt2_size, filt2_3rdDim, chan3_n, filt3_size, filt3_3rdDim,chan4_n, filt4_size, filt4_3rdDim,int(BatchNorm),MaxPool,c_trial,fev_medianUnits_bestEpoch,predCorr_medianUnits_bestEpoch,rrCorr_medianUnits,trainingSamps_dur_orig,t_elapsed,job_id]
        
        fname_csv_file = 'performance_'+expFold+'.csv'
        fname_csv_file = os.path.join(path_save_performance,fname_csv_file)
        if not os.path.exists(fname_csv_file):
            with open(fname_csv_file,'w',encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile) 
                csvwriter.writerow(csv_header) 
                
        with open(fname_csv_file,'a',encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(csv_data) 

    fname_validation_excel = os.path.join(path_save_model_performance,expFold+'_validation_'+fname_model+'.csv')
    csv_header = ['epoch','val_fev']
    with open(fname_validation_excel,'w',encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(csv_header) 
        
        for i in range(fev_medianUnits_allEpochs.shape[0]):
            csvwriter.writerow([str(i),str(np.round(fev_medianUnits_allEpochs[i],2))]) 
        
        
    print('-----FINISHED-----')
    return model_performance, mdl

        
if __name__ == "__main__":
    print('In "Main"')
    args = parser_run_model()
    # Raw print arguments
    print("Arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))       
    run_model(**vars(args))


# %% Recycle
"""
    # % Prepare dataloaders for MAML Training
    
    n_tasks = len(fname_data_train_val_test_all)    
    frac_queries = 0.5 # percent
    
    data_train_s,data_train_q = dataloaders.support_query_sets(dict_train,frac_queries)
    
    Retinadatasets_train_s = []
    Retinadatasets_train_q = []

    d=0
    for d in range(len(fname_data_train_val_test_all)):
        dset = fname_data_train_val_test_all[d]
        
        rgb = dataloaders.RetinaDatasetMAML(data_train_s[dset].X,data_train_s[dset].y,transform=None)
        Retinadatasets_train_s.append(rgb)
        
        rgb = dataloaders.RetinaDatasetMAML(data_train_q[dset].X,data_train_q[dset].y,transform=None)
        Retinadatasets_train_q.append(rgb)

    
    batch_size_train = 256
    combined_dataset = dataloaders.CombinedDataset(Retinadatasets_train_s,Retinadatasets_train_q,num_samples=batch_size_train)
    dataloader_train = DataLoader(combined_dataset,batch_size=1,collate_fn=dataloaders.jnp_collate_MAML,shuffle=False)
    batch = next(iter(dataloader_train));a,b,c,d=batch
    
    
    # Validation
    data_val_s,_ = dataloaders.support_query_sets(dict_val,frac_queries=0.0001)

    Retinadatasets_val = []

    d=0
    for d in range(len(fname_data_train_val_test_all)):
        dset = fname_data_train_val_test_all[d]
        
        rgb = dataloaders.RetinaDatasetMAML(data_val_s[dset].X,data_val_s[dset].y,transform='jax')
        Retinadatasets_val.append(rgb)
        
    batch_size_val = 256
    combined_dataset = dataloaders.CombinedDataset(Retinadatasets_val,None,num_samples=batch_size_val)
    dataloader_val = DataLoader(combined_dataset,batch_size=1,collate_fn=dataloaders.jnp_collate_MAML,shuffle=False)
    batch = next(iter(dataloader_val));a,b=batch
    



# %% FineTune - ALL

    ft_dset_name = os.path.split(ft_fname_data_train_val_test)[-1]
    ft_dset_name = re.split('_',ft_dset_name)[0]
    
    raw_restored = orbax_checkpointer.restore(weight_file)
    mdl_state = maml.load(mdl,raw_restored['model'],lr)
    
    
    # Arrange the data
    ft_dict_train_shuffled = dataloaders.shuffle_dataset(ft_dict_train)    
    ft_data_train = ft_dict_train_shuffled[ft_fname_data_train_val_test_all]
    ft_n_units = ft_data_train.y[0].shape[-1]
    # ft_samps = 1000
    # X = ft_data_train.X[:ft_samps]
    # y = ft_data_train.y[:ft_samps]
    # ft_data_train = Exptdata(X,y)
    
    ft_data_test = ft_dict_test[ft_fname_data_train_val_test_all]
    ft_data_val = ft_dict_val[ft_fname_data_train_val_test_all]
    
    batch_size = 256   #1280 1536 1792 2048

    RetinaDataset_test = dataloaders.RetinaDataset(ft_data_test.X,ft_data_test.y,transform=None)
    dataloader_test = DataLoader(RetinaDataset_test,batch_size=batch_size,collate_fn=dataloaders.jnp_collate,shuffle=False)
    
    RetinaDataset_val_val = dataloaders.RetinaDataset(ft_data_val.X,ft_data_val.y,transform=None)
    dataloader_val_val = DataLoader(RetinaDataset_val_val,batch_size=batch_size,collate_fn=dataloaders.jnp_collate,shuffle=False)

    ft_nb_epochs = 40
    n_batches = np.ceil(len(ft_data_train.X)/batch_size).astype('int')
    
    max_lr = 0.1
    min_lr = 0.001
    
    n_warmup = 2
    warmup_schedule = optax.linear_schedule(init_value=0,end_value=max_lr,transition_steps=n_batches*n_warmup)
    n_const = 5
    constant_schedule = optax.constant_schedule(value=max_lr)
    n_decay = 40-n_warmup
    # decay_schedule = optax.cosine_decay_schedule(init_value=max_lr,decay_steps=n_batches*n_decay,alpha=min_lr/max_lr)
    decay_schedule = optax.exponential_decay(init_value=max_lr,transition_steps=n_batches,decay_rate=0.3,staircase=False,transition_begin=1)
    # decay_schedule = optax.linear_schedule(init_value=max_lr,end_value=min_lr,transition_steps=n_batches*n_decay)
    ft_lr_schedule_train = optax.join_schedules(schedules=[warmup_schedule,decay_schedule],boundaries=[n_batches*n_warmup])
    # ft_lr_schedule = optax.join_schedules(schedules=[warmup_schedule,constant_schedule,decay_schedule],boundaries=[n_batches*n_warmup,n_batches*n_const])

    # ft_lr_schedule = optax.cosine_decay_schedule(init_value=max_lr,decay_steps=n_batches*n_decay,alpha=min_lr/max_lr)
    # ft_lr_schedule = optax.exponential_decay(init_value=max_lr,transition_steps=n_batches*3,decay_rate=0.5,staircase=True,transition_begin=1)

    ft_lr_fixed = 0.001
    
    epochs = np.arange(0,ft_nb_epochs)
    epochs_steps = np.arange(0,ft_nb_epochs*n_batches,n_batches)
    rgb_lrs = [ft_lr_schedule_train(i) for i in epochs_steps]
    plt.plot(epochs,rgb_lrs);plt.show()

    layers_finetune = ('Dense_0','LayerNorm_4','LayerNorm_IN') #
    ft_params_fixed,ft_params_trainable = maml.split_dict(mdl_state.params,layers_finetune)

    
    dict_params['nout'] = ft_n_units        # CREATE THE MODEL BASED ON THE SPECS OF THE FIRST DATASET
    # model_func = getattr(models_jax,mdl_name)
    model_func = getattr(models_jax,'CNN2D_FT')
    ft_mdl = model_func
    ft_mdl_state,ft_mdl,ft_config = maml.initialize_model(ft_mdl,dict_params,inp_shape,lr,save_model=True,lr_schedule=lr_schedule)
    models_jax.model_summary(ft_mdl,inp_shape,console_kwargs={'width':180})

    
    # Initialize new dense layer weights
    key = jax.random.PRNGKey(1)

    ft_kern_init = jax.random.normal(key, shape= (mdl_state.params['Dense_0']['kernel'].shape[0],ft_n_units))
    ft_bias_init = jnp.zeros((ft_n_units))


    ft_params_trainable['Dense_0']['kernel'] = ft_kern_init
    ft_params_trainable['Dense_0']['bias'] = ft_bias_init
    
    ft_params_trainable['TrainableAF_0'] = ft_mdl_state.params['TrainableAF_0']
    ft_params_trainable['LayerNorm_IN'] = ft_mdl_state.params['LayerNorm_IN']
    ft_params_trainable['LayerNorm_4'] = ft_mdl_state.params['LayerNorm_4']
    ft_params_trainable['LayerNorm_5'] = ft_mdl_state.params['LayerNorm_5']


    param_labels = {}
    for p in ft_params_fixed.keys():
        param_labels[p] = 'Fixed'
            
    for p in ft_params_trainable.keys():
        param_labels[p] = 'Trainable'

    optimizers = {
                    "Trainable": optax.adam(ft_lr_schedule_train),
                    "Fixed": optax.adam(ft_lr_fixed)
                    }


    # optimizer = optax.adam(learning_rate=ft_lr_schedule) #,weight_decay=1e-4)
    optimizer = optax.multi_transform(optimizers, param_labels)

    ft_mdl_state = maml.TrainState.create(
                apply_fn=ft_mdl.apply,
                params={**ft_params_trainable,**ft_params_fixed},
                tx=optimizer)
    
    
    ft_path_model_save = os.path.join(path_model_save,'finetuning_%s'%ft_dset_name)
    if not os.path.exists(ft_path_model_save):
        os.makedirs(ft_path_model_save)

    ft_loss_epoch_train,ft_loss_epoch_val,ft_mdl_state,fev_epoch_train,fev_epoch_val,lr_epoch,lr_step = maml.ft_train(
        ft_mdl_state,ft_params_fixed,config,ft_data_train,ft_data_val,batch_size,ft_nb_epochs,ft_path_model_save,save=True,ft_lr_schedule=ft_lr_schedule)

    ft_val_loss,pred_rate_val,y_val = maml.ft_eval_step(ft_mdl_state,ft_params_fixed,dataloader_val_val)
    fev_val, fracExVar_val, predCorr_val, rrCorr_val = model_evaluate_new(y_val,pred_rate_val,temporal_width_eval,lag=int(0),obs_noise=0)

    ft_test_loss,pred_rate_test,y_test = maml.ft_eval_step(ft_mdl_state,ft_params_fixed,dataloader_test)
    fev_test, fracExVar_val, predCorr_test, rrCorr_test = model_evaluate_new(y_test,pred_rate_test,temporal_width_eval,lag=int(0),obs_noise=0)

    plt.plot(fev_epoch_val)        
  
"""