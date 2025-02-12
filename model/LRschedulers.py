#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:31:57 2023

@author: Saad Idrees idrees.sa@gmail.com
         jZ Lab, York University
"""

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow import keras

class printLR_constant(keras.callbacks.Callback):
    def __init__(self,config):
        self.config=config
        
    def on_epoch_end(self, epoch, logs=None):
        lr = tf.keras.backend.eval(self.model.optimizer.learning_rate)
        print("Epoch - {:d} | LR - {:.2E}".format(epoch,lr))
            

@tf.function
def stepLR(step,config):
    initial_lr = config['initial_lr']
    drop = config['drop']
    steps_drop = config['steps_drop']
    lr_new = initial_lr*(drop**tf.floor((1+step)/steps_drop))
    return lr_new

@tf.function
def linear(step,config):
    initial_lr = config['initial_lr']
    decay = config['decay']
    if config['steps_constant'] > step:
        lr_new = initial_lr
    else:
        lr_new = initial_lr*(1/1-(decay*step))
    return lr_new


class CustomLRSchedule(LearningRateSchedule):
    def __init__(self,config):
        self.config = config
        
    def __call__(self,step):
        if self.config['scheduler']=='stepLR':
            print(step)
            lr_new = stepLR(step,self.config)
        

        if self.config['scheduler']=='linearLR':
            lr_new = linearLR(step,self.config)

        return lr_new



def getConfig(lr,scheduler,bz):
    if scheduler=='constant':
        config = dict(
            initial_lr=lr,
            scheduler=scheduler)
    
    elif scheduler=='stepLR':
        config = dict(
            initial_lr=lr,
            scheduler=scheduler,
            drop=0.5,
            steps_drop=1*bz)       # basically epochsxbz = steps

    elif scheduler=='linear':
        config = dict(
            initial_lr=lr,
            scheduler=scheduler,
            steps_constant=10,
            decay=0.001)

    else:
        raise ValueError('Invalid LR scheduler')
        
    return config




"""
# %%
initial_lr = 0.001
config = {}
config['drop'] = 0.5
config['steps_drop'] = 10
config['decay'] = 0.01

scheduler = 'stepLR'
lr_schedule = CustomLRSchedule(initial_lr,config)


optimizer = Adam(lr_schedule)
mdl.compile(optimizer=optimizer,
              loss='poisson',
              metrics=['accuracy'])


X = np.asarray(data_train.X[:3])
y = np.asarray(data_train.y[:3])
mdl.fit(X, y, epochs=100,batch_size = 8,callbacks= [CustomCallback()])



# epochs = np.arange(0,100)
# lr_new = []
# initial_lr = 0.001
# for i in epochs:
#     # rgb = stepLR(i,initial_lr,0.5,10)
#     rgb = linearLR(i,initial_lr,0.001)
#     lr_new.append(rgb)

# plt.plot(lr_new)
    
    
"""