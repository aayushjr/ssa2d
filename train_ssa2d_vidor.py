#!/usr/bin/env python
"""
Train the network 
"""

import os
import cv2
import numpy as np
import sys
import time
import os.path
import pdb

from keras.models import model_from_json
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adam   #, SGD, RMSprop, Adadelta
from keras.utils import print_summary

from utils.metrics_util import *
from model_i3d_flow import Inception_Inflated3d_full_cls 
from keras.models import load_model
from custom_metric_losses import *
from model_params import ModelParameters as Params, Classes
from model_params import RunModes


import tensorflow as tf
import keras.backend as K
dim_ordering = K.image_dim_ordering()
print("[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(dim_ordering))
backend = dim_ordering

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), (1,2,3,4))
    #intersection = K.print_tensor(intersection, message="intersection: ")
    union = K.sum(K.square(y_true), (1,2,3,4)) + K.sum(K.square(y_pred), (1,2,3,4)) + K.epsilon()
    #return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    return (2. * intersection) / union

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def cross_entropy_with_dice(y_true, y_pred):
    
    sloss = K.mean(K.categorical_crossentropy(y_true, y_pred), (1,2,3))
    
    # FG dice
    fg_true = y_true[:,:,:,:,:-1]
    fg_pred = y_pred[:,:,:,:,:-1]
    dc_fg = dice_coef(fg_true, fg_pred)
    dcl_fg = 1 - dc_fg
       
    combined_loss = sloss + dcl_fg
    return combined_loss
    

def dice_coef_per_class(y_true, y_pred):
    #fg_true = y_true[:,:,:,:-1]
    #fg_pred = y_pred[:,:,:,:-1]
    
    intersection = K.sum(K.abs(y_true * y_pred), (1,2,3))
    union = K.sum(K.square(y_true), (1,2,3)) + K.sum(K.square(y_pred), (1,2,3)) + K.epsilon()
    dc_fg = (2. * intersection) / union
    dcl_fg = 1 - dc_fg  # shape (b, c)
    dcl_fg = K.expand_dims(dcl_fg, 1)
    dcl_fg = K.expand_dims(dcl_fg, 1)
    dcl_fg = K.expand_dims(dcl_fg, 1)   # shape (b, 1, 1, 1, c)
    return dcl_fg
    
def instance_weighted_cce(num_classes):
    
    def loss(y_true, y_pred):
        pixel_per_class = K.sum(y_true, (1,2,3))
        n_shape = K.int_shape(y_true)
        #y_true = K.print_tensor(y_true, message="y_true: ")
        #n_shape = K.print_tensor(n_shape, message="n_shape: ")
        n_samples = 8*56*56*81 #n_shape[1] * n_shape[2] * n_shape[3]
        weights = n_samples / ((num_classes * pixel_per_class) + 1.)
        weights = K.expand_dims(weights, 1)
        weights = K.expand_dims(weights, 1)
        weights = K.expand_dims(weights, 1) # Bring to shape (b, 1, 1, 1, c)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss


def instance_weighted_bce(num_classes):
    
    def loss(y_true, y_pred):
        n_samples = 8*56*56 #n_shape[1] * n_shape[2] * n_shape[3]
        pixel_per_class_fg = K.sum(y_true, (1,2,3))
        pixel_per_class_bg = n_samples - pixel_per_class_fg
        n_shape = K.int_shape(y_true)
        #y_true = K.print_tensor(y_true, message="y_true: ")
        weights_fg = (n_samples / ((2 * pixel_per_class_fg) + 1.)) + K.epsilon()
        weights_fg = K.expand_dims(weights_fg, 1)
        weights_fg = K.expand_dims(weights_fg, 1)
        weights_fg = K.expand_dims(weights_fg, 1) # Bring to shape (b, 1, 1, 1, c)
        
        weights_bg = (n_samples / ((2 * pixel_per_class_bg) + 1.)) + K.epsilon()
        weights_bg = K.expand_dims(weights_bg, 1)
        weights_bg = K.expand_dims(weights_bg, 1)
        weights_bg = K.expand_dims(weights_bg, 1) # Bring to shape (b, 1, 1, 1, c)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
        loss_fg = y_true * K.log(y_pred) * weights_fg
        loss_bg = (1-y_true) * K.log(1-y_pred) * weights_bg
        loss = (loss_fg + loss_bg)
        loss = -K.mean(loss, -1)
        return loss
    
    return loss
        
       
def train(clip_shape=(8, 450, 800), batch_size=1, val_batch_size=1, n_epoch=10, model='c3d', data_type='vidor', model_dir='./models/', num_steps=3000, lr=1e-4, decay=1e-6, validation_steps = 100):

    n_depth, n_height, n_width = clip_shape

    # Helper: Save the model.

    type_prefix = 'softmax'
    name_prefix = 'SOFTMAX_ADAM_RGB_cce_actor_iwbce_action_50_skip2_FG_ActorwMask_ActionwMask'
    model_name_prefix = '{}-{}-lr_{}_{}Frames_{}_epoch'.format(model, data_type, lr, '16', name_prefix)
    
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('trained', 'checkpoints_vidor_'+type_prefix+'_'+model, model_name_prefix + '-{epoch:03d}.h5'),
        verbose=1,
        save_best_only=False,
        period=1)
        #save_weights_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('trained', 'logs_vidor_'+type_prefix+'_'+model, model+'_'+'lr'+str(lr)+'_'+name_prefix, ))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=16)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('trained', 'logs_vidor_'+type_prefix+'_'+model, model_name_prefix  + str(timestamp) + '.log'))

    if model=='i3d':
      NUM_FRAMES = n_depth
      FRAME_HEIGHT = int(n_height)
      FRAME_WIDTH = int(n_width)
      NUM_RGB_CHANNELS = 3
      NUM_CLASSES = 50
      
      
      load_models_in_loader = False
      
      rgb_model_base = Inception_Inflated3d_full_cls(   
                      include_top=False,
                      weights='rgb_kinetics_only',
                      input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                      classes=NUM_CLASSES)
            
      print_summary(rgb_model_base, line_length=140)
      model_weight_filename = 'models/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
      
      if not load_models_in_loader:
        print("[Info] Loading model weights...")
        print(model_weight_filename)
        rgb_model_base.load_weights(model_weight_filename, by_name=True)
        print("[Info] Loading model weights -- DONE!")
      
      optimizer = Adam(lr=lr, decay=decay, amsgrad=True)
      
      loss={'conv1d_fg': 'categorical_crossentropy', 'conv1d_actor': cross_entropy_with_dice, 'conv1d_cls': instance_weighted_bce(2)}
      
      metrics = {'conv1d_fg': custom_miou_fg, 'conv1d_actor': custom_miou,'conv1d_cls': custom_miou_sig} 
      
      rgb_model_base.compile(loss=loss, optimizer=optimizer, metrics=metrics)
      
      print("="*30)
      print("i3d model loaded")
      print("Save for: {}".format(model_name_prefix))
      print("Loss ", loss)
      print("Metrics ", metrics)
      print("learning rate: {}".format(lr))  
      print("="*30)
      train_model = rgb_model_base

    else:
      print("Model invalid/not implemented yet. Use i3d")
      exit(0)
    
    # Dataloader
    from dataloader_vidor import VidORDataloaderV1
    training_generator = VidORDataloaderV1(mode=RunModes.training, shuffle=True, visualize=False,
                                           per_class_sample=True, focus_key=Classes.act_rel, samples_per_class=100)  
    validation_generator = VidORDataloaderV1(mode=RunModes.validation, shuffle=True, visualize=False,
                                             per_class_sample=True, focus_key=Classes.act_rel, samples_per_class=60)                                       
                                             
    num_steps = int(len(training_generator))
    validation_steps = int(len(validation_generator))
    print("Batch size: ", training_generator._batch_size)
    
    train_model.fit_generator(
            generator=training_generator,
            validation_data=validation_generator,
            steps_per_epoch=num_steps,
            validation_steps=validation_steps,
            callbacks=[tb, csv_logger, checkpointer],
            epochs=n_epoch,
            workers=5,
            use_multiprocessing=True,
            verbose=2)
    

def main():
    """These are the main training settings. Set before running this file."""
    batch_size = 6   # Doesnt affect for vidor, use model_params to set values
    val_batch_size = 6  # Doesnt affect for vidor, use model_params to set values
    n_epoch = 500       # Doesnt affect for vidor, use model_params to set values
    num_steps = int(4000/float(batch_size))   # Doesnt affect for vidor, use model_params to set values
    validation_steps = int(640/float(val_batch_size))   # Doesnt affect for vidor, use model_params to set values 
    lr = 1e-3
    decay = 3e-6
    model = 'i3d' #'c3d'

    clip_shape = (16, 224, 224) 

    train(clip_shape=clip_shape, batch_size=batch_size, val_batch_size=val_batch_size, n_epoch=n_epoch, model = model, num_steps=num_steps, lr=lr, decay=decay, validation_steps = validation_steps)

if __name__ == '__main__':
    main()

