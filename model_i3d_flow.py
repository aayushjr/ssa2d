"""Inception-v1 Inflated 3D ConvNet used for Kinetics CVPR paper.
 
The model is introduced in:
 
Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
Joao Carreira, Andrew Zisserman
https://arxiv.org/abs/1705.07750v1
"""
        
from __future__ import print_function
from __future__ import absolute_import

import warnings

import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv3D
from keras.layers import Deconv3D
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers import SpatialDropout3D
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Permute
from keras.layers import Lambda
from keras.layers import UpSampling3D
from keras.layers import GlobalAveragePooling3D
from keras.layers import concatenate, Add, Multiply, Dot
from keras.layers.convolutional import Deconvolution3D
from keras.layers.core import Flatten

from keras import regularizers

from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K

import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras.utils import print_summary
import os
import sys

WEIGHTS_NAME = ['rgb_kinetics_only', 'flow_kinetics_only', 'rgb_imagenet_and_kinetics', 'flow_imagenet_and_kinetics']

# path to pretrained models with top (classification layer)
WEIGHTS_PATH = {
    'rgb_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'flow_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'rgb_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5',
    'flow_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
}

# path to pretrained models with no top (no classification layer)
WEIGHTS_PATH_NO_TOP = {
    'rgb_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'rgb_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
}


def _obtain_input_shape(input_shape,
                        default_frame_size,
                        min_frame_size,
                        default_num_frames,
                        min_num_frames,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate the model's input shape.
    (Adapted from `keras/applications/imagenet_utils.py`)

    # Arguments
        input_shape: either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_frame_size: default input frames(images) width/height for the model.
        min_frame_size: minimum input frames(images) width/height accepted by the model.
        default_num_frames: default input number of frames(images) for the model.
        min_num_frames: minimum input number of frames accepted by the model.
        data_format: image data format to use.
        require_flatten: whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: one of `None` (random initialization)
            or 'kinetics_only' (pre-training on Kinetics dataset).
            or 'imagenet_and_kinetics' (pre-training on ImageNet and Kinetics datasets).
            If weights='kinetics_only' or weights=='imagenet_and_kinetics' then
            input channels must be equal to 3.

    # Returns
        An integer shape tuple (may include None entries).

    # Raises
        ValueError: in case of invalid argument values.
    """
    if weights != 'kinetics_only' and weights != 'imagenet_and_kinetics' and input_shape and len(input_shape) == 4:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_num_frames, default_frame_size, default_frame_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_num_frames, default_frame_size, default_frame_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_num_frames, default_frame_size, default_frame_size)
        else:
            default_shape = (default_num_frames, default_frame_size, default_frame_size, 3)
    if (weights == 'kinetics_only' or weights == 'imagenet_and_kinetics') and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape

    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 4:
                    raise ValueError(
                        '`input_shape` must be a tuple of four integers.')
                if input_shape[0] != 3 and (weights == 'kinetics_only' or weights == 'imagenet_and_kinetics'):
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if input_shape[1] is not None and input_shape[1] < min_num_frames:
                    raise ValueError('Input number of frames must be at least ' +
                                     str(min_num_frames) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if ((input_shape[2] is not None and input_shape[2] < min_frame_size) or
                   (input_shape[3] is not None and input_shape[3] < min_frame_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_frame_size) + 'x' + str(min_frame_size) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 4:
                    raise ValueError(
                        '`input_shape` must be a tuple of four integers.')
                if input_shape[-1] != 3 and (weights == 'kinetics_only' or weights == 'imagenet_and_kinetics'):
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if input_shape[0] is not None and input_shape[0] < min_num_frames:
                    raise ValueError('Input number of frames must be at least ' +
                                     str(min_num_frames) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if ((input_shape[1] is not None and input_shape[1] < min_frame_size) or
                   (input_shape[2] is not None and input_shape[2] < min_frame_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_frame_size) + 'x' + str(min_frame_size) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None, None)
            else:
                input_shape = (None, None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape

def conv3d_bn(x,
              filters,
              num_frames,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1, 1),
              use_bias = False,
              kernel_regularizer = regularizers.l2(0.0001),
              use_activation_fn = True,
              activation_type = 'relu',
              use_bn = True,
              name=None):
    """Utility function to apply conv3d + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv3D`.
        num_frames: frames (time depth) of the convolution kernel.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv3D`.
        strides: strides in `Conv3D`.
        use_bias: use bias or not  
        use_activation_fn: use an activation function or not.
        use_bn: use batch normalization or not.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv3D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv3D(
        filters, (num_frames, num_row, num_col),
        strides=strides,
        padding=padding,
        kernel_regularizer=kernel_regularizer,
        use_bias=use_bias,
        name=conv_name)(x)

    if use_bn:
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 4
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    if use_activation_fn:
        if activation_type == 'leakyrelu':
          x = LeakyReLU(alpha=0.3, name=name)(x)
        else:
          x = Activation(activation_type, name=name)(x)

    return x
  
def adjust(x, depth_factor, height_factor, width_factor, reps):
    x = K.resize_volumes(x, depth_factor, height_factor, width_factor, 'channels_last')
    x = K.repeat_elements(x, rep=reps, axis=-1)
    
    return x

def Inception_Inflated3d_full_cls(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                dropout_prob=0.0,
                endpoint_logit=True,
                classes=400,
                name='i3d_inception_cd3d_flow'):
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 4
    
    lbl_depth = 8
    conv_stride = 1
    actor_classes = 81 # 7 for A2D
    b_size = 56 #144
    #mask_input = Input(shape=(4,14,14,832))
    mask_input_cls = Input(shape=(lbl_depth,b_size,b_size,classes))
    mask_input_cls_add = Input(shape=(lbl_depth,b_size,b_size,classes))
    
    mask_input_actor = Input(shape=(lbl_depth,b_size,b_size,actor_classes))
    mask_input_actor_add = Input(shape=(lbl_depth,b_size,b_size,actor_classes))
    obj_features = Input(shape=(1,b_size,b_size,64))
    
    kernel_regularizer = None #regularizers.l2(0)
        
    #with tf.device('/gpu:0'):                       # FOR MULTI GPU
    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(img_input, 64, 7, 7, 7, strides=(2, 2, 2), kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_1a_7x7')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_2a_3x3')(x)
    x = conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_2b_1x1')
    x_2c = conv3d_bn(x, 192, 3, 3, 3, strides=(1, 1, 1), kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_2c_3x3')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_3a_3x3')(x_2c)

    # Mixed 3b
    branch_0 = conv3d_bn(x, 64, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_3b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_3b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 128, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_3b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_3b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 32, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_3b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 32, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_3b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3b')

    # Mixed 3c
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_3c_0a_1x1')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_3c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 192, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_3c_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_3c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 96, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_3c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_3c_3b_1x1')

    x_3c = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3c')


    # Downsampling (spatial and temporal)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_4a_3x3')(x_3c)

    # Mixed 4b
    branch_0 = conv3d_bn(x, 192, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 208, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 48, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4b')

    # Mixed 4c
    branch_0 = conv3d_bn(x, 160, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4c_0a_1x1')

    branch_1 = conv3d_bn(x, 112, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 224, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4c_1b_3x3')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4c_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4c')

    # Mixed 4d
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4d_0a_1x1')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4d_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 256, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4d_1b_3x3')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4d_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4d_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4d_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4d_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4d')

    # Mixed 4e
    branch_0 = conv3d_bn(x, 112, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4e_0a_1x1')

    branch_1 = conv3d_bn(x, 144, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4e_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 288, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4e_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4e_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4e_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4e_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4e_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4e')

    # Mixed 4f
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4f_0a_1x1')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4f_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4f_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4f_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4f_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4f_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_4f_3b_1x1')

    x_4f = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4f')


    # Downsampling (spatial and temporal)
    #x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='MaxPool2d_5a_2x2')(x_4f)

    # Mixed 5b
    branch_0 = conv3d_bn(x_4f, 256, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_5b_0a_1x1')

    branch_1 = conv3d_bn(x_4f, 160, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_5b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_5b_1b_3x3')

    branch_2 = conv3d_bn(x_4f, 32, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_5b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_5b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5b_3a_3x3')(x_4f)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', name='Conv3d_5b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5b')

    # =========
    # Mask attn
    x_red = conv3d_bn(x, 480, 1, 1 ,1, kernel_regularizer=kernel_regularizer, padding='same', name='conv3d_reduced_5b')
    attntheta = Conv3D(480, (1, 1, 1), kernel_regularizer=kernel_regularizer, padding='same', name='selfattn_m1_theta')(x)
    attntheta = BatchNormalization(name='selfattn_m1_theta_bn')(attntheta)
    attnphi = Conv3D(480, (1, 1, 1), kernel_regularizer=kernel_regularizer, padding='same', name='selfattn_m1_phi')(x)
    attnphi = BatchNormalization(name='selfattn_m1_phi_bn')(attnphi)
    attngaus = Conv3D(480, (1, 1, 1), kernel_regularizer=kernel_regularizer, padding='same', name='selfattn_m1_gaus')(x)
    attngaus = BatchNormalization(name='selfattn_m1_gaus_bn')(attngaus)   # (attnphi)    # 784 1296
    attntheta = Reshape((784, 480))(attntheta)
    attnphi = Reshape((784, 480))(attnphi)
    attnphi = Permute((2,1))(attnphi)
    attngaus = Reshape((784, 480))(attngaus)
    dot_prod_1 = Dot((2,1))([attntheta, attnphi])
    dot_prod_1 = Activation('softmax', name='dot_prod_1_softmax')(dot_prod_1)
    dot_prod_2 = Dot((2,1))([dot_prod_1, attngaus])
    attnconv1 = Reshape((4,14,14,480))(dot_prod_2)
    attn_concat = Add()([x_red, attnconv1])
    #attn_concat = conv3d_bn(attn_concat, 832, conv_stride, 1, 1, kernel_regularizer=kernel_regularizer, padding='same', strides=(conv_stride, 1, 1), name='attn_concat')
    conv1d_cls_out_enc = UpSampling3D(size=(2,4,4), name='upsampling_4')(attn_concat)
    
    #NO ATTN
    #x_red_2 = MaxPooling3D((4,1,1), strides=(4, 1, 1), padding='valid', name='MaxPool_x_red2')(x)
    #conv1d_cls_out_enc = UpSampling3D(size=(1,4,4), name='upsampling_4')(x_red_2) #(attn_concat)

    # ========== Skip links
    x_3c = MaxPooling3D((2, 1, 1), strides=(2, 1, 1), padding='valid', name='MaxPool2d_x_3c')(x_3c)
    #x_2c = MaxPooling3D((2, 1, 1), strides=(2, 1, 1), padding='valid', name='MaxPool2d_x_2c')(x_2c)

    # ========== Added decoder part 
    dropout_prob = 0.3
    deconv3d_3c = Deconvolution3D(480, (3, 3, 3), kernel_regularizer=kernel_regularizer, padding='same', strides=(1, 2, 2), name='deconv3d_3c')(x)
    deconv3d_3c = BatchNormalization(axis=4, scale=False, name='deconv3d_3c_bn')(deconv3d_3c)
    deconv3d_3c = Activation('relu', name='deconv3d_3c_act')(deconv3d_3c)
    
    deconv3d_3c_out = conv3d_bn(deconv3d_3c, 240, conv_stride, 1, 1, kernel_regularizer=kernel_regularizer, activation_type = 'relu', padding='same', strides=(conv_stride, 1, 1), name='deconv3d_3c_out')
    deconv3d_3c_out = UpSampling3D(size=(2,2,2), name='upsampling_3')(deconv3d_3c_out)
    deconv3d_3c_cat = layers.concatenate([x_3c, deconv3d_3c], axis=channel_axis, name='Merge_3c')
      
    deconv3d_2b = Deconvolution3D(192, (3, 3, 3), kernel_regularizer=kernel_regularizer, padding='same', strides=(2, 2, 2), name='deconv3d_2b')(deconv3d_3c_cat)
    deconv3d_2b = BatchNormalization(axis=4, scale=False, name='deconv3d_2b_bn')(deconv3d_2b)
    deconv3d_2b = Activation('relu', name='deconv3d_2b_act')(deconv3d_2b)
    # x_2c skip moved above
    deconv3d_2b_cat = layers.concatenate([x_2c, deconv3d_2b], axis=channel_axis, name='Merge_2c')
    deconv3d_2b_cat = SpatialDropout3D(dropout_prob)(deconv3d_2b_cat)
    
    # ===================
    # Actor
    conv1d_cls_1a_actor = conv3d_bn(deconv3d_2b_cat, 64, 3, 3, 3, kernel_regularizer=kernel_regularizer, activation_type = 'relu', padding='same', name='conv1d_cls_1a_actor')
    # Dilated/atrous at rate [3,6,9,12]
    conv1d_cls_a_actor = Conv3D(16, (3, 3, 3), kernel_regularizer=kernel_regularizer, padding='same', dilation_rate=(1, 3, 3), name='dilated_conv3d_cls_a_actor')(conv1d_cls_1a_actor)
    conv1d_cls_a_actor = BatchNormalization(name='dilated_conv3d_cls_a_bn_actor')(conv1d_cls_a_actor)
    conv1d_cls_a_actor = Activation('relu',name='dilated_conv3d_cls_a_act_actor')(conv1d_cls_a_actor)
    conv1d_cls_b_actor = Conv3D(32, (3, 3, 3), kernel_regularizer=kernel_regularizer, padding='same', dilation_rate=(1, 6, 6), name='dilated_conv3d_cls_b_actor')(conv1d_cls_1a_actor)
    conv1d_cls_b_actor = BatchNormalization(name='dilated_conv3d_cls_b_bn_actor')(conv1d_cls_b_actor)
    conv1d_cls_b_actor = Activation('relu',name='dilated_conv3d_cls_b_act_actor')(conv1d_cls_b_actor)
    conv1d_cls_c_actor = Conv3D(32, (3, 3, 3), kernel_regularizer=kernel_regularizer, padding='same', dilation_rate=(1, 9, 9), name='dilated_conv3d_cls_c_actor')(conv1d_cls_1a_actor)
    conv1d_cls_c_actor = BatchNormalization(name='dilated_conv3d_cls_c_bn_actor')(conv1d_cls_c_actor)
    conv1d_cls_c_actor = Activation('relu',name='dilated_conv3d_cls_c_act_actor')(conv1d_cls_c_actor)
    conv1d_cls_d_actor = Conv3D(16, (3, 3, 3), kernel_regularizer=kernel_regularizer, padding='same', dilation_rate=(1, 12, 12), name='dilated_conv3d_cls_d_actor')(conv1d_cls_1a_actor)
    conv1d_cls_d_actor = BatchNormalization(name='dilated_conv3d_cls_d_bn_actor')(conv1d_cls_d_actor)
    conv1d_cls_d_actor = Activation('relu',name='dilated_conv3d_cls_d_act_actor')(conv1d_cls_d_actor)
    conv1d_cls_1a_actor = layers.concatenate([conv1d_cls_a_actor, conv1d_cls_b_actor, conv1d_cls_c_actor, conv1d_cls_d_actor], axis=channel_axis, name='conv1d_cls_pyramid_actor')
    conv1d_cls_1a_actor = layers.concatenate([conv1d_cls_1a_actor, deconv3d_3c_out, conv1d_cls_out_enc], axis=channel_axis, name='conv1d_actor_merge')
    #conv1d_cls_1a_actor = conv3d_bn(conv1d_cls_1a_actor, 64, conv_stride, 1, 1, kernel_regularizer=kernel_regularizer, activation_type = 'relu', padding='same', strides=(conv_stride, 1, 1), name='conv1d_cls_1a_pyramid_actor')
    

    # ===================
    # Action
    
    conv1d_cls_1a = conv3d_bn(deconv3d_2b_cat, 64, 3, 3, 3, kernel_regularizer=kernel_regularizer, activation_type = 'relu', padding='same', name='conv1d_cls_1a')
    # Dilated/atrous at rate [3,6,9,12]
    conv1d_cls_a = Conv3D(16, (3, 3, 3), kernel_regularizer=kernel_regularizer, padding='same', dilation_rate=(1, 3, 3), name='dilated_conv3d_cls_a')(conv1d_cls_1a)
    conv1d_cls_a = BatchNormalization(name='dilated_conv3d_cls_a_bn')(conv1d_cls_a)
    conv1d_cls_a = Activation('relu',name='dilated_conv3d_cls_a_act')(conv1d_cls_a)
    conv1d_cls_b = Conv3D(32, (3, 3, 3), kernel_regularizer=kernel_regularizer, padding='same', dilation_rate=(1, 6, 6), name='dilated_conv3d_cls_b')(conv1d_cls_1a)
    conv1d_cls_b = BatchNormalization(name='dilated_conv3d_cls_b_bn')(conv1d_cls_b)
    conv1d_cls_b = Activation('relu',name='dilated_conv3d_cls_b_act')(conv1d_cls_b)
    conv1d_cls_c = Conv3D(32, (3, 3, 3), kernel_regularizer=kernel_regularizer, padding='same', dilation_rate=(1, 9, 9), name='dilated_conv3d_cls_c')(conv1d_cls_1a)
    conv1d_cls_c = BatchNormalization(name='dilated_conv3d_cls_c_bn')(conv1d_cls_c)
    conv1d_cls_c = Activation('relu',name='dilated_conv3d_cls_c_act')(conv1d_cls_c)
    conv1d_cls_d = Conv3D(16, (3, 3, 3), kernel_regularizer=kernel_regularizer, padding='same', dilation_rate=(1, 12, 12), name='dilated_conv3d_cls_d')(conv1d_cls_1a)
    conv1d_cls_d = BatchNormalization(name='dilated_conv3d_cls_d_bn')(conv1d_cls_d)
    conv1d_cls_d = Activation('relu',name='dilated_conv3d_cls_d_act')(conv1d_cls_d)
    conv1d_cls_1a = layers.concatenate([conv1d_cls_a, conv1d_cls_b, conv1d_cls_c, conv1d_cls_d], axis=channel_axis, name='conv1d_cls_pyramid')
    
    # Actor Prior Infusion
    conv1d_cls_merge = layers.concatenate([conv1d_cls_1a, deconv3d_3c_out, conv1d_cls_out_enc, conv1d_cls_1a_actor], axis=channel_axis, name='conv1d_cls_merge')
    
    # FG can be upsampled to larger spatial dimension for higher accuracy
    conv1d_fg_a = UpSampling3D(size=(1,2,2), name='upsampling_fg')(conv1d_cls_1a_actor)
    conv1d_fg = Conv3D(2, (1, 1, 1), activation='softmax', kernel_regularizer=kernel_regularizer, padding='same', name='conv1d_fg')(conv1d_fg_a)
    
    conv1d_actor_out = Conv3D(actor_classes, (1, 1, 1), activation='softmax', kernel_regularizer=kernel_regularizer, padding='same', name='conv1d_actor_out')(conv1d_cls_1a_actor)
    conv1d_actor = Multiply(name='conv1d_actor_mask')([mask_input_actor, conv1d_actor_out])
    conv1d_actor = Add(name='conv1d_actor')([conv1d_actor, mask_input_actor_add])
    
    # SSA-Masking (Uses GT mask for training, and FG mask for inference. Inference will take conv1d_cls_out features and do masking on it using FG output.)
    conv1d_cls = conv3d_bn(conv1d_cls_merge, classes, 3, 3, 3, kernel_regularizer=kernel_regularizer, padding='same', name='conv1d_cls_out')
    conv1d_cls = Multiply(name='conv1d_cls_mul')([mask_input_cls, conv1d_cls])
    #conv1d_cls = Add(name='conv1d_cls_add')([conv1d_cls, mask_input_cls_add])  # Not needed for BCE (No separate background class)
    conv1d_cls = layers.concatenate([conv1d_cls, conv1d_cls_merge], axis=channel_axis, name='conv1d_cls_ssa_merge')
    
    #conv1d_cls = Conv3D(classes, (1, 1, 1), activation='softmax', kernel_regularizer=kernel_regularizer, padding='same', name='conv1d_cls')(conv1d_cls)
    conv1d_cls = Conv3D(classes, (1, 1, 1), activation='sigmoid', kernel_regularizer=kernel_regularizer, padding='same', name='conv1d_cls')(conv1d_cls)
    
    # create model
    inputs = img_input
    
    model = Model([inputs, mask_input_actor, mask_input_actor_add, mask_input_cls], [conv1d_fg, conv1d_actor, conv1d_cls], name='i3d_inception_ssa2d')
    
    return model
    