
import tensorflow as tf
import keras.backend as K
import keras
import numpy as np

def custom_categorical_accuracy(y_true, y_pred):
    slice_true = y_true[:,:,:,:,:-1]
    slice_pred = y_pred[:,:,:,:,:-1]
    accuracy_val = keras.metrics.categorical_accuracy(slice_true,slice_pred)
    return accuracy_val
    

def custom_miou_fg(y_true, y_pred):
    slice_true = y_true[:,:,:,:,:-1]
    slice_pred = K.cast(K.one_hot(K.argmax(y_pred,-1),2), dtype='float32')
    slice_pred = slice_pred[:,:,:,:,:-1]
    #slice_pred = K.cast(K.greater(slice_pred, threshold), dtype='float32') 
    inter = K.sum(slice_pred*slice_true)
    union = K.any(K.stack([slice_true, slice_pred], axis=0), axis=0)
    union = K.sum(K.cast(union,dtype='float32')) + K.epsilon()
    iou = inter/union
    return iou

def custom_miou(y_true, y_pred):
    slice_true = y_true[:,:,:,:,:-1]
    slice_pred = K.cast(K.one_hot(K.argmax(y_pred,-1),81), dtype='float32')
    slice_pred = slice_pred[:,:,:,:,:-1]
    #slice_pred = K.cast(K.greater(slice_pred, threshold), dtype='float32') 
    inter = K.sum(slice_pred*slice_true)
    union = K.any(K.stack([slice_true, slice_pred], axis=0), axis=0)
    union = K.sum(K.cast(union,dtype='float32')) + K.epsilon()
    iou = inter/union
    return iou
    

def custom_miou_sig(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, 0.2), dtype='float32') 
    inter = K.sum(y_pred*y_true)
    union = K.any(K.stack([y_true, y_pred], axis=0), axis=0)
    union = K.sum(K.cast(union,dtype='float32')) + K.epsilon()
    iou = inter/union
    return iou
    

def custom_miou_img(y_true, y_pred):
    
    slice_true = y_true[:,:,:,:-1]
    slice_pred = K.cast(K.one_hot(K.argmax(y_pred,-1),8), dtype='float32')
    slice_pred = slice_pred[:,:,:,:-1]
    #slice_pred = K.cast(K.greater(slice_pred, threshold), dtype='float32') 
    inter = K.sum(slice_pred*slice_true)
    union = K.any(K.stack([slice_true, slice_pred], axis=0), axis=0)
    union = K.sum(K.cast(union,dtype='float32'))
    iou = inter/union
    return iou

def custom_miou_fg_img(y_true, y_pred):
    
    slice_true = y_true[:,:,:,:-1]
    slice_pred = K.cast(K.one_hot(K.argmax(y_pred,-1),2), dtype='float32')
    slice_pred = slice_pred[:,:,:,:-1]
    #slice_pred = K.cast(K.greater(slice_pred, threshold), dtype='float32') 
    inter = K.sum(slice_pred*slice_true)
    union = K.any(K.stack([slice_true, slice_pred], axis=0), axis=0)
    union = K.sum(K.cast(union,dtype='float32'))
    iou = inter/union
    return iou

                
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
    
    
def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
    loss_val = 1 - np.mean(numerator / (denominator + epsilon)) # average over classes and batch
    return loss_val
    
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), (1,2,3,4))
    union = K.sum(K.square(y_true), (1,2,3,4)) + K.sum(K.square(y_pred), (1,2,3,4)) + K.epsilon()
    #return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
    return (2. * intersection) / union

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def selective_dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    fg = K.sum(y_true, axis=-1)
    efg = tf.expand_dims(fg, axis=-1)   # expanded foreground
    y_pred = tf.multiply(y_pred, efg)   # broadcast foreground and multiply with preds
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(y_true,-1) + K.sum(y_pred,-1) + smooth)

def selective_dice_coef_loss(y_true, y_pred):
    return 1-selective_dice_coef(y_true, y_pred)
        
    
def selective_cross_entropy_loss(y_true, y_pred):
    fg = K.sum(y_true, axis=-1)
    locs = tf.where(tf.greater(fg, 0))
    g_pred = tf.gather_nd(y_pred, locs)
    g_true = tf.gather_nd(y_true, locs)
    
    # scale predictions so that the class probas of each sample sum to 1
    g_pred /= K.sum(g_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    g_pred = K.clip(g_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = g_true * K.log(g_pred)
    loss = -K.sum(loss, -1)
    loss = tf.reduce_mean(loss)
    
    return loss
    
def cross_entropy_with_dice(y_true, y_pred):
    
    #sloss = scel(y_true, y_pred)
    sloss = K.categorical_crossentropy(y_true, y_pred)
    dc = dice_coef(y_true, y_pred)
    dcl = 1 - dc
    
    combined_loss = sloss + dcl
    return combined_loss
     
