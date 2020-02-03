'''
Disparity utils

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
import cv2
import numpy as np 
from main_utils.pfm_utils import load_pfm
from kitti.test.ops import *

def colormap_jet(img, color_factor):
    color_image = cv2.cvtColor(cv2.applyColorMap(np.uint8(img*color_factor),2), cv2.COLOR_BGR2RGB)
    return color_image

def color_disparity(disparity, color_factor):
    '''
        Color a disparity batch
    '''
    with tf.variable_scope('color_disparity'):
        batch_size = disparity.shape[0]
        color_maps = []
        for i in range(batch_size):
            color_disp = tf.py_func(colormap_jet, [disparity[i], color_factor], tf.uint8)
            color_maps.append(color_disp)
        color_batch = tf.stack(color_maps, axis=0)
        return color_batch

def tf_load_disparity_pfm(filename):
    ''' Read a pfm flow '''
    file_pfm = tf.py_func( load_pfm, [filename],[tf.float32])
    output = file_pfm[0]
    return output

def save_kitti_disp(disparity, destination):
    '''
        Save a disparity map as expected by Kitti Sceneflow Benchmark
    '''
    cv2.imwrite(destination, (disparity * 256).astype(np.uint16))

def tf_load_16_bit_disparity(filename):
    '''
        Load a 16 bit disparity map, 
        scaled by 256 as defined in KITTI
    '''
    with tf.variable_scope('tf_load_16_bit_disparity'):
        disp = tf.image.decode_png(tf.read_file(filename),dtype=tf.uint16)
        disp =  tf.cast(disp, tf.float32)
        disp = disp / 256
    return disp

def extract_disparity_mask(disparity):
    '''
    Disparity are saved as single channel
    image, with 0 where a pixel is not valid.
    This function return a boolean mask, with 0
    where disparity is 0.
    Returns:
        disparity: [BxHxWx1]
        mask:[BxHxWx1]
    '''
    with tf.variable_scope('extract_disparity_mask'):
        valid_pixel = tf.ones_like(disparity, tf.float32)
        not_valid_pixel = tf.zeros_like(disparity, tf.float32)
        mask = tf.where(tf.equal(disparity, 0.), not_valid_pixel, valid_pixel)
        return disparity, mask