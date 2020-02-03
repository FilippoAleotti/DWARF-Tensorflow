'''
Utilities for Optical Flow

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
import numpy as np
from main_utils.pfm_utils import load_pfm
import cv2

def flow_to_color(flow, mask=None, max_flow=None):
    '''
    From Unflow by Meister et al
    https://arxiv.org/pdf/1711.07837.pdf
    https://github.com/simonmeister/UnFlow
    
    Converts flow to 3-channel color image.
    Args:
        flow: tensor of shape [num_batch, height, width, 2].
        mask: flow validity mask of shape [num_batch, height, width, 1].
    '''
    n = 8
    num_batch, height, width, _ = tf.unstack(tf.shape(flow))
    mask = tf.ones([num_batch, height, width, 1]) if mask is None else mask
    flow_u, flow_v = tf.unstack(flow, axis=3)
    if max_flow is not None:
        max_flow = tf.maximum(max_flow, 1)
    else:
        max_flow = tf.reduce_max(tf.abs(flow * mask))
    mag = tf.sqrt(tf.reduce_sum(tf.square(flow), 3))
    angle = atan2(flow_v, flow_u)

    im_h = tf.mod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = tf.clip_by_value(mag * n / max_flow, 0, 1)
    im_v = tf.clip_by_value(n - im_s, 0, 1)
    im_hsv = tf.stack([im_h, im_s, im_v], 3)
    im = tf.image.hsv_to_rgb(im_hsv)
    return im * mask

def atan2(y, x):
    '''
        From Unflow by Meister et al
        https://raw.githubusercontent.com/simonmeister/UnFlow/master/src/e2eflow/core/flow_util.py
    '''
    angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x,0.0), tf.greater_equal(y,0.0)),
                      tf.atan(y/x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x,0.0), tf.less(y,0.0)),
                      tf.atan(y/x) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)),
                      np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)),
                      -np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0),tf.equal(y,0.0)),
                      np.nan * tf.zeros_like(x), angle)
    return angle


def flow_error_avg(flow_1, flow_2, mask):
    '''
        From Unflow by Meister et al
        https://raw.githubusercontent.com/simonmeister/UnFlow/master/src/e2eflow/core/flow_util.py
        Evaluates the average endpoint error between flow batches.
    '''
    with tf.variable_scope('flow_error_avg'):
        diff = euclidean(flow_1 - flow_2) * mask
        error = tf.reduce_sum(diff) / tf.reduce_sum(mask)
        return error


def outlier_ratio(gt_flow, flow, mask, threshold=3.0, relative=0.05):
    '''
        From Unflow by Meister et al
        https://raw.githubusercontent.com/simonmeister/UnFlow/master/src/e2eflow/core/flow_util.py
    '''
    diff = euclidean(gt_flow - flow) * mask
    if relative is not None:
        threshold = tf.maximum(threshold, euclidean(gt_flow) * relative)
        outliers = tf.cast(tf.greater_equal(diff, threshold), tf.float32)
    else:
        outliers = tf.cast(tf.greater_equal(diff, threshold), tf.float32)
    ratio = tf.reduce_sum(outliers) / tf.reduce_sum(mask)
    return ratio


def outlier_pct(gt_flow, flow, mask, threshold=3.0, relative=0.05):
    '''
        From Unflow by Meister et al
        https://raw.githubusercontent.com/simonmeister/UnFlow/master/src/e2eflow/core/flow_util.py
    '''
    frac = outlier_ratio(gt_flow, flow, mask, threshold, relative) * 100
    return frac


def euclidean(t):
    '''
        From Unflow by Meister et al
        https://raw.githubusercontent.com/simonmeister/UnFlow/master/src/e2eflow/core/flow_util.py
    '''
    return tf.sqrt(tf.reduce_sum(t ** 2, [3], keepdims=True))

def evaluate_flow(flow, gt_flow, mask=None):
    outliers = outlier_pct(gt_flow, flow, mask)
    avg_error = flow_error_avg(flow, gt_flow, mask)
    return avg_error, outliers

def load_flo(filename):
    '''
        Load a .flo file
        More info about flo format here: http://vision.middlebury.edu/flow/code/flow-code/README.txt
    '''
    
    with open(filename,'rb') as f:
        sanity_check = np.fromstring(f.read(4), dtype='<f4')
        assert sanity_check == 202021.25

        width = np.asscalar(np.fromstring(f.read(4), dtype=np.int32))
        height = np.asscalar(np.fromstring(f.read(4), dtype=np.int32))

        number_of_elements = width*height*2
        flows = np.fromstring(f.read(number_of_elements*4), dtype='<f4')
        flows = np.reshape(flows, (height, width,2))
        
        return flows

def tf_load_flo(filename):
    ''' Read a flo flow '''
    file_flo = tf.py_func( load_flo, [filename],[tf.float32])
    output = file_flo[0]
    return output

def tf_load_flow_pfm(filename):
    ''' Read a pfm flow '''
    file_pfm = tf.py_func(load_pfm, [filename],tf.float32)
    output = file_pfm[:,:,:2]
    return output

def save_16_bit_flow(flow, destination):
    '''
        Save a 16 bit optical flow at a given destination
    '''
    cv2.imwrite(destination, flow.astype(np.uint16))

def expand_flow_dimensionality(flow):
    '''
        Prepare a BGR flow:
        First channel is made by 1, Second is Flow along Y axis and finally Flow along X axis.
        This flow is the one expected by OpenCv, since it save with BGR format.
        If you don't use OpenCv, you have to swap the channels in order to have [Flow X, Flow Y, 1] 
    '''
    flow_x = flow[:,:,0:1]
    flow_y = flow[:,:,1:2]
    height, width,_ = flow_x.shape
    ones = np.ones((height, width,1), np.float32)
    flow = np.concatenate((ones,flow_y,flow_x), axis=-1)
    return flow

def center_flow(flow):
    return flow * 64. + 2**15

def save_kitti_flow(flow, destination):
    '''
        Save a flow as expected by Kitti Benchmark
    '''
    flow = center_flow(flow)
    flow = expand_flow_dimensionality(flow)
    save_16_bit_flow(flow, destination)

def tf_load_16_bit_flow(filename):
    '''
        Load a 16 bit flow map,
        scaled as defined in KITTI
    '''
    with tf.variable_scope('tf_load_16_bit_flow'):
        flow = tf.image.decode_png(tf.read_file(filename),dtype=tf.uint16)
        flow =  tf.cast(flow, tf.float32)
        flow_uv = flow[:,:,:2]
        validity_mask = flow[:,:,2:3] #validity mask must not be scaled!
        flow_uv = (flow_uv - 2.0**(15)) /64.
        scaled_flow = tf.concat([flow_uv, validity_mask], axis=-1) 
    return scaled_flow

def extract_flow_mask(flow):
    '''
    Ground truth flow is 3 channel flow, where
    last channel is validity mask.
    This function extract that mask from the given
    flow.
    Returns:
        2 channel flow: [BxHxWx2] 
        mask: [BxHxWx1]
    '''
    with tf.variable_scope('extract_flow_mask'):
        two_channel_flow = flow[:,:,:,:2]#first 2 channel of flow
        mask = flow[:,:,:,2:3] #validity mask
        return two_channel_flow, mask


UNKNOWN_FLOW_THRESH = 1e7
def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))
  
    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)
    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0
    return np.uint8(img)

def read_16_bit_flow(f):
    flow = cv2.cvtColor(cv2.imread(f, -1), cv2.COLOR_BGR2RGB)
    flow_uv = flow[:,:,:2]
    validity_mask = flow[:,:,2:3] #validity mask must not be scaled!
    flow_uv = (flow_uv - 2.0**(15)) /64.
    scaled_flow = np.concatenate([flow_uv, validity_mask], axis=-1) 
    return scaled_flow
