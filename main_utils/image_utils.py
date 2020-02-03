'''
Utilities for handling images

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
import numpy as np

def error_image(predicted, gt, mask_occ, mask_noc=None, log_colors=True):
    '''
        From Unflow by Meister et al
        https://raw.githubusercontent.com/simonmeister/UnFlow/master/src/e2eflow/core/flow_util.py

        Visualize the error between two flows as 3-channel color image.

        Adapted from the KITTI C++ devkit.

        Args:
            predicted: first motion of shape [num_batch, height, width, c].
            gt: ground truth motion vector
            mask_occ: validity mask of shape [num_batch, height, width, 1].
                Equals 1 at (occluded and non-occluded) valid pixels.
            mask_noc: Is 1 only at valid pixels which are not occluded.
    '''
    mask_noc = tf.ones(tf.shape(mask_occ)) if mask_noc is None else mask_noc
    diff_sq = (predicted - gt) ** 2
    diff = tf.sqrt(tf.reduce_sum(diff_sq, [3], keepdims=True))
    if log_colors:
        num_batch, height, width, _ = tf.unstack(tf.shape(predicted))
        colormap = [
            [0,0.0625,49,54,149],
            [0.0625,0.125,69,117,180],
            [0.125,0.25,116,173,209],
            [0.25,0.5,171,217,233],
            [0.5,1,224,243,248],
            [1,2,254,224,144],
            [2,4,253,174,97],
            [4,8,244,109,67],
            [8,16,215,48,39],
            [16,1000000000.0,165,0,38]]
        colormap = np.asarray(colormap, dtype=np.float32)
        colormap[:, 2:5] = colormap[:, 2:5] / 255
        mag = tf.sqrt(tf.reduce_sum(tf.square(gt), 3, keepdims=True))
        error = tf.minimum(diff / 3, 20 * diff / mag)
        im = tf.zeros([num_batch, height, width, 3])
        for i in range(colormap.shape[0]):
            colors = colormap[i, :]
            cond = tf.logical_and(tf.greater_equal(error, colors[0]),
                                  tf.less(error, colors[1]))
            im = tf.where(tf.tile(cond, [1, 1, 1, 3]),
                           tf.ones([num_batch, height, width, 1]) * colors[2:5],
                           im)
        im = tf.where(tf.tile(tf.cast(mask_noc, tf.bool), [1, 1, 1, 3]),
                       im, im * 0.5)
        im = im * mask_occ
    else:
        error = (tf.minimum(diff, 5) / 5) * mask_occ
        im_r = error # errors in occluded areas will be red
        im_g = error * mask_noc
        im_b = error * mask_noc
        im = tf.concat(axis=3, values=[im_r, im_g, im_b])
    return im