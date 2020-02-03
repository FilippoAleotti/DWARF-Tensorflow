'''
Correlation 1D in pure tensorflow
https://github.com/fabiotosi92/monoResMatch-Tensorflow
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def correlation1D(x, y, pad, kernel_size, max_displacement, stride_1, stride_2, stride=1):
    with tf.variable_scope('tf_correlation'):
        assert stride_1 == 1
        assert stride_2 == 1
        assert kernel_size == 1        
        corr_tensors = []
        y_shape = tf.shape(y)
        y_feature = tf.pad(y,[[0,0],[0,0],[max_displacement,max_displacement],[0,0]])
        for i in range(-max_displacement, max_displacement+1,stride_1):
            shifted = tf.slice(y_feature, [0, 0, i + max_displacement, 0], [-1, y_shape[1], y_shape[2], -1])
            corr_tensors.append(tf.reduce_mean(shifted*x, axis=-1, keepdims=True))

        result = tf.concat(corr_tensors,axis=-1)
        return result