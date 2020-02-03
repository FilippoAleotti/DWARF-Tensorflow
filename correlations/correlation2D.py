'''
Correlation 2D in pure TensorFlow
from https://github.com/philferriere/tfoptflow/blob/master/tfoptflow/core_costvol.py
License: MIT
'''
import tensorflow as tf

def correlation2D(feature_t0, warped_features, pad, kernel_size, max_displacement, stride_1, stride_2):
    '''
        Adapter, in order to be compatible with CUDA interface.
    ''' 
    assert stride_1 == 1
    assert stride_2 == 1
    assert kernel_size == 1
    search_range = max_displacement
    padded_lvl = tf.pad(warped_features, [[0, 0], [search_range, search_range], [search_range, search_range], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(feature_t0))
    max_offset = search_range * 2 + 1

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
            cost = tf.reduce_mean(feature_t0 * slice, axis=3, keepdims=True)
            cost_vol.append(cost)
    cost_vol = tf.concat(cost_vol, axis=3)
    return cost_vol