'''
Correlation 3D in TensorFlow

Given a tensor BxHxWxC, Correlation 2D would get back a tensor with shapes BxHxWxD^2, where 
D is 2d+1 with d=max_displacement.
Correlation3D requires to call Correlation2D N times, with N equals to 2*max_depth_displacement+1
All the correlation2D results are concatenated along channel dimension, so the final tensor would have shape
BxHxWxND^2

Author: Filippo Aleotti

Mail: filippo.aleotti2@unibo.it
'''

from __future__ import division
import tensorflow as tf
from correlations.correlation2D import correlation2D

def correlation3D(tensor_a, tensor_b, pad, kernel_size, max_displacement, max_depth_displacement, stride_1, stride_2):
    '''
        Compute correlation 3D
        Parameters:
            tensor_a (tensor BxHxWxC): first tensor
            tensor_b (tensor BxHxWxC): second tensor
            pad (int): padding value used in correlation2D
            kernel_size (int): length of kernel used in correlation2D
            max_displacement (int): number of elements looked at in neighborhood during correlation2D
            max_depth_displacement (int): number of elements looked at in neighborhood during correlation3D
            stride_1 (int): stride used for tensor_a
            stride_2 (int): stride used for tensor_b

        Returns:
            output_tensor tensor BxHxWxQ): resulting tensor. Q would be (max_displacement*2+1)**2 * (max_depth_displacement*2)+1
    '''
    assert max_depth_displacement >=0

    with tf.variable_scope('correlation3D'):
        corr2D_params = {
            'kernel_size': kernel_size,
            'max_displacement': max_displacement,
            'stride_1': stride_1,
            'stride_2': stride_2
        }
        correlation_results = []
        for current_index in range(-max_depth_displacement, max_depth_displacement+1):
            correlation = _corr(current_index, tensor_a, tensor_b, corr2D_params)
            correlation_results.append(correlation)
        output_tensor = tf.concat(correlation_results, axis=-1)
    return output_tensor

def _corr(current_index, tensor_a, tensor_b, corr2D_params):
    '''
        Inner correlation op
        At each iteration, output_tensor accumulator must be update with the result of the 2D correlation
        applied with a slice of the original second tensor
    '''
    with tf.variable_scope('operation'):
        b,h,w,c = tensor_a.get_shape().as_list()
        starting_channel = current_index if current_index>0 else 0
        offset = c - abs(current_index)
        initial_pad = abs(current_index) if current_index<0 else 0 
        ending_pad = starting_channel
        tensor_slice = tf.pad(tensor_b[:,:,:,starting_channel:starting_channel+offset],[ [0,0],[0,0],[0,0],[initial_pad,ending_pad]])        
        corr2d_values = (2*corr2D_params['max_displacement']+1)**2
        correlation = correlation2D(tensor_a, tensor_slice,
            pad=corr2D_params['max_displacement'], kernel_size=corr2D_params['kernel_size'], max_displacement=corr2D_params['max_displacement'],
            stride_1=corr2D_params['stride_1'], stride_2=corr2D_params['stride_2'])
    return correlation

if __name__ == '__main__':
    import numpy as np

    shape = [2,256,256,64]
    tensor_a = tf.random_uniform(shape, minval=0, maxval=50.)
    tensor_b = tf.random_uniform(shape, minval=0, maxval=50.)
    max_displacement = 4
    kernel_size=1
    stride_1=1
    stride_2=1
    max_depth_displacement = 0
    corr2d = correlation2D(tensor_a, tensor_b, pad=max_displacement, kernel_size=kernel_size, max_displacement=max_displacement, stride_1=stride_1, stride_2=stride_2)
    corr3d = correlation3D(tensor_a, tensor_b, pad=max_displacement, kernel_size=kernel_size, max_displacement=max_displacement, stride_1=stride_1, stride_2=stride_2, max_depth_displacement=max_depth_displacement)
    corr3d_2 = correlation3D(tensor_a, tensor_b, pad=max_displacement, kernel_size=kernel_size, max_displacement=max_displacement, stride_1=stride_1, stride_2=stride_2, max_depth_displacement=4)

    session = tf.Session()
    corr2D_res, corr3D_res = session.run([corr2d, corr3d])
    assert np.array_equal(corr2D_res,corr3D_res)
    assert corr3D_res.shape == (2,256,256,81)

    corr3D_res = session.run(corr3d_2)
    assert corr3D_res.shape == (2,256,256,729)

