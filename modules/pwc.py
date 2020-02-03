'''
Network like the PWC-Net proposed by D. Sun et al in https://arxiv.org/pdf/1709.02371.pdf
Code at https://github.com/NVlabs/PWC-Net

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
from main_utils.ops import image_warp
from external_packages.correlation2D.ops import correlation

class Encoder(object):
    def __init__(self, image):
        self.image = image
        self.build()

    def build(self):
        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
            level_1 = self.pyramid_layer(self.image,   16,  'level_1')    # H/2
            level_2 = self.pyramid_layer(level_1,      32,  'level_2')    # H/4
            level_3 = self.pyramid_layer(level_2,      64,  'level_3')    # H/8
            level_4 = self.pyramid_layer(level_3,      96,  'level_4')    # H/16
            level_5 = self.pyramid_layer(level_4,      128, 'level_5')    # H/32
            level_6 = self.pyramid_layer(level_5,      196, 'level_6')    # H/64

            self.features = [None, level_1, level_2, level_3, level_4, level_5, level_6]

    def pyramid_layer(self, t0, out_c, name):
        with tf.variable_scope(name):
            c1 = slim.conv2d(t0, out_c, 3,  stride=2, activation_fn = leaky_relu)
            c2 = slim.conv2d(c1, out_c, 3,  stride=1, activation_fn = leaky_relu)
            c3 = slim.conv2d(c2, out_c, 3,  stride=1, activation_fn = leaky_relu)
        return c3

class Decoder(object):
    def __init__(self, enc_1, enc_2, height, width, flow_scale, use_dense_connections, use_context, name):
        self.enc_1 = enc_1
        self.enc_2 = enc_2
        self.height = height
        self.width = width
        self.use_dense_connections = use_dense_connections
        self.use_context = use_context
        self.name = name
        self.flow_scale = flow_scale
        self.final_layer_index = 2
        self.build()
    
    def build(self):
   
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('block_6'):
                with tf.variable_scope('correlation'):
                    corr6 = correlation(self.enc_1[6], self.enc_2[6], 
                        pad=4, kernel_size=1, max_displacement=4, stride_1=1, stride_2=1)
                    corr6 = tf.nn.leaky_relu(corr6, alpha= 0.1)
                flow6, up_flow6, up_feat6 = self.estimator(corr6)                                     # H/64
                        
            flow5, up_flow5, up_feat5 = self.decoder_block(up_flow6, up_feat6, 5, 'block_5')    # H/32
            flow4, up_flow4, up_feat4 = self.decoder_block(up_flow5, up_feat5, 4, 'block_4')    # H/16
            flow3, up_flow3, up_feat3 = self.decoder_block(up_flow4, up_feat4, 3, 'block_3')    # H/8
            flow2, _, _               = self.decoder_block(up_flow3, up_feat3, 2, 'block_2')    # H/4

            self.flows = [flow2, flow3, flow4, flow5, flow6]

    def decoder_block(self, flow, upsampled_features, feature_index, name):
        with tf.variable_scope(name):
            width = self.scale_dimension(self.width, feature_index)
            height = self.scale_dimension(self.height, feature_index)
            final_layer = feature_index == self.final_layer_index
            weight = self.flow_scale  / (2**(feature_index)) 

            with tf.variable_scope('feature_warping'):
                warp = image_warp( self.enc_2[feature_index], flow * weight)
            
            with tf.variable_scope('correlation'):
                correlation_volume = correlation(self.enc_1[feature_index], warp, 
                    pad=4, kernel_size=1, max_displacement=4, stride_1=1, stride_2=1)
                correlation_volume = tf.nn.leaky_relu(correlation_volume, alpha=0.1)
            
            with tf.variable_scope('build_input'):
                x = tf.concat([correlation_volume, self.enc_1[feature_index], flow, upsampled_features], -1)

            flow, up_flow, up_feat = self.estimator(x, final_layer)

            return flow, up_flow, up_feat

    def estimator(self, x, final_layer=False):
        with tf.variable_scope('estimator'):
        
            x_0 = self.estimator_block(x,   128,  0, self.use_dense_connections)
            x_1 = self.estimator_block(x_0, 128,  1, self.use_dense_connections)
            x_2 = self.estimator_block(x_1,  96,  2, self.use_dense_connections)
            x_3 = self.estimator_block(x_2,  64,  3, self.use_dense_connections)
            x_4 = self.estimator_block(x_3,  32,  4, self.use_dense_connections)
            
            flow = slim.conv2d(x_4,  num_outputs=2, kernel_size=3,  stride=1, activation_fn=None, scope='flow') 
            
            if final_layer and self.use_context:
                residual = self.context_network(x_4)
                refined_flow = flow + residual
                return refined_flow, None, None

            up_flow  = slim.conv2d_transpose(flow, num_outputs=2, kernel_size=4, stride=2, activation_fn=None, scope='upsampled_flow')
            up_features = slim.conv2d_transpose(x_4, num_outputs=2, kernel_size=4, stride=2, activation_fn=None, scope='upsampled_flow_features')
            
            return flow, up_flow, up_features

    def estimator_block(self, x, channel_out, index, use_dense_connections):
        with tf.variable_scope('estimator_block_'+str(index)):
            conv = slim.conv2d(x, channel_out, 3,   stride=1, activation_fn=leaky_relu)
            if use_dense_connections:
                conv = tf.concat([x,conv],-1, name='dense_connection')
            return conv

    def context_network(self, x, out_channels=2, name='context_network'):
        with tf.variable_scope(name):
            dc_conv1 = slim.conv2d(x,          128, 3, stride=1,  activation_fn=leaky_relu)
            dc_conv2 = slim.conv2d(dc_conv1,   128, 3, stride=1,  activation_fn=leaky_relu,   rate= 2)
            dc_conv3 = slim.conv2d(dc_conv2,   128, 3, stride=1,  activation_fn=leaky_relu,   rate= 4)
            dc_conv4 = slim.conv2d(dc_conv3,    96, 3, stride=1,  activation_fn=leaky_relu,   rate= 8)
            dc_conv5 = slim.conv2d(dc_conv4,    64, 3, stride=1,  activation_fn=leaky_relu,   rate=16)
            dc_conv6 = slim.conv2d(dc_conv5,    32, 3, stride=1,  activation_fn=leaky_relu)
            dc_conv7 = slim.conv2d(dc_conv6, out_channels, 3, stride=1,  activation_fn=None)
            return dc_conv7
    
    def scale_dimension(self, dimension, index):
        return dimension / (2**index)

def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha= 0.1)





