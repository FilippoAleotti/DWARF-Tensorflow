'''
Modules for Dwarf 

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
from main_utils.ops import image_warp
from correlations import correlation_factory

correlation1D = correlation_factory.get_correlation1D()
correlation2D = correlation_factory.get_correlation2D()
correlation3D = correlation_factory.get_correlation3D()

"""
from external_packages.correlation2D.ops import correlation as correlation2D
from external_packages.correlation1D.corr1d import correlation1d as correlation1D
from external_packages.correlation3D.ops import correlation3D
"""

from main_utils.bilinear_sampler_1D import generate_image_left
from modules.pwc import Decoder, leaky_relu

class DwarfDecoder(Decoder):
    def __init__(self, encoder_features, height, width, max_displacement, max_depth_displacement, scale_factor,
        is_training, use_dense_connection=False, final_layer_index=2, 
        pyramids=None, use_volumes_correlation=True, use_context=True):
        '''
            Args:
                encoders_features: features from encoder. List from l1 (H/2) to l6 (H/64) of encoder pyramid
                height, width: size of images at fullres
                max_displacement: max range of search for 2D correlation
                max_depth_displacement: max range of search along channel dimension in 3D correlation
                pyramids: pyramids of images left_t0, right_t0, left_t1, right_t1.
                    Each image is scaled by factor of 2**level.
                    Used only in refinement version
        '''
        self.max_displacement = max_displacement
        self.max_depth_displacement = max_depth_displacement
        self.encoders_features = encoder_features
        self.width = width
        self.height = height
        self.scale_factor = scale_factor
        self.use_dense_connections = use_dense_connection
        self.final_layer_index = final_layer_index
        self.is_training = is_training
        self.use_volumes_correlation = use_volumes_correlation
        self.pyramids = pyramids
        self.use_context = use_context
        self.build()

    def build(self):
        with tf.variable_scope('decoder'):
            sceneflow_6, up_sceneflow_6, up_feat_6 = self.first_block()             # H/64
            sceneflow_5, up_sceneflow_5, up_feat_5 = self.decoder_block(
                up_sceneflow_6, up_feat_6, 5, 'block_5')                            # H/32
            sceneflow_4, up_sceneflow_4, up_feat_4 = self.decoder_block(
                up_sceneflow_5, up_feat_5, 4, 'block_4')                            # H/16
            sceneflow_3, up_sceneflow_3, up_feat_3 = self.decoder_block(
                up_sceneflow_4, up_feat_4, 3, 'block_3')                            # H/8
            sceneflow_2, up_sceneflow_2, up_feat_2 = self.decoder_block(
                up_sceneflow_3, up_feat_3, 2, 'block_2')                            # H/4

            sceneflows = [None, None, sceneflow_2, sceneflow_3,
                          sceneflow_4, sceneflow_5, sceneflow_6]

            if self.is_training:
                self.motion_vectors = [self.extract_motion_vectors(
                    sceneflow) for sceneflow in sceneflows]
            else:
                self.motion_vectors = [self.extract_motion_vectors(sceneflow_2)]

    def first_block(self):
        '''
            First block: direct correlation between encoder features w/o warping
        '''
        with tf.variable_scope('block_6'):
            level = 6
            features = self.get_encoder_features(level)
            correlation_volume = self.create_correlation_volume(
                features, level)
            sceneflow, up_sceneflow, up_features = self.estimator(
                correlation_volume, level)
            return sceneflow, up_sceneflow, up_features

    def decoder_block(self, sceneflow, upsampled_features, level, name):
        '''
            Warping of encoder features in accord with sceneflow, followed by
            correlation volume computation and estimation of next sceneflow
        '''
        with tf.variable_scope(name):
            weight = self.scale_factor / (2**level)
            print('level:{} , weight:{}, shape:{}'.format(level,weight, sceneflow.shape))

            features = self.get_encoder_features(level)
            warped_features = self.warp_features(features, sceneflow * weight, level)
            correlation_volume = self.create_correlation_volume(
                warped_features, level)
            
            input_volume = tf.concat(
                [correlation_volume, features['left_t0'], sceneflow, upsampled_features], -1)

            sceneflow, up_sceneflow, up_features = self.estimator(
                input_volume, level)
            return sceneflow, up_sceneflow, up_features

    def estimator(self, volume, level):
        '''
            Extract two disparity (left t0 and left_t01) and optical flow 
            from volume a correlation volume
        '''
        with tf.variable_scope('estimator'):
            final_layer = level == self.final_layer_index

            x_0 = self.estimator_block(
                volume,   128,  0, self.use_dense_connections)
            x_1 = self.estimator_block(
                x_0,      128,  1, self.use_dense_connections)
            x_2 = self.estimator_block(
                x_1,       96,  2, self.use_dense_connections)

            flow, features_flow = self.motion_extractor(x_2, 2, 'optical_flow')
            disparity, features_disparity = self.motion_extractor(x_2, 1, 'disparity')
            disparity_change, features_disparity_change = self.motion_extractor(x_2, 1, 'disparity_change')
            
            up_sceneflow = None
            up_features = None
            
            if not final_layer:
                
                up_flow, up_disparity, up_disparity_change = self.upsample(flow, disparity, disparity_change, 'upsample_motion_vectors')
                up_features_flow, up_features_disparity, up_features_disparity_change = self.upsample(features_flow, features_disparity, 
                    features_disparity_change, 'upsample_features')

                up_sceneflow = tf.concat([up_disparity, up_disparity_change, up_flow],-1)
                up_features  = tf.concat([up_features_disparity, up_features_disparity_change, up_features_flow],-1)
                sceneflow = tf.concat([disparity, disparity_change, flow],-1)

                return sceneflow, up_sceneflow, up_features

            if self.use_context:
                residual_flow = self.context_network(features_flow, out_channels=2, name='context_network_flow')
                residual_disparity = self.context_network(features_disparity, out_channels=1, name='context_network_disparity')
                residual_disparity_change = self.context_network(features_disparity_change, out_channels=1, name='context_network_disparity_change')

                flow += residual_flow
                disparity += residual_disparity
                disparity_change += residual_disparity_change

            refined_sceneflow = tf.concat([disparity, disparity_change, flow],-1)
            
            return refined_sceneflow, None, None

    def warp_features(self, features, sceneflow, level):
        '''
            Warp encoder features in accord with estimated sceneflow
        '''
        with tf.variable_scope('warp_features'):
            motion_vectors = self.extract_motion_vectors(sceneflow)

            left_t0_from_right_t0 = self.align_present_right_features(
                features['right_t0'], motion_vectors['disparity'])

            left_t0_from_left_t1 = self.align_future_left_features(
                features['left_t1'], motion_vectors['forward_flow'])

            computed_right_flow = self.compute_flow_to_right(
                motion_vectors['forward_flow'], motion_vectors['disparity_change'], level)
            left_t0_from_right_t1 = self.align_future_right_features(
                features['right_t1'], computed_right_flow)

            warped_features = {
                'left_t0': features['left_t0'],
                'right_t0': left_t0_from_right_t0,
                'left_t1': left_t0_from_left_t1,
                'right_t1': left_t0_from_right_t1
            }
            return warped_features

    def get_encoder_features(self, level):
        '''
            Extract encoder features at a given index
        '''
        with tf.variable_scope('get_encoder_features'):
            features = {}
            features['left_t0'] = self.encoders_features['left_t0'][level]
            features['right_t0'] = self.encoders_features['right_t0'][level]
            features['left_t1'] = self.encoders_features['left_t1'][level]
            features['right_t1'] = self.encoders_features['right_t1'][level]
        return features

    def extract_motion_vectors(self, sceneflow):
        '''
            Put motion vectors (disparity, disp_t1, flow) in a dict
        '''
        with tf.variable_scope('extract_motion_vectors'):
            if sceneflow is None:
                return sceneflow
            motion_vectors = {
                'disparity': tf.expand_dims(sceneflow[:, :, :, 0], axis=-1),
                'disparity_change': tf.expand_dims(sceneflow[:, :, :, 1], axis=-1),
                'forward_flow': sceneflow[:, :, :, 2:],
            }
        return motion_vectors

    def align_present_right_features(self, features_right_t0, disparity_left_t0):
        '''
            Warping right features into left features using left disparity.
            Note that, since we want to predict NEGATIVE disparity, we
            warp with positive disparity and let the network to decide the
            sign
        '''
        with tf.variable_scope('align_present_right_features'):
            return generate_image_left(features_right_t0, disparity_left_t0)

    def align_future_left_features(self, features_left_t1, forward_flow):
        '''
            Warping future left features into curret left features using forward flow.
        '''
        with tf.variable_scope('align_future_left_features'):
            return image_warp(features_left_t1, forward_flow)

    def align_future_right_features(self, features_right_t1, computed_right_flow):
        '''
            Warping future right features into curret left features using
            the computed forward flow.
        '''
        with tf.variable_scope('align_future_right_features'):
            return image_warp(features_right_t1, computed_right_flow)

    def compute_flow_to_right(self, forward_flow, disparity_left_t01, level):
        '''
            Compute the forward flow that maps right t1 image into left t0.
            Note that, since we want to predict NEGATIVE disparity, we
            warp with positive disparity and let the network to decide the
            sign  
        '''
        with tf.variable_scope('compute_flow_to_right'):
            width = self.scale_dimension(self.width, level)
            height = self.scale_dimension(self.height, level)
            zeros = tf.zeros([1, height, width, 1], tf.float32)
            zeros = tf.tile(zeros, [tf.shape(forward_flow)[0],1,1,1])
            computed_forward_flow = forward_flow + tf.concat([disparity_left_t01, zeros], 3)

            return computed_forward_flow

    def create_correlation_volume(self, features, level):
        '''
            Create a volume starting from features
        '''
        with tf.variable_scope('correlation_volume'):
            volume_t0 = self.correlation_1D_block(
                features['left_t0'], features['right_t0'], level, 'volume_disparity_t0')
            volume_t1 = self.correlation_1D_block(
                features['left_t1'], features['right_t1'], level, 'volume_disparity_t1')
            flow_volume = self.correlation_2D_block(features['left_t0'], features['left_t1'], 'flow_volume')
            if self.use_volumes_correlation:
                correlation_3D_volume = self.correlation_3D_block(volume_t0, volume_t1, name='volumes_correlation')
            with tf.variable_scope('volume_fusion'):
                if self.use_volumes_correlation:
                    volume = tf.concat([volume_t0, volume_t1, correlation_3D_volume, flow_volume], -1)
                else:
                    volume = tf.concat([volume_t0, volume_t1, flow_volume], -1)

            return volume
    
    def correlation_3D_block(self, volume_t0, volume_t1, name):
        '''
            Compute 3D correlation
        '''
        with tf.variable_scope(name):
            correlation_3D_volume = correlation3D(volume_t0, volume_t1, pad=self.max_displacement, 
                kernel_size=1, max_displacement=self.max_displacement, max_depth_displacement=self.max_depth_displacement,
                stride_1=1, stride_2=1)
            correlation_3D_volume = tf.nn.leaky_relu(correlation_3D_volume, alpha=0.1)
            return correlation_3D_volume

    def correlation_2D_block(self, volume_t0, volume_t1, name):
        '''
            Compute 2D correlation
        '''
        with tf.variable_scope(name):
            correlation_2D_volume = correlation2D(volume_t0, volume_t1,
                pad=self.max_displacement, kernel_size=1, max_displacement=self.max_displacement, stride_1=1, stride_2=1)
            correlation_2D_volume = tf.nn.leaky_relu(
                correlation_2D_volume, alpha=0.1)
            return correlation_2D_volume

    def correlation_1D_block(self, left_features, right_features, level, name):
        '''
            Compute 1D correlation
        '''
        with tf.variable_scope(name):
            correlation_1D_volume = correlation1D(left_features, right_features, 
                pad=self.max_displacement, kernel_size=1, max_displacement=self.max_displacement, stride_1=1, stride_2=1)
            correlation_1D_volume = tf.nn.leaky_relu(
                correlation_1D_volume, alpha=0.1)
            return correlation_1D_volume

    def scale_dimension(self, dimension, level):
        return dimension / (2**level)
            
    def motion_extractor(self, x, channel_out, name):
        '''
            Analyze volume and extract a single motion vector
        '''
        with tf.variable_scope(name):
            x_0 = self.estimator_block(
                x,       64,  0, self.use_dense_connections)
            x_1 = self.estimator_block(
                x_0,       32,  1, self.use_dense_connections)
            motion = slim.conv2d(x_1,  num_outputs=channel_out, kernel_size=3,  stride=1, activation_fn=None)
        return motion, x_1
    
    def upsample(self, flow, disparity, disparity_change, name):
        with tf.variable_scope(name):
            up_flow = slim.conv2d_transpose(
                flow, num_outputs=2, kernel_size=4, stride=2, activation_fn=None, scope='flow')
            up_disparity = slim.conv2d_transpose(
                disparity, num_outputs=1, kernel_size=4, stride=2, activation_fn=None, scope='disparity')
            up_disparity_change = slim.conv2d_transpose(
                disparity_change, num_outputs=1, kernel_size=4, stride=2, activation_fn=None, scope='disparity_change')
        return up_flow, up_disparity, up_disparity_change
