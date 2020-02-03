'''
Crop an image

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

from general.filter import GeneralFilter
import tensorflow as tf
import numpy as np 

class Cropper(GeneralFilter):

    def __init__(self, params):
        super(Cropper, self).__init__(params)
        self.crop_width = self.filter_params[self.name]['width']
        self.crop_height =self.filter_params[self.name]['height']
        self.final_dimensions = self.filter_params[self.name]['final_dimensions']

    def filter(self, list_of_samples):
        ''' 
            Crop a list of samples
        '''
        with tf.variable_scope('cropper_filter'):
            summed_dimension = np.sum(self.final_dimensions)
            crops = tf.random_crop(tf.concat([sample for sample in list_of_samples], -1), [self.crop_height, self.crop_width, summed_dimension])
            splits = tf.split(crops, self.final_dimensions, -1)

            with tf.variable_scope('set_shapes'):
                splits[0].set_shape([self.crop_height, self.crop_width, self.final_dimensions[0]]) #left_t0
                splits[1].set_shape([self.crop_height, self.crop_width, self.final_dimensions[1]]) #right_t0
                splits[2].set_shape([self.crop_height, self.crop_width, self.final_dimensions[2]]) #left_t1
                splits[3].set_shape([self.crop_height, self.crop_width, self.final_dimensions[3]]) #right_t1
                splits[4].set_shape([self.crop_height, self.crop_width, self.final_dimensions[4]]) #disp_t0
                splits[5].set_shape([self.crop_height, self.crop_width, self.final_dimensions[5]]) #disp_t01
                splits[6].set_shape([self.crop_height, self.crop_width, self.final_dimensions[6]]) #gt_flow
            return splits
    
    def _set_name(self):
        return 'cropping_params'