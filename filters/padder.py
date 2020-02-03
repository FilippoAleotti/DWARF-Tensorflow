'''
Pad an image to a given shape 

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

from general.filter import GeneralFilter
import tensorflow as tf
import numpy as np 
from main_utils.dataloader_utils import get_number_of_paths, tf_get_height_width

class Padder(GeneralFilter):

    def __init__(self, params):
        super(Padder, self).__init__(params)
        self.width = self.filter_params[self.name]['width']
        self.height =self.filter_params[self.name]['height']
        self.filename_file = params['dataloader']['filenames']
        self.mode = self.filter_params[self.name]['mode']
        self.number_of_paths_in_line = get_number_of_paths(self.filename_file)
        self.excluded_indexes = self.get_key_or_default('excluded_indexes', None)

    def filter(self, list_of_samples):
        ''' 
            Pad a list of samples
        '''
        with tf.variable_scope('padder_filter'):
            resized_images= []
            shape = tf_get_height_width(list_of_samples[0])
            h = shape[0]
            w = shape[1]

            for i in range(self.number_of_paths_in_line):
                if self.excluded_indexes is not None and i in self.excluded_indexes:
                    resized_images.append(list_of_samples[i])
                else:
                    missing_h  = self.height - h
                    top_pad    = missing_h // 2
                    bottom_pad = missing_h - top_pad
                    missing_w  = self.width - w
                    left_pad   = missing_w // 2 
                    right_pad  = missing_w - left_pad
                    image = tf.pad(list_of_samples[i], [[top_pad, bottom_pad], [left_pad, right_pad],[0,0]], 'REFLECT')
                    image.set_shape([self.height, self.width, image.shape[-1]])
                    resized_images.append(image)
            if self.mode == 'testing':
                resized_images.append(list_of_samples[-1]) # original image must not be resized	
            return resized_images
    
    def _set_name(self):
        return 'padding_params'