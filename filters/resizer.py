'''
Resize an image using AREA method to a given dimension

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

from general.filter import GeneralFilter
import tensorflow as tf
import numpy as np 
from main_utils.dataloader_utils import get_number_of_paths

class Resizer(GeneralFilter):

    def __init__(self, params):
        super(Resizer, self).__init__(params)
        self.width = self.filter_params[self.name]['width']
        self.height =self.filter_params[self.name]['height']
        self.filename_file = params['dataloader']['filenames']
        self.mode = self.filter_params[self.name]['mode']
        self.number_of_paths_in_line = get_number_of_paths(self.filename_file)
        self.excluded_indexes = self.get_key_or_default('excluded_indexes', None)

    def filter(self, list_of_samples):
        ''' 
            Resize a list of samples
        '''
        with tf.variable_scope('resizer_filter'):
            resized_images= []
            for i in range(self.number_of_paths_in_line):
                if self.excluded_indexes is not None and i in self.excluded_indexes:
                    resized_images.append(list_of_samples[i])
                else:
                    resized_images.append(tf.image.resize_images(list_of_samples[i], [self.height, self.width], tf.image.ResizeMethod.AREA))
            if self.mode == 'testing':
                resized_images.append(list_of_samples[-1]) # original image must not be resized	
            return resized_images
    
    def _set_name(self):
        return 'resizing_params'