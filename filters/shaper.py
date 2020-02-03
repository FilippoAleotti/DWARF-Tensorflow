'''
Set shape of image

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

from general.filter import GeneralFilter
import tensorflow as tf
import numpy as np 
from main_utils.dataloader_utils import get_number_of_paths

class Shaper(GeneralFilter):

    def __init__(self, params):
        super(Shaper, self).__init__(params)
        self.channels = self.filter_params[self.name]['channels']
        self.height = self.get_key_or_default('height', None)
        self.width = self.get_key_or_default('width', None)

    def filter(self, list_of_samples):
        with tf.variable_scope('shaper_filter'):
            for i in range(len(self.channels)):
                list_of_samples[i].set_shape([self.height,self.width,self.channels[i]])
        return list_of_samples

    def _set_name(self):
        return 'shaping_params'