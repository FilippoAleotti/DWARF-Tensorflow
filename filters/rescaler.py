'''
Resize an image, applying a rescaling factor
Use this filter during training if ground-truth (e.g., LiDAR) are available and you want
to resize an image to a given dimension.
In fact, resizing a ground-truth requires to rescale the values by a scaling factor

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

from general.filter import GeneralFilter
import tensorflow as tf
import numpy as np 
from main_utils.dataloader_utils import get_number_of_paths

class Rescaler(GeneralFilter):

    def __init__(self, params):
        super(Rescaler, self).__init__(params)
        self.width = self.filter_params[self.name]['width']
        self.height =self.filter_params[self.name]['height']

        self.flow_index = self.filter_params[self.name]['flow_index']
        self.disp_indexis = self.filter_params[self.name]['disp_indexes']
        self.resize_gt_method = self._select_resizing_method(self.filter_params[self.name]['resize_gt_method'])
     
        self.filename_file = params['dataloader']['filenames']
        self.number_of_paths_in_line = get_number_of_paths(self.filename_file)

    def filter(self, list_of_samples):
        ''' 
            Resize a list of samples
        '''
        with tf.variable_scope('rescaler_filter'):
            resized_images= []
            image_width= tf.cast(tf.shape(list_of_samples[0])[1], tf.float32)
            image_height= tf.cast(tf.shape(list_of_samples[0])[0], tf.float32)
            scaling_factor_width = self.width/ image_width
            scaling_factor_height = self.height/ image_height
            for i in range(self.number_of_paths_in_line):
                resized_images.append(self._resize(list_of_samples[i], i, scaling_factor_width, scaling_factor_height))
            return resized_images
 
    def _resize(self, image, index, scaling_factor_width, scaling_factor_height):
        with tf.variable_scope('resize'):
            resized_image = self._resize_image(image, index, self.resize_gt_method)

            if self.disp_indexis is not None and index in self.disp_indexis:
                resized_image = resized_image * scaling_factor_width
                return resized_image
            
            if self.flow_index is not None and index in self.flow_index:
                scalings = tf.ones(3, tf.float32) * [scaling_factor_width, scaling_factor_height, 1.]
                resized_image = resized_image * scalings
                return resized_image

            return resized_image

    def _resize_image(self, image, index, resize_gt_method):
        '''
            Resize an image.
            If the image is in flow_index or disp_indexes, the resizing method is selected in
            configuration, otherwise AREA will be applied
        '''
        with tf.variable_scope('resize_image'):
            if  index in self.disp_indexis or index in self.flow_index:
                return tf.image.resize_images(image, [self.height, self.width], resize_gt_method)
            else:
                return tf.image.resize_images(image, [self.height, self.width], tf.image.ResizeMethod.AREA)
    
    def _select_resizing_method(self, selected_method):
        with tf.variable_scope('select_resizing_method'):
            if selected_method == 'AREA':
                return  tf.image.ResizeMethod.AREA
            if selected_method == 'NN':
                return tf.image.ResizeMethod.NEAREST_NEIGHBOR
            raise ValueError('Not valid method for resizing gt. Expected AREA or NN')
        
    def _set_name(self):
        return 'rescaling_params'