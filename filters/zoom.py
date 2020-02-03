'''
Agument images using zoom

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
from general.filter import GeneralFilter

class Zoom(GeneralFilter):
    def __init__(self, params):
        super(Zoom, self).__init__(params)
        self.zooming_probability = self.filter_params[self.name]['probability']
        self.flow_index = self.filter_params[self.name]['flow_index']
        self.disp_indexis = self.filter_params[self.name]['disp_indexis']
        self.method = self.filter_params[self.name]['method']
        self.channels = self.filter_params[self.name]['channels']
        self._set_exceptions()

    def _set_name(self):
        return 'zooming_params'
    
    def _set_exceptions(self):
        if 'filters.region_remover.RegionRemover' in self.params['filters']:
            self.method = 'NN'
            print('Since SlowObjects filter is used, zooming method will be NN')

    def filter(self, list_of_samples):
        ''' Zoom images'''
        with tf.variable_scope('augmentation_zoom_filter'): 
            apply_zoom = tf.random_uniform([], 0, 1)
            zoomed_samples = tf.cond(apply_zoom <= self.zooming_probability, lambda: self.apply_zoom(list_of_samples), lambda: list_of_samples)
            return zoomed_samples

    def apply_zoom(self, batch):
        with tf.variable_scope('apply_zoom'):
            random_zoom = tf.random_uniform([], 1, 1.8)
            height, width, _ =  batch[0].get_shape().as_list()
            image_height= tf.cast(random_zoom * height, tf.int32) # zoomed height
            image_width=  tf.cast(random_zoom * width,  tf.int32) # zoomed width
            
            scalar_factor_height = tf.cast(image_height / height, tf.float32) 
            scalar_factor_width  = tf.cast(image_width / width, tf.float32)
            
            zoomed_samples =[]
            for index,sample in enumerate(batch):
                if self.method == 'AREA':
                    resized_image = tf.image.resize_images(sample, [image_height, image_width], tf.image.ResizeMethod.AREA)
                elif self.method == 'NN':
                    resized_image = tf.image.resize_images(sample, [image_height, image_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                else:
                    raise ValueError('resizing method not valid. AREA and NN expected')

                # gt must be scaled depending on zoom
                if index in self.disp_indexis:
                    resized_image = resized_image * scalar_factor_width
            
                if index in self.flow_index:
                    if self.channels == 2:
                        scalings = tf.ones(2, tf.float32) * [scalar_factor_width, scalar_factor_height]
                    else:
                        scalings = tf.ones(3, tf.float32) * [scalar_factor_width, scalar_factor_height, 1]
                    resized_image = resized_image * scalings

                # now, time for crop
                resized_image = tf.image.resize_image_with_crop_or_pad(resized_image, height, width)

                zoomed_samples.append(resized_image)
            
            return zoomed_samples
