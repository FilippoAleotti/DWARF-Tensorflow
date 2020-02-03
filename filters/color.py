'''
Augment an image similarly as in Monodepth by Godard et Al. https://arxiv.org/pdf/1609.03677.pdf
https://github.com/mrharicot/monodepth/blob/master/monodepth_dataloader.py

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
from general.filter import GeneralFilter

class Color(GeneralFilter):
    def __init__(self, params):
        super(Color, self).__init__(params)
        self.excluded_samples = self.filter_params[self.name]['excluded_indexes']
        self.probability = self.filter_params[self.name]['probability']

    def _set_name(self):
        return 'coloring_params'
        
    def filter(self, list_of_samples):
        with tf.variable_scope('augmentation_color_filter'):
            apply_augmentation = tf.random_uniform([], 0, 1)
            augmented_samples = tf.cond(apply_augmentation <= self.probability, lambda: self.apply_color_augmentation(list_of_samples), lambda: list_of_samples)
            return augmented_samples

    def apply_color_augmentation(self, list_of_samples):
        ''' Apply Godard augmentation indipendently for each image'''
        with tf.variable_scope('apply_filter'):
            list_of_augmented_samples = []
            for index, sample in enumerate(list_of_samples):
                if index in self.excluded_samples:
                    list_of_augmented_samples.append(sample)
                    continue

                random_gamma = tf.random_uniform([], 0.8, 1.2)
                random_brightness = tf.random_uniform([], 0.5, 2.0)
                random_colors = tf.random_uniform([3], 0.8, 1.2)
                white = tf.ones([tf.shape(list_of_samples[0])[0], tf.shape(list_of_samples[0])[1]])
                color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
                augmented_image = sample  ** random_gamma
                augmented_image =  augmented_image * random_brightness 
                augmented_image = augmented_image * color_image 
                augmented_image = tf.clip_by_value(augmented_image,  0, 1)
                list_of_augmented_samples.append(augmented_image)

            return list_of_augmented_samples