'''
Dataloader for DWARF

It load 4 frames and 3 grount truths: 

left_t0 (png), right_t0 (png), left_t1(png), right_t1(png), disp_t0_gt(pfm), 
disp_t01_gt(pfm), forward_flow_gt(pfm) 

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
from main_utils.flow_utils import tf_load_flo
from main_utils.disp_utils import tf_load_disparity_pfm
from general.dataloader import GeneralLoader
from main_utils.dataloader_utils import *

class Loader(GeneralLoader):

    def set_params(self):
        super(Loader, self).set_params()
        self.is_disparity_change_residual = self.params['training']['dataloader']['is_disparity_change_residual']

    def load_images(self, line):
        ''' Load images and ground truths'''
        with tf.variable_scope('load_images'):
            self.image_paths = get_image_paths_from_line(
                line, self.data_path, self.number_of_paths_in_line)

            images = [read_image(self.image_paths[i]) for i in range(4)]
            gt_disp_t0 = tf_load_disparity_pfm(self.image_paths[4])
            gt_disp_t01 = tf_load_disparity_pfm(self.image_paths[5])

            if self.is_disparity_change_residual:
                gt_disp_t01 += gt_disp_t0
            
            gt_forward_flow = tf_load_flo(self.image_paths[6])
            
            images.append(gt_disp_t0)
            images.append(gt_disp_t01)
            images.append(gt_forward_flow)

        return images

    def get_next_batch(self):
        ''' Return the next batch '''
        with tf.variable_scope('get_next_batch'):
            super(Loader, self).get_next_batch()
            batch = {}
            batch['left_t0'] = self.batch[0]
            batch['right_t0'] = self.batch[1]
            batch['left_t1'] = self.batch[2]
            batch['right_t1'] = self.batch[3]
            batch['gt_disparity'] = self.batch[4]
            batch['gt_disparity_change'] = self.batch[5]
            batch['gt_flow'] = self.batch[6]
        return batch