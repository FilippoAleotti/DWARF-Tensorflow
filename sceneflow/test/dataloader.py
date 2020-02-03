'''
Dataloader for testing purposes
It load 4 frames: 
left_t0 (png), right_t0 (png), left_t1(png), right_t1(png)

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
from main_utils.flow_utils import tf_load_flo
from main_utils.disp_utils import tf_load_disparity_pfm
from main_utils.dataloader_utils import *
from general.dataloader import GeneralLoader

class Loader(GeneralLoader):

    def load_images(self, line):
        ''' Load images and ground truths'''

        self.image_paths = get_image_paths_from_line(
            line, self.data_path, self.number_of_paths_in_line)

        images = [read_image(self.image_paths[i]) for i in range(4)]
        
        gt_disp_t0 = tf_load_disparity_pfm(self.image_paths[4])
        gt_disp_t01 = tf_load_disparity_pfm(self.image_paths[5])
        
        # disparity change is residual, make it complete
        gt_disp_t01 += gt_disp_t0

        gt_forward_flow = tf_load_flo(self.image_paths[6])

        images.append(gt_disp_t0)
        images.append(gt_disp_t01)
        images.append(gt_forward_flow)
        
        return images
    
    def get_next_batch(self):
        ''' Return the next batch '''
        super(Loader, self).get_next_batch()
        batch = {}
        batch['left_t0'] = self.batch[0]
        batch['right_t0'] = self.batch[1]
        batch['left_t1'] = self.batch[2]
        batch['right_t1'] = self.batch[3]
        batch['final_shape'] = tf_get_height_width(self.batch[4]) # original height and width
        return batch