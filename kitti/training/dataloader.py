'''
Dataloader for DWARF

It load 4 frames and 3 grount truths: 

left_t0 (png), right_t0 (png), left_t1(png), right_t1(png), disp_t0_gt(16 bit), 
disp_t01_gt(png 16 bit), forward_flow_gt(png 16 bit) 

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it

NOTE:
    disparity and disparity change are saved as positive numbers.
    The network instead predict negative values, so
    ground truth must be multiplied by -1 when loaded
'''

from sceneflow.training.dataloader import Loader as Dataloader
import tensorflow as tf
from main_utils.flow_utils import tf_load_16_bit_flow
from main_utils.disp_utils import tf_load_16_bit_disparity
from main_utils.dataloader_utils import *

class Loader(Dataloader):

    def load_images(self, line):
        ''' Load images and ground truths'''
        with tf.variable_scope('load_images'):
            self.image_paths = get_image_paths_from_line(
                line, self.data_path, self.number_of_paths_in_line)

            images = [read_image(self.image_paths[i]) for i in range(4)]
            
            gt_disp_t0 = tf_load_16_bit_disparity(self.image_paths[4])
            gt_disp_t01 = tf_load_16_bit_disparity(self.image_paths[5])

            gt_forward_flow = tf_load_16_bit_flow(self.image_paths[6])
            with tf.variable_scope('reverse_disparity_sign'):
                gt_disp_t0  *= -1.
                gt_disp_t01 *= -1

            images.append(gt_disp_t0)
            images.append(gt_disp_t01)
            images.append(gt_forward_flow)

        return images