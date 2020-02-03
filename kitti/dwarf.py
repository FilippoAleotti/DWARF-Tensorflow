'''
Dwarf for fine-tuning on KITTI.
Predictions are up-scaled to full-resolution 

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
from sceneflow.dwarf import Dwarf as BaseDwarf
from main_utils.ops import masked_l1_loss, regularization_term
from main_utils.flow_utils import extract_flow_mask
from main_utils.disp_utils import extract_disparity_mask

class Dwarf(BaseDwarf):
    def build_losses(self):
        ''' 
            Losses used by the network.
            The loss of each level is firstly upsampled
            to full_res (the resolution of the crop),
            then is compared with the gt.
            Invalid pixel are not considered while measuring the
            error.
        '''
        with tf.variable_scope('losses'):
            self.total_loss = 0.
            self.decoders_loss = []
            full_size = [self.height, self.width]
            
            self.ground_truths, self.validity_masks = self.extract_motions_and_masks(self.gt_motion_vectors[0])
            self.upsampled_motions = []
            
            for i in range(self.starting_index):
                print('level:{} , loss_weight:{}'.format(i,self.alphas[i]))
                self.decoders_loss.append(0)
                self.upsampled_motions.append(None)

            for i in range(self.starting_index, self.number_of_scales):
                with tf.variable_scope('upsampling_to_fullres'):
                    upsampled_flow = tf.image.resize_bilinear(self.predicted_motion_vectors[i]['forward_flow'], full_size)
                    upsampled_disparity = tf.image.resize_bilinear(self.predicted_motion_vectors[i]['disparity'], full_size)
                    upsampled_disparity_change =  tf.image.resize_bilinear(self.predicted_motion_vectors[i]['disparity_change'],full_size)
                    up_motion = {
                        'disparity': upsampled_disparity,
                        'disparity_change': upsampled_disparity_change,
                        'forward_flow': upsampled_flow
                    }
                    self.upsampled_motions.append(up_motion)
                     
                loss_optical_flow = masked_l1_loss(upsampled_flow,
                    self.ground_truths[i]['forward_flow'], self.validity_masks[i]['forward_flow'], self.alphas[i])
                loss_disparity  = masked_l1_loss(upsampled_disparity, 
                    self.ground_truths[i]['disparity'], self.validity_masks[i]['disparity'], self.alphas[i])
                loss_disparity_change = masked_l1_loss(upsampled_disparity_change, 
                    self.ground_truths[i]['disparity_change'], self.validity_masks[i]['disparity_change'], self.alphas[i])
                loss = self.optical_flow_weight * loss_optical_flow + loss_disparity + loss_disparity_change

                self.total_loss += loss
                self.decoders_loss.append(loss)

                if i == self.starting_index:
                    self.loss_disparity = loss_disparity
                    self.loss_disparity_change = loss_disparity_change
                    self.loss_optical_flow = loss_optical_flow * self.optical_flow_weight
            
            self.regularization_loss = regularization_term(self.gamma) 
            self.total_loss += self.regularization_loss
    
    def extract_motions_and_masks(self, gt_motion_vectors):
        '''
            Given a set of motions, this functions extract the
            validity mask for each motion and also the real motion
            values (e.g., flow are stored as 3 channels maps, but just
            1 and 2 channels are usefull)
            Returns:
                ground_truths: usefull motions
                masks: validity mask for each motion
        '''
        with tf.variable_scope('extract_motions_and_masks'):
            ground_truths = []
            masks = []
            for i in range(self.starting_index):
                ground_truths.append(None)
                masks.append(None)
            for i in range(self.starting_index, self.number_of_scales):

                gt_disparity, disparity_mask = extract_disparity_mask(gt_motion_vectors['disparity'])
                gt_disparity_change, disparity_change_mask = extract_disparity_mask(gt_motion_vectors['disparity_change'])
                gt_forward_flow, flow_mask = extract_flow_mask(gt_motion_vectors['forward_flow'])
                masks.append(
                    {
                        'forward_flow':flow_mask,
                        'disparity': disparity_mask,
                        'disparity_change': disparity_change_mask
                    }
                )
                ground_truths.append(
                    {
                        'forward_flow':gt_forward_flow,
                        'disparity': gt_disparity,
                        'disparity_change': gt_disparity_change
                    }
                )
        return ground_truths, masks

    def build_summaries(self):
        ''' Summaries'''
        with tf.variable_scope('summaries'):
            
            with tf.variable_scope('ground_truth'):
                self.colored_gt = self.color_motion_vectors(self.ground_truths, self.scale_factor)
            
            with tf.variable_scope('predicted_sceneflow'):
                self.color_predicted_motion_vectors_upsampled = self.color_motion_vectors(self.upsampled_motions, self.scale_factor)
                self.color_predicted_motion_vectors = self.color_motion_vectors(self.predicted_motion_vectors, self.scale_factor)
                self.error_images = self.create_error_images(self.upsampled_motions, self.ground_truths, self.validity_masks)

            for i in range(self.starting_index, self.number_of_scales):
                with tf.variable_scope('level_'+str(i)):
                    with tf.variable_scope('predictions'):
                        tf.summary.image('disparity',   self.color_predicted_motion_vectors[i]['disparity'],  max_outputs=self.max_output)
                        tf.summary.image('disparity_change', self.color_predicted_motion_vectors[i]['disparity_change'],  max_outputs=self.max_output)
                        tf.summary.image('flow',  self.color_predicted_motion_vectors[i]['forward_flow'],  max_outputs=self.max_output)
                        tf.summary.scalar('decoder_loss', self.decoders_loss[i])

                if i == self.starting_index:
                    with tf.variable_scope('errors'):
                        tf.summary.image('flow', self.error_images[i]['forward_flow'], max_outputs=self.max_output)
                        tf.summary.image('disparity', self.error_images[i]['disparity'], max_outputs=self.max_output)
                        tf.summary.image('disparity_change', self.error_images[i]['disparity_change'], max_outputs=self.max_output)
                    with tf.variable_scope('full_resolution'):
                        tf.summary.image('disparity',   self.color_predicted_motion_vectors_upsampled[i]['disparity'],  max_outputs=self.max_output)
                        tf.summary.image('disparity_change', self.color_predicted_motion_vectors_upsampled[i]['disparity_change'],  max_outputs=self.max_output)
                        tf.summary.image('flow',  self.color_predicted_motion_vectors_upsampled[i]['forward_flow'],  max_outputs=self.max_output)                                                

            with tf.variable_scope('validity_mask'):
                tf.summary.image('disparity',   self.validity_masks[self.starting_index]['disparity'],  max_outputs=self.max_output)
                tf.summary.image('disparity_change', self.validity_masks[self.starting_index]['disparity_change'],  max_outputs=self.max_output)
                tf.summary.image('flow',  self.validity_masks[self.starting_index]['forward_flow'],  max_outputs=self.max_output)
            
            with tf.variable_scope('ground_truth'):
                tf.summary.image('disparity',   self.colored_gt[self.starting_index]['disparity'],  max_outputs=self.max_output)
                tf.summary.image('disparity_change', self.colored_gt[self.starting_index]['disparity_change'],  max_outputs=self.max_output)
                tf.summary.image('flow',  self.colored_gt[self.starting_index]['forward_flow'],  max_outputs=self.max_output)
            
            with tf.variable_scope('images'):
                tf.summary.image('left_t0',   self.left_t0,  max_outputs=self.max_output)
                tf.summary.image('left_t1',   self.left_t1,  max_outputs=self.max_output)
                tf.summary.image('right_t0',  self.right_t0, max_outputs=self.max_output)
                tf.summary.image('right_t1',  self.right_t1, max_outputs=self.max_output)

            with tf.variable_scope('losses'):
                tf.summary.scalar('loss_optical_flow', self.loss_optical_flow)
                tf.summary.scalar('loss_disparity', self.loss_disparity)
                tf.summary.scalar('loss_disparity_change', self.loss_disparity_change)
                tf.summary.scalar('regularization' , self.regularization_loss)
                
            tf.summary.scalar('gamma' , self.gamma)
            tf.summary.scalar('max_displacement' , self.max_displacement)
            tf.summary.scalar('optical_flow_weight' , self.optical_flow_weight)
            tf.summary.scalar('scale_factor' , self.scale_factor)            
