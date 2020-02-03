'''
Testing writer. It produces color images for
disparity, disparity change and optical flow

Author: Filippo Aleotti

Mail: filippo.aleotti2@unibo.it
'''

from general.writer import GeneralWriter
from main_utils.disp_utils import colormap_jet
from main_utils.flow_utils import flow_to_image
from main_utils.system import create_dir, cv_save
import os
import numpy as np
import cv2
from kitti.test.ops import *

class Writer(GeneralWriter):
    
    def __init__(self, params):
        '''
            Prepare a folder valid for Kitti Sceneflow benchmark
        '''
        super(Writer, self).__init__(params)
        self.visual_writer = self.writing_settings['visual_writer']
        self.writing_path = os.path.join(self.visual_writer['output_dir'],self.model_name,'visual')
        self.disparity_color_factor =self.visual_writer['disparity_color_factor']
        self.prepare_folders()

    def prepare_folders(self):
        folders = ['flow','disp_0','disp_1']
        for f in folders:
            create_dir(os.path.join(self.writing_path, f))

    def write(self, testing_result):
        '''
            Save in final folder the motion, as expected by Kitti SceneFlow benchmark
            This function expects in order:
                - motion: optical flow, disparity or disparity change
                - type of motion: string 'flow', 'disparity_change' or 'disparity', according with motion
                - step: current testing step
        '''
        motion = testing_result.motion
        type_of_motion = testing_result.name
        step = testing_result.step
        name = str(step).zfill(6) + "_10"

        if type_of_motion == 'flow':
            destination = os.path.join(self.writing_path, 'flow', '{}.png'.format(name))
            motion = flow_to_image(motion)
            cv_save(motion, destination)
        
        elif type_of_motion == 'disparity_change':
            destination = os.path.join(self.writing_path, 'disp_1', '{}.png'.format(name))
            motion = reverse_motion(motion)
            motion = discard_negative(motion)
            motion = colormap_jet(motion, self.disparity_color_factor)
            cv_save(motion, destination)

        elif type_of_motion == 'disparity':
            destination = os.path.join(self.writing_path, 'disp_0', '{}.png'.format(name))
            motion = reverse_motion(motion)
            motion = discard_negative(motion)
            motion = colormap_jet(motion, self.disparity_color_factor)
            cv_save(motion, destination)

        else:
            raise ValueError("Type of motion {} not found. Valid motion are flow, disparity and disparity_change")