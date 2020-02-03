'''
Testing writer for Kitti. It produces a 16 bit images for
disparity, disparity change and optical flow

Author: Filippo Aleotti

Mail: filippo.aleotti2@unibo.it
'''

from general.writer import GeneralWriter
from main_utils.disp_utils import save_kitti_disp
from main_utils.flow_utils import save_kitti_flow
from main_utils.system import create_dir, cv_save
from kitti.test.ops import *
import os

class Writer(GeneralWriter):
    
    def __init__(self, params):
        '''
            Prepare a folder valid for Kitti Sceneflow benchmark
        '''
        super(Writer, self).__init__(params)
        self.params = params
        self.kitti_writer_settings = self.writing_settings['kitti_writer']
        self.writing_path = os.path.join(self.kitti_writer_settings['output_dir'], self.model_name, 'KITTI')
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
            save_kitti_flow(motion, destination)
        
        elif type_of_motion == 'disparity_change':
            # have to reverse motion (since disparity change is predicted negative), 
            # and to discard negative values 
            destination = os.path.join(self.writing_path, 'disp_1', '{}.png'.format(name))
            motion = reverse_motion(motion)
            motion = discard_negative(motion)
            save_kitti_disp(motion, destination)
        else:
            # have to reverse motion (since disparity is predicted negative) 
            # and to discard negative values
            destination = os.path.join(self.writing_path, 'disp_0', '{}.png'.format(name)) 
            motion = reverse_motion(motion)
            motion = discard_negative(motion)
            save_kitti_disp(motion, destination)