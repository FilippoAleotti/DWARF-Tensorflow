'''
Testing for KITTI Benchmark

Author: Filippo Aleotti

Mail: filippo.aleotti2@unibo.it
'''

from general.tester import GeneralTester
from collections import namedtuple
import numpy as np

TestingResult = namedtuple('TestingResult', 'motion name step')

class Tester(GeneralTester):

    def handle_single_output(self, network_output, step):
        '''
            Kitti test suite expect a folder with 16 bit
        '''
        disp = network_output['disparity']
        disp_change = network_output['disparity_change']
        flow = network_output['forward_flow']
        
        for writer in self.writers:
            writer.write(TestingResult(motion=flow, name='flow', step=step))
            writer.write(TestingResult(motion=disp, name='disparity', step=step))
            writer.write(TestingResult(motion=disp_change, name='disparity_change', step=step))
    
    def final_elaboration(self):
        pass
