
'''
Tester for sceneflow dataset

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

from general.tester import GeneralTester
from collections import namedtuple
from main_utils.testing import evaluate
import numpy as np
from terminaltables import AsciiTable

Output = namedtuple('Output','predicted gt')
TestingResult = namedtuple('TestingResult', 'motion name step')

class Tester(GeneralTester):
    
    def setup(self):
        # each motion is made up by two values: EPE and BAD pixels
        self.optical_flows =  np.zeros((2), dtype=np.float32) 
        self.disp_t01 =  np.zeros((2), dtype=np.float32)
        self.disp_t0 =  np.zeros((2), dtype=np.float32)
        self.number_samples = 0

    def handle_single_output(self, network_output, step):
        flow = network_output.predicted['forward_flow']
        disp = network_output.predicted['disparity']
        disp_change = network_output.predicted['disparity_change']
        self.optical_flows = self.evaluate_results(flow, network_output.gt['forward_flow'], self.optical_flows, step)
        self.disp_t0 = self.evaluate_results(disp, network_output.gt['disparity'], self.disp_t0, step)
        self.disp_t01 = self.evaluate_results(disp_change, network_output.gt['disparity_change'], self.disp_t01, step)
        self.number_samples += 1
        
        for writer in self.writers:
            writer.write(TestingResult(motion=flow, name='flow', step=step))
            writer.write(TestingResult(motion=disp, name='disparity', step=step))
            writer.write(TestingResult(motion=disp_change, name='disparity_change', step=step))
    
    def evaluate_results(self, predicted, gt, accumulator, step):
        epe_error, outliers = evaluate(predicted, gt, step)
        accumulator = np.add(accumulator, np.asarray([epe_error, outliers]))
        return accumulator
    
    def final_elaboration(self):
        results = {
            'flow':{
                'epe': '{:5f}'.format(self.optical_flows[0] / max(self.number_samples, 1.)),
                'bad': '{:5f}'.format(self.optical_flows[1] / max(self.number_samples, 1.))
            },
            'disp':{
                'epe': '{:5f}'.format(self.disp_t0[0] / max(self.number_samples, 1.)),
                'bad': '{:5f}'.format(self.disp_t0[1] / max(self.number_samples, 1.))
            },
            'disp_change':{
                'epe': '{:5f}'.format(self.disp_t01[0] / max(self.number_samples, 1.)),
                'bad': '{:5f}'.format(self.disp_t01[1] / max(self.number_samples, 1.))
            }
        }
        table_data = [
            ['FLOW EPE','FLOW F1', 'DISP EPE', 'DISP D1', 'DISP CHANGE EPE', 'DISP CHANGE D1'],
            [results['flow']['epe'],results['flow']['bad'], results['disp']['epe'], results['disp']['bad'], results['disp_change']['epe'],results['disp_change']['bad']],
        ]
        table = AsciiTable(table_data)
        print(table.table)
        return results
    
    def run(self, session, network, dataloader):
        batch = dataloader.batch
        motions, gt_disp, gt_disp_change, gt_flow = session.run([network.outputs, batch[4], batch[5], batch[6]])
        gt = {
            'forward_flow': np.squeeze(gt_flow, axis=0),
            'disparity': np.squeeze(gt_disp, axis=0),
            'disparity_change': np.squeeze(gt_disp_change, axis=0)
        }
        return Output(predicted=motions, gt=gt)