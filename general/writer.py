'''
General writer

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it

'''

import os
from main_utils.system import create_dir

class GeneralWriter(object):

    def __init__(self, params):
        self.params = params
        self.mode = params['experiment']['mode']
        self.writing_settings = params[self.mode]['writing_settings']
        self.writing_path = None
        self.model_name = params['experiment']['network_name']
        
    def write(self, results):
        pass

    def _create_dir(self):
        if self.writing_path is not None:
            create_dir(self.writing_path)