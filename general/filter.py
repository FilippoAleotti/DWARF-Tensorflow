'''
Generic filter for dataloader 

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
from abc import  abstractmethod, ABCMeta

class GeneralFilter(object, metaclass=ABCMeta):
    def __init__(self, params):
        self.params = params
        self.filter_params = params['filters_settings']
        self.name = self._set_name()
        
    @abstractmethod
    def filter(self, list_of_samples):
        ''' Define here the behaviour of the filter '''
        pass
    
    @abstractmethod
    def _set_name(self):
        pass

    def key_exists(self, key):
        return key in self.filter_params[self.name].keys()
    
    def get_key_or_default(self, key, default):
        if self.key_exists(key):
            return self.filter_params[self.name][key]
        return default