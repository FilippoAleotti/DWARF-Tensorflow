'''
General optimizer

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
from abc import ABCMeta, abstractmethod

class GeneralOptimizer(object, metaclass=ABCMeta):
    def __init__(self, params, global_step):
        self.params = params
        self.initial_learning_rate = params['training']['optimizer']['initial_learning_rate']
        self.global_step = global_step