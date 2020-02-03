'''
Adam optimizer with changes in learning rate at specific breakpoints

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
from general.optimizer import GeneralOptimizer

class Optimizer(GeneralOptimizer):
    def __init__(self, params, global_step):
        super(Optimizer, self).__init__(params, global_step)
        self.learning_rate = self._get_learning_rate()
        self.instance = tf.train.AdamOptimizer(self.learning_rate)
    
    def _get_learning_rate(self):
        boundaries = self.params['training']['optimizer']['steps']
        boundaries = [int(boundary) for boundary in boundaries]
        values = [ self.initial_learning_rate / (2**i) for i in range(len(boundaries)+1)]
        learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
        return learning_rate