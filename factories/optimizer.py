'''
Factory for optimizers

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
import numpy as np
from sceneflow.training.optimizer import Optimizer

OPTIMIZER_FACTORY = {
    'sceneflow': Optimizer,
    'kitti': Optimizer,
}

AVAILABLE_OPTIMIZERS = OPTIMIZER_FACTORY.keys()

def get_optimizer(params):
    name = params['experiment']['factory']
    assert(name in AVAILABLE_OPTIMIZERS)
    return OPTIMIZER_FACTORY[name]