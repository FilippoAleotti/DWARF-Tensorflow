'''
Factory for writer

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
import numpy as np
from sceneflow.test.tester import Tester as SCENEFLOW_TESTER
from kitti.test.tester import Tester as KITTI_TESTER

TESTER_FACTORY = {
    'sceneflow': SCENEFLOW_TESTER,
    'kitti': KITTI_TESTER,
}

AVAILABLE_TESTERS = TESTER_FACTORY.keys()

def get_tester(params):
    name = params['experiment']['factory']
    assert(name in AVAILABLE_TESTERS)
    return TESTER_FACTORY[name]