'''
Factory for networks

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
import numpy as np
from sceneflow.dwarf import Dwarf as SYNTH_DWARF
from kitti.dwarf import Dwarf as KITTI_DWARF

NETWORK_FACTORY = {
    'sceneflow': SYNTH_DWARF,
    'kitti': KITTI_DWARF,
}

AVAILABLE_NETWORKS = NETWORK_FACTORY.keys()

def get_network(params):
    name = params['experiment']['factory']
    assert(name in AVAILABLE_NETWORKS)
    return NETWORK_FACTORY[name]