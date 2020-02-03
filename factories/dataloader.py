'''
Factory for dataloaders

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
import numpy as np
from sceneflow.training.dataloader import Loader as SYNTH_LOADER
from kitti.training.dataloader import Loader as KITTI_LOADER
from sceneflow.test.dataloader import Loader as SF_TESTING_LOADER
from kitti.test.dataloader import Loader as KITTI_TESTING_LOADER

DATALOADER_FACTORY_TRAIN = {
    'sceneflow': SYNTH_LOADER,
    'kitti': KITTI_LOADER,
}

DATALOADER_FACTORY_TEST = {
    'sceneflow': SF_TESTING_LOADER ,
    'kitti': KITTI_TESTING_LOADER,
}

AVAILABLE_DATALOADER_TRAIN = DATALOADER_FACTORY_TRAIN.keys()
AVAILABLE_DATALOADER_TEST =  DATALOADER_FACTORY_TEST.keys()

def get_dataloader(params):
    name = params['experiment']['factory']
    if params['experiment']['mode'] == 'training':
        assert(name in AVAILABLE_DATALOADER_TRAIN)
        return DATALOADER_FACTORY_TRAIN[name]
    elif params['experiment']['mode'] == 'testing':
        assert(name in AVAILABLE_DATALOADER_TEST)
        return DATALOADER_FACTORY_TEST[name]
    raise ValueError('Not valid mode. Expected training or testing')