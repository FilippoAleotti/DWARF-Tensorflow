'''
Testing script

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
from main_utils.helper import load_settings
from factories import dataloader as DataloaderFactory, network as NetworkFactory, tester as TesterFactory
from main_utils.dataloader_utils import get_num_samples
import argparse
import os
import numpy as np
import time
from tqdm import tqdm
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

parser = argparse.ArgumentParser(description='Run a test')
parser.add_argument('--cfg', type=str, required=True, help='path to a specific configuration')
parser.add_argument('--cuda-off', action='store_true', help='run test w/o cuda')
args = parser.parse_args()

def test(params):
    if args.cuda_off:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        print('WARNING: Cuda Disabled')  

    tf.reset_default_graph()
    params['experiment']['mode'] = 'testing'
    dataloader = DataloaderFactory.get_dataloader(params)(params)
    next_batch = dataloader.get_next_batch()
    network = NetworkFactory.get_network(params)(params, next_batch)
    loader = tf.train.Saver()

    config = tf.ConfigProto(allow_soft_placement=True)
    session = tf.Session(config=config)

    total_num_parameters = 0
    for variable in tf.trainable_variables():
        total_num_parameters += np.array(variable.get_shape().as_list()).prod()
    print("number of trainable parameters: {}".format(total_num_parameters))

    ckpt_path = params['testing']['selected_checkpoint']
    ckpts = params['testing']['checkpoint_paths']
    if ckpt_path == None:
        raise ValueError('Expected checkpoint path but it is None')
    if ckpt_path not in ckpts:
        raise ValueError('Not a valid checkpoint')
    loader.restore(session, ckpts[ckpt_path])
    
    tester = TesterFactory.get_tester(params)(params)
    num_test_samples = get_num_samples(params['testing']['dataloader']['filenames'])

    with tqdm(total=num_test_samples) as bar:
        for step in range(num_test_samples):
            results = tester.run(session, network, dataloader)
            tester.handle_single_output(results, step)
            bar.update(1)

    tester.final_elaboration()
    print('Done!')

if __name__ == '__main__':
    params = load_settings(args.cfg)
    test(params)