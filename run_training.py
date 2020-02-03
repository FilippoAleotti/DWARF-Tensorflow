'''
Training script

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
from main_utils.helper import load_settings
from factories import dataloader as DataloaderFactory, network as NetworkFactory, optimizer as OptimizerFactory
import argparse
import os
import numpy as np
import time
from main_utils.system import copy_configuration
from main_utils.time import add_hours_to_date, get_formatted_date

parser = argparse.ArgumentParser(description='Run a training')
parser.add_argument('--cfg', type=str, required=True, help='path to a specific configuration')
args = parser.parse_args()

def train(params):

    global_step = tf.Variable(0, trainable=False)
    params['experiment']['mode'] = 'training' 
    optimizer = OptimizerFactory.get_optimizer(params)(params, global_step)

    dataloader = DataloaderFactory.get_dataloader(params)(params)
    next_batch = dataloader.get_next_batch()
    network = NetworkFactory.get_network(params)(params, next_batch)
    minimize_op = optimizer.instance.minimize(network.total_loss, global_step=global_step)
    tf.summary.scalar('learning_rate', optimizer.learning_rate)
    summary_op = tf.summary.merge_all()

    config = tf.ConfigProto(allow_soft_placement=True)
    session = tf.Session(config=config)
    dest_folder = os.path.join(params['training']['logs']['log_directory'], params['experiment']['network_name'])
    summary_writer = tf.summary.FileWriter(dest_folder, session.graph)
    copy_configuration(dest_folder, args.cfg)
    train_saver = tf.train.Saver()

    total_num_parameters = 0
    for variable in tf.trainable_variables():
        total_num_parameters += np.array(variable.get_shape().as_list()).prod()
    print("number of trainable parameters: {}".format(total_num_parameters))
    
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    
    ckpt_path = params['training']['selected_checkpoint']
    ckpts = params['training']['checkpoint_paths']
    if ckpt_path != None:
        if ckpt_path not in ckpts:
            raise ValueError('Not a valid checkpoint')
        train_saver.restore(session, ckpts[ckpt_path])

    dataloader.initialize(session)
    
    if params['training']['retrain']:
        session.run(global_step.assign(0))

    starting_step = global_step.eval(session=session)
    params['total_number_of_iterations'] = dataloader.total_steps
    starting_time = time.time()
    
    for step in range(starting_step, dataloader.total_steps):
        before_op_time = time.time()
        _, loss_value = session.run([minimize_op, network.total_loss])
        duration = time.time() - before_op_time
        if step and step % params['training']['logs']['summary_step'] == 0:
            update_monitor_summaries(params, starting_time, loss_value, step, duration)
            summary_str = session.run(summary_op)
            summary_writer.add_summary(summary_str, global_step=step)
        if step and step % params['training']['saving_step'] == 0:
            train_saver.save(session, params['training']['logs']['log_directory'] + '/' + params['experiment']['network_name'] + '/model', global_step=step)

    train_saver.save(session, params['training']['logs']['log_directory'] + '/' + params['experiment']['network_name'] + '/model', global_step=dataloader.total_steps)

def update_monitor_summaries(params, start_time, loss, current_iteration, duration):
    ''' 
        Write summaries to console
    '''
    examples_per_sec = params['training']['dataloader']['batch_size'] / duration
    time_sofar = (time.time() - start_time) / 3600
    
    training_hours_left = ((params['total_number_of_iterations'] - current_iteration)/examples_per_sec)* params['training']['dataloader']['batch_size']/3600.0
    expected_ending = add_hours_to_date(training_hours_left)

    print_string = 'step {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h | expected ending: {}'
    print(print_string.format(current_iteration, examples_per_sec, loss, time_sofar, training_hours_left, get_formatted_date(expected_ending)))


if __name__ == '__main__':
    params = load_settings(args.cfg)
    train(params)
