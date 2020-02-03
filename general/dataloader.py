'''
General Dataloader

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
from tensorflow.data import Dataset, Iterator
from main_utils.dataloader_utils import *
import multiprocessing
import numpy as np
from filters.filter_manager import FilterManager

class GeneralLoader(object):

    def __init__(self, params):
        self.params = params
        self.mode = params['experiment']['mode']
        self.filename_file = params[self.mode]['dataloader']['filenames']
        self.number_of_paths_in_line = get_number_of_paths(self.filename_file)
        self.data_path = self._get_data_path()
        self.number_samples = get_num_samples(params[self.mode]['dataloader']['filenames'])
        self.set_params()
        self.dataset = self.prepare_dataset()
        
        if self.mode == 'training':
            self.iterator =self.dataset.make_initializable_iterator()
        else:
            self.iterator = self.dataset.make_one_shot_iterator()

    def _get_data_path(self):
        data_paths = self.params[self.mode]['dataloader']['data_paths']
        selected_data_path = self.params[self.mode]['dataloader']['selected_data_path']
        if selected_data_path not in data_paths:
            raise ValueError("Not a valid data path selected")
        return data_paths[selected_data_path]

    def set_params(self):
        self.batch_size = 1
        if self.mode == 'training':
            self.batch_size = self.params[self.mode]['dataloader']['batch_size']
            self.total_epochs, self.total_steps = self.get_number_of_epochs_and_steps()

    def prepare_dataset(self):
        with tf.variable_scope('prepare_dataset'):
            dataset = tf.data.TextLineDataset(self.filename_file)
            if self.mode == 'training':
                dataset = dataset.shuffle(buffer_size=self.number_samples)
                dataset = dataset.map(self.prepare_batch, num_parallel_calls=multiprocessing.cpu_count())
                dataset = dataset.repeat()
                dataset = dataset.prefetch(30)
            else:
                dataset = dataset.map(self.prepare_batch)
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset

    def prepare_batch(self, x):
        ''' Set here the behaviour of the dataloader'''
        with tf.variable_scope("Dataloader"):
            batch = self.load_images(x)
            batch = self.apply_filters(batch)
        return batch

    def get_next_batch(self):
        ''' Return the next batch. Override to show portion of the full batch '''
        with tf.variable_scope('get_next_batch'):
            self.batch = self.iterator.get_next()
    
    def initialize(self, session):
        with tf.variable_scope('initialize'):
            session.run(self.iterator.initializer)

    def get_number_of_epochs_and_steps(self):
        '''
            This dataloader needs the number of epochs.
            This functions read the number of epochs if
            defined in params, otherwise it get this number
            from steps
        '''
        with tf.variable_scope('get_epochs'):
            iterations = self.params['training']['iterations']
            epochs = self.params['training']['epochs']
            use_iteration, use_epoch = False, False

            if iterations >= 0:
                steps_per_epoch = np.ceil(
                    self.number_samples / self.batch_size).astype(np.int32)
                total_epochs = iterations / steps_per_epoch
                use_iteration = True
                total_steps = iterations

            if epochs >= 0:
                use_epoch = True
                total_epochs = epochs
                steps_per_epoch = np.ceil(self.num_samples / self.batch_size).astype(np.int32)
                total_steps = epochs * steps_per_epoch
            
            if use_epoch is False and use_iteration is False:
                raise ValueError(
                    'No stop criteria found iterations: {}, epochs: {} '.format(iterations, epochs))

            if use_epoch is True and use_iteration is True:
                raise ValueError(
                    'Both criteria found iterations: {}, epochs: {} '.format(iterations, epochs))
            return int(total_epochs), int(total_steps)
    
    def apply_filters(self, samples):
        ''' Apply all the added filters '''
        with tf.variable_scope('apply_filters'):
                
            filter_manager = FilterManager(self.params)
            filters = filter_manager.get_filters()

            if not filters:
                return samples

            for f in filters:
                output = f.filter(samples)
                samples = output

            return output
    
    def __len__(self):
        return self.number_samples
