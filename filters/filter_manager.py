'''
Filter manager: it loads and create Filter objects

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
from main_utils.system import import_service

class FilterManager(object):
    def __init__(self, params):
        self.params = params
        self.mode = params['experiment']['mode']
        self.filters = self.load_filters()
    
    def load_filters(self):
        filters = []
        filter_classes = self.params[self.mode]['filters']
        for filter_class in filter_classes:
            try:
                created_filter = import_service(filter_class)(self.params[self.mode])
                filters.append(created_filter)
            except Exception as e:
                print('Filter {} not loaded'.format(filter_class))
                print('Due to: {}'.format(e))
                raise ValueError

        return filters
    
    def get_filters(self):
        return self.filters
