'''
Generic tester 

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

from abc import ABCMeta, abstractmethod
from main_utils.system import import_service

class GeneralTester(object, metaclass=ABCMeta):

    def __init__(self, params):
        self.params = params
        self.setup()
        self.writers = self._import_writers()

    def setup(self):
        pass

    @abstractmethod
    def final_elaboration(self):
        ''' Insert code to test (eventually) the set of outputs '''
        pass

    @abstractmethod
    def handle_single_output(self, network_output, step):
        ''' 
            Insert code to handle a network output.
            You can memorize it or evaluate sample by sample
        '''
        pass
        
    def _import_writers(self):
        writers = []
        writer_classes = self.params['testing']['writers']

        if writer_classes is None:
            return writers

        for writer in writer_classes:
            try:
                created_writer = import_service(writer)(self.params)
                writers.append(created_writer)
            except Exception as e:
                print('Writer {} not loaded'.format(writer))
                print('Due to: {}'.format(e))
                raise ValueError

        return writers
    
    def run(self, session, network, dataloader):
        return session.run(network.outputs)