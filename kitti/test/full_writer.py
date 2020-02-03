'''
Testing writer for SceneFlow. It calls both visual_writer and kitti_writer

Author: Filippo Aleotti

Mail: filippo.aleotti2@unibo.it
'''

from general.writer import GeneralWriter
from kitti.test.visual_writer import Writer as VisualWriter
from kitti.test.kitti_writer import Writer as KittiWriter

class Writer(GeneralWriter):
    
    def __init__(self, params):
        super(Writer, self).__init__(params) 
        self.kitti_writer = KittiWriter(params)
        self.visual_writer = VisualWriter(params)
        self.kitti_writer
        self.visual_writer
    
    def write(self, results):
        self.kitti_writer.write(results)
        self.visual_writer.write(results)