'''
Utilities for handling file system

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''
import cv2
import os
from shutil import copyfile

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def cv_save(rgb_image, dest):
    '''
    Save a RGB image with opencv
    '''
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dest, bgr_image)

def bgr_to_rgb(image):
    '''
    Convert to RGB color space an image or a list of BGR images
    Params:
        image: list of images or a single image
    Returns:
        an image if a single image is given, a list of images otherwise
    '''
    def _to_rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    if isinstance(image, (list,)):
        return [_to_rgb(img) for img in image]
    else:
        return _to_rgb(image)

def import_service(service_name):
    """
        Import service by its fully qualified name.
        From http://python-dependency-injector.ets-labs.org/containers/dynamic.html
    """

    path_services = service_name.split('.')
    service = __import__('.'.join(path_services[:-1]),locals(),globals(),fromlist=path_services[-1:])
    return getattr(service, path_services[-1])

def copy_configuration(dest, cfg):
    '''
        Copy a configuration file into dest folder.
        If dest not exists, it will be created
    '''
    create_dir(dest)
    cfg_name = os.path.basename(cfg)
    copyfile(cfg, os.path.join(dest, cfg_name))