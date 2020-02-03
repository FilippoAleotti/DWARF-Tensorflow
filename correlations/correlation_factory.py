'''
Get correlation to use
'''

from external_packages import correlation1D as cuda_corr1, correlation2D as cuda_corr2, correlation3D as cuda_corr3
from correlations import correlation1D as tf_corr1, correlation2D as tf_corr2, correlation3D as tf_corr3

default = 'tf'

def get_correlation1D():
    if default == 'tf':
        print('selected tf correlation 1D')
        return tf_corr1.correlation1D
    else:
        return cuda_corr1.correlation1d

def get_correlation2D():
    if default == 'tf':
        print('selected tf correlation 2D')
        return tf_corr2.correlation2D
    else:
        return cuda_corr2.ops.correlation

def get_correlation3D():
    if default == 'tf':
        print('selected tf correlation 3D')
        return tf_corr3.correlation3D
    else:
        return cuda_corr3.ops.correlation3D
