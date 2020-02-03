'''
Utilities for handling PFM images

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import re
import numpy as np

def load_pfm(filename):
    '''
    Read a pfm flow map
    From https://github.com/JiaRenChang/PSMNet/blob/master/dataloader/readpfm.py
    '''
    color = None
    width = None
    height = None
    scale = None
    endian = None

    with open(filename, 'r', encoding="latin-1") as f:
        header = f.readline().rstrip()
        if header == 'PF':
            color = True    
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width, 1)

    return np.flipud(np.reshape(data, shape))