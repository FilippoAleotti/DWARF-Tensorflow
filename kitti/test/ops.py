import numpy as np

def reverse_motion(motion):
    '''
        Some motions of the Network (disparity, disp. change)
        are predicted as negative values.
        However, benchmark as KITTI expect positive values, so
        the predictions must be reversed  
    '''
    return motion * -1.

def discard_negative(motion):
    '''
        Discard negative predictions
    '''
    return motion.clip(min=0)

def flip_motion(motion):
    '''
        Flip left-to-right the motion
    '''
    return np.fliplr(motion)