import numpy as np
import numpy.ma as ma

def epe_error(x1, x2):
    '''
        Evaluates the average endpoint error between flow batches.
    '''
    diff = np.linalg.norm(x1-x2, axis=-1)
    error = np.sum(diff) / x1[:,:,0].size
    return error

def outlier_ratio(gt, predicted, threshold, step):
    if np.where(np.isinf(gt),1,0).sum() > 0:
        print('THERE IS AN INF IN GT')
    if np.where(np.isinf(predicted),1,0).sum() > 0:
        print('THERE IS AN INF IN PRED')
    if np.where(np.isnan(gt),1,0).sum() > 0:
        print('THERE IS AN NAN IN GT')
    if np.where(np.isnan(predicted),1,0).sum() > 0:
        print('THERE IS AN NAN IN PRED')
    diff =  np.linalg.norm(predicted-gt, axis=-1)
    bad_pixels = diff >= threshold
    d1 = 100.0 * bad_pixels.sum() / gt[:,:,0].size
        
    return d1

def evaluate(predicted, gt, step, threshold=3, relative=0.05, mask=None):
    '''
        Evaluate otpical flow.
        Params:
            predicted: prediction of the alghoritm [HxWxN]
            gt: ground truth [BxHxWxN] or [HxWxN]
            threshold: threshold for l1 in d1
            relative: relative threshold for d1
            mask: optional mask to hide some pixels
        Returns:
            epe error: l2 error of this sample
            outliers: percentage of bad flow pixels
            num_pixels: num of valid pixels
    '''
    outliers = outlier_ratio(gt, predicted, threshold, step)
    avg_epe = epe_error(predicted, gt)
    return avg_epe, outliers