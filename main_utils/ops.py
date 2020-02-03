'''
Operations used by networks

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf

def regularization_term(gamma):
    '''
        Regularisation term on weights
    '''
    with tf.variable_scope('weight_decay'):
        regularization_loss = 0
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for weight in weights:
            regularization_loss += tf.reduce_sum(tf.square(weight))
        
        regularization_loss = regularization_loss * gamma
        return regularization_loss
        
def l1_loss(prediction, gt, alpha):
    '''
    L1 loss
    '''
    with tf.variable_scope('l1'):
        mask = nan_mask(gt)
        gt = replace_nan_values(gt)
        loss = tf.norm(gt - prediction, ord=1, axis=3)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2]))
        loss = loss * alpha
    return loss

def masked_l1_loss(prediction, gt, mask, alpha):
    '''
        L1 loss with mask
    '''
    with tf.variable_scope('masked_l1'): 
        valid_values = tf.expand_dims(tf.norm(gt-prediction, ord=1, axis=3), axis=-1)
        loss = tf.where(tf.equal(mask, 0.), mask, valid_values)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=[1,2]))
        loss = loss * alpha
    return loss

def downsample_image(image, original_height, original_width, downsampling_factor, method=tf.image.ResizeMethod.AREA):
    ''' 
        Downsample an image give original size and downsampling factor
    '''
    with tf.variable_scope('downsample_image'):
        new_height = int(original_height / (2**downsampling_factor))
        new_width  = int(original_width / (2**downsampling_factor))
        downsampled = tf.image.resize_images(image, [new_height, new_width], method=method) 
        return downsampled

def nan_mask(gt):
    with tf.variable_scope('remove_nan'):
        nan_mask = tf.where(tf.is_nan(gt), tf.zeros_like(gt), tf.ones_like(gt))
    return nan_mask

def replace_nan_values(gt):
    with tf.variable_scope('replace_nan'):
        gt = tf.where(tf.is_nan(gt), tf.zeros_like(gt), gt)
        return gt

def image_warp(im, flow):
    '''
        From Unflow by Meister et al
        https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/image_warp.py
        
        Performs a backward warp of an image using the predicted flow.
        Args:
            im: Batch of images. [num_batch, height, width, channels]
            flow: Batch of flow vectors. [num_batch, height, width, 2]
        Returns:
            warped: transformed image of the same shape as the input image.
    '''
    with tf.variable_scope('image_warp'):

        num_batch, height, width, channels = tf.unstack(tf.shape(im))
        max_x = tf.cast(width - 1, 'int32')
        max_y = tf.cast(height - 1, 'int32')
        zero = tf.zeros([], dtype='int32')

        # We have to flatten our tensors to vectorize the interpolation
        im_flat = tf.reshape(im, [-1, channels])
        flow_flat = tf.reshape(flow, [-1, 2])

        # Floor the flow, as the final indices are integers
        # The fractional part is used to control the bilinear interpolation.
        flow_floor = tf.to_int32(tf.floor(flow_flat))
        bilinear_weights = flow_flat - tf.floor(flow_flat)

        # Construct base indices which are displaced with the flow
        pos_x = tf.tile(tf.range(width), [height * num_batch])
        grid_y = tf.tile(tf.expand_dims(tf.range(height), 1), [1, width])
        pos_y = tf.tile(tf.reshape(grid_y, [-1]), [num_batch])

        x = flow_floor[:, 0]
        y = flow_floor[:, 1]
        xw = bilinear_weights[:, 0]
        yw = bilinear_weights[:, 1]

        # Compute interpolation weights for 4 adjacent pixels
        # expand to num_batch * height * width x 1 for broadcasting in add_n below
        wa = tf.expand_dims((1 - xw) * (1 - yw), 1) # top left pixel
        wb = tf.expand_dims((1 - xw) * yw, 1) # bottom left pixel
        wc = tf.expand_dims(xw * (1 - yw), 1) # top right pixel
        wd = tf.expand_dims(xw * yw, 1) # bottom right pixel

        x0 = pos_x + x
        x1 = x0 + 1
        y0 = pos_y + y
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        dim1 = width * height
        batch_offsets = tf.range(num_batch) * dim1
        base_grid = tf.tile(tf.expand_dims(batch_offsets, 1), [1, dim1])
        base = tf.reshape(base_grid, [-1])

        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        warped_flat = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
        warped = tf.reshape(warped_flat, [num_batch, height, width, channels])

        return warped
