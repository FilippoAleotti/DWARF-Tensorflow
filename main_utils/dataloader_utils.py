'''
Utilities for dataloader

Author: Filippo Aleotti
Mail: filippo.aleotti2@unibo.it
'''

import tensorflow as tf
import numpy as np

def string_length_tf(t):
    '''
        From Monodepth by Godard et Al. 
        https://github.com/mrharicot/monodepth/blob/master/monodepth_dataloader.py
    '''
    return tf.py_func(len, [t], [tf.int64])

def read_image(image_path):
    ''' 
        Read images given a path
        From Monodepth by Godard et Al. 
        https://github.com/mrharicot/monodepth/blob/master/monodepth_dataloader.py
    '''

    path_length = string_length_tf(image_path)[0]
    file_extension = tf.substr(image_path, path_length - 3, 3)
    file_cond = tf.equal(file_extension, 'jpg')

    image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(
        image_path)), lambda: tf.image.decode_png(tf.read_file(image_path)))

    image = tf.image.convert_image_dtype(image,  tf.float32)
    return image


def get_next_line(filename_file):
    ''' Read next line from filenames file '''

    input_queue = tf.train.string_input_producer([filename_file])
    line_reader = tf.TextLineReader()
    _, line = line_reader.read(input_queue)
    return line


def get_image_paths_from_line(line, data_path, number_of_paths):
    ''' Read relative path from line in filenames and append the data_path '''

    paths = []
    split_line = tf.string_split([line]).values
    for index in range(number_of_paths):
        paths.append(tf.string_join([data_path, '/', split_line[index] ]))
    return paths

def get_number_of_paths(filename_file):
    with open(filename_file, 'r') as f:
        line = f.readline()
        number_of_paths = len(line.split(' '))
    return number_of_paths

def get_num_samples(filename_file):
    return sum(1 for line in open(filename_file))

def tf_get_height_width(x):
    def _get_height_width(inp):
        if inp.ndim == 4:
            _,h,w,_ = inp.shape
        else:
            h,w,_ = inp.shape
        shape = np.array([h,w], dtype=np.int32)
        return shape
    out = tf.py_func(_get_height_width, [x], tf.int32, stateful=False)
    return out

def tf_get_height_width_of_list(x, lenght):
    out = []
    for i in range(lenght):    
        out.append(tf_get_height_width(x[i]))
    return out