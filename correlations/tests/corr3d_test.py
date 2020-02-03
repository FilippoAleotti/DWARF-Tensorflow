import tensorflow as tf
import numpy as np
from external_packages.correlation3D.ops import correlation3D as cuda_corr 
from correlations.correlation3D import correlation3D as native_corr
import os

class Corr3DTest(tf.test.TestCase):
    def test_equals_mdd0(self):
        x =  np.random.rand(2,480,640,128)
        y =  np.random.rand(2,480,640,128)
        with self.test_session():
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.float32)
            md= 3
            mdd=0
            native_corr_res = native_corr(x,y,pad=md, kernel_size=1, max_displacement=md, stride_1=1, stride_2=1, max_depth_displacement=mdd).eval()
            cuda_corr_res = cuda_corr(x,y,pad=md, kernel_size=1, max_displacement=md, stride_1=1, stride_2=1, max_depth_displacement=mdd).eval()
            assert cuda_corr_res.shape == native_corr_res.shape
            print(np.max(cuda_corr_res - native_corr_res))
            number_errors = np.sum(np.abs(cuda_corr_res - native_corr_res) > 0.01)
            self.assertAllEqual(number_errors, 0)

    def test_equals_mdd1(self):
        x =  np.random.rand(2,480,640,128)
        y =  np.random.rand(2,480,640,128)

        with self.test_session():
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.float32)
            md= 4
            mdd=1
            native_corr_res = native_corr(x,y,pad=md, kernel_size=1, max_displacement=md, stride_1=1, stride_2=1, max_depth_displacement=mdd).eval()
            cuda_corr_res = cuda_corr(x,y,pad=md, kernel_size=1, max_displacement=md, stride_1=1, stride_2=1, max_depth_displacement=mdd).eval()
            number_errors = np.sum(np.abs(cuda_corr_res - native_corr_res) > 0.01)
            self.assertAllEqual(number_errors, 0)

if __name__ == '__main__':
    tf.test.main()