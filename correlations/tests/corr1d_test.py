import tensorflow as tf
import numpy as np
from external_packages.correlation1D.corr1d import correlation1d as cuda_corr 
from correlations.correlation1D import correlation1D as native_corr

class Corr1DTest(tf.test.TestCase):
    def test_equals_md3(self):
        x =  np.random.rand(2,480,640,128)
        y =  np.random.rand(2,480,640,128)

        with self.test_session():
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.float32)
            md= 3
            native_corr_res = native_corr(x,y,md,stride=1).eval()
            cuda_corr_res = cuda_corr(x,y,pad=md, kernel_size=1, max_displacement=md, stride_1=1, stride_2=1).eval()
            number_errors = np.sum(np.abs(cuda_corr_res - native_corr_res) > 0.01)
            self.assertAllEqual(number_errors, 0)

    def test_equals_md4(self):
        x =  np.random.rand(2,480,640,128)
        y =  np.random.rand(2,480,640,128)

        with self.test_session():
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            y = tf.convert_to_tensor(y, dtype=tf.float32)
            md= 4
            native_corr_res = native_corr(x,y,md,stride=1).eval()
            cuda_corr_res = cuda_corr(x,y,pad=md, kernel_size=1, max_displacement=md, stride_1=1, stride_2=1).eval()
            number_errors = np.sum(np.abs(cuda_corr_res - native_corr_res) > 0.01)
            self.assertAllEqual(number_errors, 0)

if __name__ == '__main__':
    tf.test.main()