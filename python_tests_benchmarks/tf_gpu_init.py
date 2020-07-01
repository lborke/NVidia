
import tensorflow as tf

print('tf version = ', tf.__version__)
tf.test.is_built_with_cuda()


tf.test.is_gpu_available()

tf.test.gpu_device_name()


with tf.device('/gpu:0'):
	a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
	b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
	c = tf.matmul(a, b)


with tf.device('/gpu:0'):
	a = tf.constant([355.0, 4826.0, 37476.0, 487346.0, 3495.0, 5556.0], shape=[2, 3], name='a')
	b = tf.constant([46661.0, 2864754.0, 75762873.0, 7486584.0, 544155.0, 645746.0], shape=[3, 2], name='b')
	c = tf.matmul(a, b)


#
from tensorflow.python.client import device_lib

device_lib.list_local_devices()


## verbose output
#successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero

# https://stackoverflow.com/questions/36838770/how-to-interpret-tensorflow-output

