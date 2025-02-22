
# cd D:\PowerTools\NVidia\python

# python tf_benchmark_simple.py cpu 100
# python tf_benchmark_simple.py cpu 1000
# python tf_benchmark_simple.py cpu 10000

# python tf_benchmark_simple.py gpu 10000


import sys
import tensorflow as tf
from datetime import datetime

# device_name = "gpu"
# device_name = "cpu"
# shape = (10000, 10000)
# shape = (20000, 20000)

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
shape = (int(sys.argv[2]), int(sys.argv[2]))


if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"


with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)


startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print(result)


# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 5)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", str(datetime.now() - startTime))


