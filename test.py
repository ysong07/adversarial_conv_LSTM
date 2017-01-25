import tensorflow as tf
import numpy as np
import pdb
batch_size = 10
input_size = 10
output_size = 10
inputs = tf.placeholder(tf.float32, shape=(batch_size, input_size))
variable = tf.get_variable("Matrix",[input_size,output_size],initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None, dtype=tf.float32))
outputs = tf.mul(inputs,variable)
abc = np.arange(100).reshape([10,10])
efg = tf.constant(abc)
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
feed_dict = {inputs:efg}
print sess.run([outputs],feed_dict=feed_dict)


