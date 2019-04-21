import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
a = tf.ones([2,3,4],dtype = tf.float32)
b = tf.square(a)
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    print(sess.run(a))
    print(sess.run(b))