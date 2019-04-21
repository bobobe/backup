import tensorflow as tf
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='1'

def unpool(pool, ind, ksize=[1, 1, 800, 1], scope='unpool'):
    with tf.variable_scope(scope):
        input_shape = pool.get_shape().as_list()
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        flat_input_size = np.prod(input_shape)#all+0.6drop element's product(total elements num)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
        return ret

#a = tf.ones([4,10], dtype=tf.float32)
b = tf.ones([4,10,5],dtype = tf.float32)
#[4,10,5,1]
b =tf.expand_dims(b,3)

#[4,10,1,800]
with tf.variable_scope("33"):
    b = tf.nn.conv2d(b,filter =tf.get_variable(name = 'f',shape = [3,5,1,800],initializer=tf.contrib.layers.xavier_initializer()),strides=[1,1,5,1],padding="SAME")
#[4,10,800,1]
b = tf.transpose(b,[0,1,3,2])

#[4,10,1,1]
b,arg = tf.nn.max_pool_with_argmax(b,ksize=[1, 1, 800, 1],strides=[1, 1, 800, 1],padding='SAME',name='pool1')
#[batch,T,filter,1]->[batch,T,embedding_dim,1]


#[4,10,800,1]
b = unpool(b,arg)

#[4,10,1,800]
b = tf.transpose(b,[0,1,3,2])

#[4,10,5,1]
b = tf.nn.conv2d_transpose(b, filter=tf.Variable(tf.ones([3,5,1,800])), output_shape=[4,10,5,1],strides=[1,1,5,1], padding="SAME")

with tf.Session() as sess:
    print(sess.run(tf.shape(b)))
    #print(sess.run(tf.shape(arg)))
