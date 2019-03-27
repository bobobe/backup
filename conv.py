#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/15 21:07
# @Author  : yanbo
# @Site    : 
# @File    : conv.py
# @Software: PyCharm
# @python version:
# import tensorflow as tf
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
import tensorflow as tf
import numpy as np

def conv1(input,scope):
    with tf.variable_scope(scope,reuse = None):
        #[batch,T,hidden]
        input = input
        #[batch,T,hidden,1]
        input = tf.expand_dims(input,-1)
        #[batch,T,1,20]
        output = tf.nn.conv2d(input = input,filter=tf.Variable(tf.random_normal([2,3,1,20])),strides = [1,1,3,1],padding = "SAME")
        #[batch,T,20,1]
        output = tf.transpose(output,[0,1,3,2])
        #[batch,T,20]
        output = tf.squeeze(output,3)
        return output

def max_pool(input,scope):
    with tf.variable_scope(scope, reuse=None):
        #[batch,T,20]
        input = input
        #[batch,T,20,1]
        input = tf.expand_dims(input,-1)
        #[batch,T,1,1]
        output = tf.nn.max_pool(value=input,ksize=[1,1,20,1],strides=[1,1,20,1],padding="VALID",name ="padding")
        output = tf.squeeze(output,2)
        #[batch,T]
        output = tf.squeeze(output,2)
        return output

def mlp(input, scope):
    with tf.variable_scope(scope,reuse = None):
        #[batch,T]
        input = input
        w = tf.get_variable(name = "w",shape = [10,2],initializer=tf.ones_initializer(),dtype=tf.float32)
        b = tf.get_variable(name = "b",shape = [2],initializer=tf.ones_initializer(),dtype=tf.float32)
        #[batch,2]
        output = tf.matmul(input,w)+b
        return output

def train_op(loss):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_op = optimizer.minimize(loss)
    return train_op



p1 = tf.placeholder(shape = [None,10,3],name = "p1",dtype=tf.float32)

d1 = np.random.randint(0,10,[100,10,3])
d1 = d1.astype(np.float32)
t1 = np.array([[1,0],[0,1]]*50)

output = conv1(p1,"conv1")
output = max_pool(output,"pool1")
output = mlp(output,"mlp")

loss = tf.nn.softmax_cross_entropy_with_logits(logits= output,labels = t1)
loss = tf.reduce_mean(loss)
train_op = train_op(loss)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(40):
        print(sess.run([train_op,loss],feed_dict={p1:d1}))




