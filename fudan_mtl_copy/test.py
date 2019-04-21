import os
import tensorflow as tf
import numpy as np

#only in cpu
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES']='-1'

#gpu
os.environ['CUDA_VISIBLE_DEVICES']='1'

def conv1d(origin_embeddings_input,scope = "sconv"):
    output = tf.nn.conv1d(
        #[batch,T,hidden]
        value = origin_embeddings_input,
        #[height,weight,outchannel]
        filters = tf.Variable(tf.ones([2,3,800])),
        stride = 1,
        padding = "SAME",
        use_cudnn_on_gpu=False,
        data_format=None,
        name="conv1d"
    )
    #[batch,T,20]
    print(output.get_shape())
    return output

def conv2d1(origin_embeddings_input,scope = "sconv"):
    # [batch,T,embed_dim,1]
    output = tf.expand_dims(origin_embeddings_input, -1)
    #[batch,T,1,800]
    output = tf.layers.conv2d(
        inputs = output,
        filters = 800,
        kernel_size = (2, 3),
        strides=(1, 3),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None
    )
    print(output.get_shape())
    # [batch,T,800,1]
    output = tf.transpose(output, [0, 1, 3, 2])
    # [batch,T,800]
    output = tf.squeeze(output, 3)
    print(output.get_shape())
    return output



def conv1(origin_embeddings_input,scope="sconv"):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        # params = {"inputs": origin_embeddings_input, "filters": 800,"kernel_size": 3, "strides": 1, "activation": tf.nn.tanh,
        #           "use_bias": True, "padding": 'SAME', "name": 'conv'}
        #[batch,T,embed_dim,1]
        input = tf.expand_dims(origin_embeddings_input,-1)
        print(input.get_shape())
        #[batch,T,1,800]
        output = tf.nn.conv2d(input = input,filter=tf.Variable(tf.ones([3,3,1,800])),strides=[1,1,3,1],padding = "SAME",use_cudnn_on_gpu=None)
        print(output.get_shape())
        #[batch,T,800,1]
        output = tf.transpose(output,[0,1,3,2])
        #[batch,T,800]
        shared_conv_output = tf.squeeze(output,3)
        # #[batch,T,1,1]
        # output = tf.nn.max_pool(output,ksize = [1,1,800,1],strides = [1,1,800,1],padding = "VALID",name = 'max_pool')
        # # print(output.get_shape())
        # #[batch,T]
        # shared_conv_output = tf.squeeze(output,2)
        # shared_conv_output = tf.squeeze(shared_conv_output,2)

        return shared_conv_output

def rd(input):
    #[batch,T,800]
    input = input
    #[batch,T]
    output = tf.reduce_mean(input,axis=2)
    return output
# def reduce_max(input ,scope):
#     #[batch,T]
#     output = tf.reduce_max(input,axis=2)
def max_pool(input,scope):
    with tf.variable_scope(scope, reuse=None):
        #[batch,T,800]
        input = input
        #[batch,T,800,1]
        input = tf.expand_dims(input,-1)
        #[batch,T,1,1]
        #output = tf.nn.max_pool(value=input,ksize=[1,1,800,1],strides=[1,1,1,1],padding="VALID",name ="max_pool")
        #print(output.get_shape())
        output = tf.reduce_max(input, axis=2, keep_dims=True)
        output = tf.squeeze(output,2)
        #[batch,T]
        output = tf.squeeze(output,2)
        return output

def mlp(data,scope):
    with tf.variable_scope(scope, reuse=None):
        # [batch,T]
        W = tf.get_variable(name="W",
                            shape=[10, 2],
                            # 通过xavier算法通过输入输出神经元的数目自动确定权值矩阵的初始化大小
                            initializer=tf.ones_initializer(),
                            dtype=tf.float32)

        b = tf.get_variable(name="b",
                            shape=[2],
                            initializer=tf.ones_initializer(),
                            dtype=tf.float32)
        #[batch,T]*[T,2] = [batch,2]
        domain_classifer_output = tf.matmul(data, W)+b
        #domain_classifer_output = tf.nn.relu(domain_classifer_output)
        return domain_classifer_output

def trainstep_op(loss):
    optim = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    #train_op = optim.minimize(loss)
    grads_and_vars = optim.compute_gradients(loss)
    grads_and_vars = [(grad, var) for grad, var in grads_and_vars if grad is not None]
    capped_gvs = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in grads_and_vars]
    # 执行对应变量的更新梯度操作
    train_op = optim.apply_gradients(capped_gvs)

    return train_op


#source_word_ids = tf.placeholder(tf.float32, shape=[None,10,3],name = "source_placeholder")
target_word_ids = tf.placeholder(tf.float32, shape=[None,10,3],name = "target_placeholder")

b= np.random.randint(0,10,size=[3,10,3])
b = b.astype(np.float32)
t = np.array([[0,1],[1,0],[1,0]])

output2 = conv1(target_word_ids,'conv1')
output2 = max_pool(output2,'pool1')
#output2 = rd(output2)
output2 = mlp(output2,"mlp1")
loss2 = tf.nn.softmax_cross_entropy_with_logits(logits=output2,labels=t)
loss2 = tf.reduce_mean(loss2)
#bb = tf.summary.scalar("loss2", loss2)
trainop2 = trainstep_op(loss2)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4

#merged = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES,'loss2')])
with tf.Session(config=config) as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(10):
        print(sess.run([trainop2,loss2],feed_dict={target_word_ids:b}))
    #print(sess.run([bb], feed_dict={target_word_ids:b}))
    #print(sess.run(tf.shape(arg)))
