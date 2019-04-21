import numpy as np
from sklearn.metrics import f1_score
# a = {'1':[1,2,3,4],'2':[1,2,3,4]}
# b = np.stack(a.values())
# print(b)
# emb_mean, emb_std = b.mean(), b.std()
# print(emb_mean,emb_std)
#
#
# a = [1,2,3]
# b = [4,5,6]
# c = [4,5,6,7,8]
# zipped = zip(a,b)     # 打包为元组的列表
# #[(1, 4), (2, 5), (3, 6)]
# d = list(zip(a,b,c))
# print(d)
# for i in d:
#     print(i)
# # 元素个数与最短的列表一致
# #[(1, 4), (2, 5), (3, 6)]
# zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
# #[(1, 2, 3), (4, 5, 6)]
#
# a = [[0,0,1,0]]*10
# print(np.array(a))
# print(np.argmax(a,axis=-1))
# #a = np.array([1,0])
#
# a = np.array([1,2,3,0,2,3,4,2,1])
# b = np.array([1,2,3,1,2,2,4,2,1])
# print(f1_score(a,b,average="macro"))
import tensorflow as tf
from tensorflow import keras

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Start print log")
logger.debug("Do something")
logger.warning("Something maybe fail.")
logger.info("Finish")

print(tf.__version__)


def private_conv_encoder(self, origin_embedding_input, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        params = {"inputs": origin_embedding_input, "filters": 800, "kernel_size": 3, "strides": 1,
                  "activation": tf.nn.tanh, \
                  "use_bias": True, "padding": 'SAME', "name": 'conv'}

        # output:[batch,T,filters]
        output = tf.layers.conv1d(**params)
        # [batch,T,filters,1]#batch,height,weight,channal
        output = tf.expand_dims(output, 3)
        # [batch,T,1,1]
        output, args = tf.nn.max_pool_with_argmax(output, ksize=[1, 1, 800, 1], strides=[1, 1, 800, 1], padding='SAME',
                                                  name='max_pool')
        # [batch,T]

        private_conv_output = tf.squeeze(output, 2)
        private_conv_output = tf.squeeze(private_conv_output, 2)
        maxpool_args = args
        return private_conv_output, maxpool_args


def shared_conv_encoder(self, origin_embeddings_input, scope="sconv"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        params = {"inputs": origin_embeddings_input, "filters": 200, "kernel_size": 3, "strides": 1,
                  "activation": tf.nn.tanh,
                  "use_bias": True, "padding": 'SAME', "name": 'conv'}

        # output:[batch,T,filters]
        output = tf.layers.conv1d(**params)
        # [batch,T,filters,1]
        output = tf.expand_dims(output, 3)
        # [batch,T,1,1]
        output = tf.nn.max_pool(output, ksize=[1, 1, 200, 1], strides=[1, 1, 200, 1], padding='SAME', name='max_pool')
        # [batch,T]
        shared_conv_output = tf.squeeze(output, 2)
        shared_conv_output = tf.squeeze(shared_conv_output, 2)
        # print(shared_conv_output.get_shape())
        return shared_conv_output
        # self.source_shared_conv_output = tf.squeeze(output)
        # self.target_shared_conv_output = tf.squeeze(output)

