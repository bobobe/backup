#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/9 16:20
# @Author  : yanbo
# @Site    : 
# @File    : GSN.py
# @Software: PyCharm
# @python version:
import numpy as np
import sys
sys.path.append("..")
import time
from inputs.data_helper import batch_iter,dev_batch_iter
from .model_helper import flip_gradient,unpool,get_logger
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.contrib.rnn import LSTMCell


class GSN(object):
    def __init__(self, args, paths,embedding,config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch_num
        self.pre_train = args.pre_train
        self.word_embeddings = embedding["word_embedding"]
        self.position_embeddings = embedding["pos_embedding"]
        self.update_embedding = args.update_embedding
        self.optimizer = args.optimizer
        self.lr = args.learning_rate
        self.clip_grad = args.clip_grad
        self.shuffle = args.shuffle
        self.model_path = paths["model_path"]
        self.summary_path = paths["summary_path"]
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.num_relations = args.num_relations
        self.position_emb_size = args.pos_embedding_dim
        self.word_emb_size = args.word_embedding_dim
        self.all_emb_size = self.word_emb_size + self.position_emb_size*2
        self.max_seq_len = args.max_seq_len
        self.config = config


    def build_graph(self):
        self.add_placeholders()
        source_embeddings,target_embeddings = self.add_embedding_layers()

        #textcnn
        #logits = self.text_cnn_encoder(source_embeddings,"text_cnn")

        #sconv
        output = self.shared_conv_encoder(source_embeddings,scope = "sconv")
        logits = self.fully_connected_layer(output)

        #lstm
        #logits = self.lstm_encoder(source_embeddings,"lstm")

        self.train_labels_predict = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        rel_loss = self.relation_loss(logits)
        self.source_loss = rel_loss
        self.source_train_op = self.trainstep_op(self.source_loss)
        tf.summary.scalar("source_loss", self.source_loss)

        #dev
        # [batch,]
        self.labels_predict = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        #init
        self.init_op()


    def add_placeholders(self):
        self.source_word_ids = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_seq_len], name="source_word_ids")#源域word在单词表中的下标，用于embedding_lookup层
        self.target_word_ids = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_seq_len], name="target_word_ids")#目标域word在单词表中的下标，用于embedding_lookup层

        self.source_position_ids1 = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_seq_len], name="source_position_ids1")#源域每个单词离实体的距离1
        self.source_position_ids2 = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_seq_len], name="source_position_ids2")  # 源域每个单词离实体的距离2
        self.target_position_ids1 = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_seq_len],name="target_position_ids1")  # 源域每个单词离实体的距离1
        self.target_position_ids2 = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_seq_len],name="target_position_ids2")  # 源域每个单词离实体的距离2

        self.domain_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 2], name="domain_labels")#数据属于的域
        self.relation_labels = tf.placeholder(tf.int32, shape=[self.batch_size,self.num_relations], name="relation_labels")#关系种类
        self.dropout = tf.placeholder(tf.float32,shape=[],name = "dropout")
    def add_embedding_layers(self):
        with tf.variable_scope("embedding"):
            _word_embeddings = tf.get_variable(initializer=self.word_embeddings,
                                                dtype=tf.float32,
                                                trainable=self.update_embedding,
                                                name="_word_embeddings")
            _position_embeddings = tf.get_variable(initializer=self.position_embeddings,
                                                dtype=tf.float32,
                                                trainable=self.update_embedding,
                                                name="_position_embeddings")
            source_word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                            ids=self.source_word_ids,
                                                            name="source_word_embeddings")
            target_word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                            ids=self.target_word_ids,
                                                            name="target_word_embeddings")
            source_position_embeddings1 = tf.nn.embedding_lookup(params=_position_embeddings,
                                                                ids=self.source_position_ids1,
                                                                name="source_position_embeddings1")
            source_position_embeddings2 = tf.nn.embedding_lookup(params=_position_embeddings,
                                                                 ids=self.source_position_ids2,
                                                                 name="source_position_embeddings2")
            target_position_embeddings1 = tf.nn.embedding_lookup(params=_position_embeddings,
                                                                ids=self.target_position_ids1,
                                                                name="target_position_embeddings1")
            target_position_embeddings2 = tf.nn.embedding_lookup(params=_position_embeddings,
                                                                 ids=self.target_position_ids2,
                                                                 name="target_position_embeddings2")
            #[batch,T,embedding_dim]
            source_embeddings = tf.concat([source_word_embeddings,source_position_embeddings1,source_position_embeddings2],axis=-1)
            target_embeddings = tf.concat([target_word_embeddings,target_position_embeddings1,target_position_embeddings2],axis=-1)
            #print(source_embeddings.get_shape())
            return source_embeddings,target_embeddings

    def shared_conv_encoder(self,origin_embedding_input,scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            '''
            #[batch,T,embeding_dim,1]
            output = tf.expand_dims(origin_embeddings_input,axis=-1)
            # [batch,T,1,filters]
            output = tf.nn.conv2d(input=output,
                                  filter=tf.Variable(tf.ones([3, self.all_emb_size, 1, 800])),
                                  strides=[1, 1, self.all_emb_size, 1],
                                  padding="SAME", use_cudnn_on_gpu=None, name="conv")
            # [batch,T,filters,1]
            output = tf.transpose(output, [0, 1, 3, 2])

            '''

            params = {"inputs": origin_embedding_input, "kernel_initializer":tf.contrib.layers.xavier_initializer(),
                      "bias_initializer": tf.zeros_initializer(), "filters": 800,"kernel_size": 3, "strides": 1,
                      "activation": tf.nn.tanh,"use_bias": True, "padding": 'SAME', "name": 'conv'}

            #output:[batch,T,filters]
            output = tf.layers.conv1d(**params)
            #[batch,T,filters,1]
            output = tf.expand_dims(output,3)


            #[batch,T,1,1]
            #output = tf.nn.max_pool(output,ksize = [1,1,800,1],strides = [1,1,800,1],padding = 'SAME',name = 'max_pool')
            #[batch,filters,1]
            output = tf.reduce_max(output,axis = 1)
            #[batch,filters]
            shared_conv_output = tf.squeeze(output,2)
            #shared_conv_output = tf.squeeze(shared_conv_output,2)
            #print(shared_conv_output.get_shape())
            shared_conv_output = tf.nn.dropout(shared_conv_output, self.dropout)
            return shared_conv_output
            # self.source_shared_conv_output = tf.squeeze(output)
            # self.target_shared_conv_output = tf.squeeze(output)

    def fully_connected_layer(self,shared_conv_output):
        with tf.variable_scope("fc_layer",reuse=tf.AUTO_REUSE):
            W = tf.get_variable(name="W",
                                shape=[800, 300],
                                #通过xavier算法通过输入输出神经元的数目自动确定权值矩阵的初始化大小
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[300],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            #[batch,T]*[T,300] = [batch,300]
            output = tf.matmul(shared_conv_output,W)+b
            hidden_output = tf.nn.relu(output)#[batch,300]

        with tf.variable_scope("proj",reuse=tf.AUTO_REUSE):
            W = tf.get_variable(name="W",
                                shape=[300, self.num_relations],
                                # 通过xavier算法通过输入输出神经元的数目自动确定权值矩阵的初始化大小
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_relations],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            #[batch,300]*[300,num_relations] = [batch,num_relations]
            fully_connected_output = tf.matmul(hidden_output, W) + b
            #print(fully_connected_output.get_shape())
            return fully_connected_output

    def lstm_encoder(self,origin_embedding_input,scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            cell_fw = LSTMCell(150)
            cell_bw = LSTMCell(150)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=origin_embedding_input,
                dtype=tf.float32)
            #[batch,max_len,150]:output_fw
            output_fw = tf.transpose(output_fw,[1,0,2])#[max_len,batch,150]
            output_bw = tf.transpose(output_bw,[1,0,2])
            output = tf.concat([output_fw, output_bw], axis=-1)#[max_len,batch,300]
            output = output[-1]#[batch,300]
            #print(output.get_shape())

        with tf.variable_scope("proj",reuse=tf.AUTO_REUSE):
            W = tf.get_variable(name="W",
                                shape=[2 * 150, self.num_relations],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32,
                                trainable=True)

            b = tf.get_variable(name="b",
                                shape=[self.num_relations],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32,
                                trainable=True)

            #s = tf.shape(output)
            output = tf.matmul(output, W) + b
            return output

    def text_cnn_encoder(self,origin_embedding_input,scope):
        with tf.variable_scope(scope,reuse= tf.AUTO_REUSE):
            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            fliter_sizes = [2,3,4,5]
            num_fliters = 128
            dropout_keep_prob = 0.5
            initializer = tf.keras.initializers.glorot_normal
            origin_embedding_input = tf.expand_dims(origin_embedding_input,-1)
            for i, filter_size in enumerate(fliter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    conv = tf.layers.conv2d(origin_embedding_input, num_fliters, [filter_size, self.all_emb_size],
                                            kernel_initializer=initializer(), activation=tf.nn.relu, name="conv")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(conv, ksize=[1, self.max_seq_len - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID', name="pool")
                    pooled_outputs.append(pooled)

            # Combine all+0.6drop the pooled features
            num_filters_total = num_fliters * len(fliter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add dropout
            with tf.variable_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, dropout_keep_prob)

            # Final scores and predictions
            with tf.variable_scope("output"):
                output = tf.layers.dense(self.h_drop, self.num_relations, kernel_initializer=initializer())
                #self.pre = tf.argmax(self.logits, 1, name="predictions")
            return output

    def relation_loss(self,fully_connected_output):
        with tf.variable_scope("relation_loss", reuse=None):
            # logits为[batch_size, num_relations]
            # labels为[batch_size,1](真实标记）
            # [batch,1]
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=fully_connected_output,
                                                           labels=self.relation_labels)
            l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            loss = tf.reduce_mean(loss) + 1e-5 * l2

            # [1],batch average loss
            tf.summary.scalar("relation_loss",loss)
            return loss


    def trainstep_op(self,loss,scope = "train_op"):
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            #optim = tf.train.GradientDescentOptimizer(1)
            optim = tf.train.AdamOptimizer()
            #optim = tf.train.AdadeltaOptimizer(self.lr, 0.9, 1e-6)

            #train_op = optim.minimize(loss)

            gradient_all = optim.compute_gradients(loss)
            # 计算全部gradient
            grads_and_vars = [(g, v) for (g, v) in gradient_all if g is not None]
            #得到可进行梯度计算的变量
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads_and_vars]
            train_op = optim.apply_gradients(capped_gvs, global_step=self.global_step)


            #每次梯度更新前计算梯度，返回(梯度，变量)的列表，即每个变量wi的梯度值
            # grads_and_vars = optim.compute_gradients(loss)
            # #梯度修剪
            # grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            # #global_step每次加1，表示迭代次数
            # train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
            return train_op

    def init_op(self):
        self.init_op = tf.global_variables_initializer()


    def train(self, train, dev):#训练集和验证集
        """

        :param train:
        :param dev:
        :return:
        """
        #golbal_variables()返回所有变量域中的变量，如果加scope参数，则返回特定变量域的变量
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config = self.config) as sess:
            sess.run(self.init_op)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, epoch, saver)

    # 一个epoch会分批训练所有样本
    def run_one_epoch(self, sess, train, dev, epoch, saver):
        """

        :param sess:
        :param train:[source_trian,target_train],source_train:list of [x,p1,p2,y]
        :param dev:
        :param epoch:
        :param saver:
        :return:
        """
        num_batches = (len(train[0]) + self.batch_size - 1) // self.batch_size

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # batches是按batch_size划分的一个迭代器，包含了全部的样本
        batches = batch_iter(train, self.batch_size,shuffle=self.shuffle)


        for step, (source_batch, target_batch) in enumerate(batches):
            #self.logger.info(str(step))
            # sys.stdout.write就是输出到屏幕
            #sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')

            source_x, source_p1, source_p2, source_y = zip(*source_batch)
            target_x, target_p1, target_p2, target_y = zip(*target_batch)
            source_domain = np.array([[1,0]]*len(source_x))
            target_domain = np.array([[0,1]]*len(target_x))

            #source train
            step_num = epoch * num_batches + step + 1
            source_feed_dict = {self.source_word_ids:source_x,
                                self.source_position_ids1:source_p1,
                                self.source_position_ids2:source_p2,
                                self.domain_labels :source_domain,
                                self.relation_labels:source_y,
                                self.dropout:0.5}
            label_pre,_, loss_train_source, step_num_ = sess.run([self.train_labels_predict,self.source_train_op, self.source_loss, self.global_step],
                                                         # global_step每次递增1，是一个全局的step计数
                                                         feed_dict=source_feed_dict)
            label_true = np.argmax(source_y,axis=-1)
            #print(label_true)
            #print(label_pre)

            macro_f1 = f1_score(label_true, label_pre, average="macro")
            log = 'epoch {}: macro_f1:{}'.format(epoch + 1, macro_f1)
            self.logger.info(log)

            if (step + 1 == 1 or (step + 1) % 10 == 0 or step + 1 == num_batches):  # log every 10 batch_steps
                self.logger.info(
                    '{} epoch {}, batch_step {}/{}, source_train_loss: {:.4}, global_step: {}'
                        .format(start_time, epoch + 1, step + 1, num_batches, loss_train_source,step_num))


            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)  # 按训练迭代轮次存储。

        # dev after every epoch
        self.logger.info('===========validation / test===========')
        # dev
        self.dev_one_epoch(sess, dev, epoch)

    def dev_one_epoch(self, sess, dev, epoch):
        """

        :param sess:
        :param dev:
        :return:
        """
        batches = dev_batch_iter(dev, self.batch_size, shuffle=False)
        true_label_list,pre_label_list = [],[]

        for step, dev in enumerate(batches):

            dev_x, dev_p1, dev_p2, dev_y = zip(*dev)
            dev_feed_dict = {self.source_word_ids: dev_x,
                             self.source_position_ids1: dev_p1,
                             self.source_position_ids2: dev_p2,
                             self.relation_labels: dev_y,
                             self.dropout:1.0}
            # [batch,]
            label_pre = sess.run(self.labels_predict,feed_dict=dev_feed_dict)
            #print(label_pre)
            # [batch,]
            label_true = np.argmax(dev_y, axis=-1)

            true_label_list.extend(label_true)
            #label_pre = label_pre.to_list()
            pre_label_list.extend(label_pre)

        print("true label:"+str(true_label_list))
        print("pre label:"+str(pre_label_list))

        macro_f1 = f1_score(true_label_list, pre_label_list, average="macro")
        self.logger.info('epoch {}, macro_f1: {:.4}'
                         .format(epoch + 1,macro_f1))
