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

        # target graph
        private_conv_output, maxpool_args = self.private_conv_encoder(target_embeddings, "target_private_conv_encoder")
        shared_conv_output = self.shared_conv_encoder(target_embeddings, "shared_conv_encoder")
        # fully_connected_output = self.fully_connected_layer(shared_conv_output)
        domain_classifer_output = self.gradient_reversed_layer(shared_conv_output)
        deconv_output = self.deconvolution_layer(shared_conv_output, private_conv_output, maxpool_args,"target_deconv_layer")
        # target_loss
        rel_loss = 0  # self.relation_loss(fully_connected_output)
        adv_loss = self.advers_loss(domain_classifer_output,"target")
        rec_loss = self.recons_loss(target_embeddings, deconv_output,"target")
        dif_loss = self.diff_loss(private_conv_output, shared_conv_output,"target")
        self.target_loss = rel_loss + 0.075 * dif_loss + 0.01 * rec_loss + 0.25 * adv_loss
        self.target_train_op = self.trainstep_op(self.target_loss)
        tf.summary.scalar("target_loss", self.target_loss)


        #source graph
        private_conv_output, maxpool_args = self.private_conv_encoder(source_embeddings,"source_private_conv_encoder")
        shared_conv_output = self.shared_conv_encoder(source_embeddings,"shared_conv_encoder")
        fully_connected_output = self.fully_connected_layer(shared_conv_output)
        domain_classifer_output = self.gradient_reversed_layer(shared_conv_output)
        deconv_output = self.deconvolution_layer(shared_conv_output, private_conv_output, maxpool_args,"source_deconv_layer")
        #source_loss
        rel_loss = self.relation_loss(fully_connected_output)
        adv_loss = self.advers_loss(domain_classifer_output,"source")
        rec_loss = self.recons_loss(source_embeddings,deconv_output,"source")
        dif_loss = self.diff_loss(private_conv_output,shared_conv_output,"source")
        self.train_accuracy = self.accuracy(fully_connected_output, "train")
        self.source_loss = rel_loss + 0.075*dif_loss+ 0.01*rec_loss+ 0.25*adv_loss
        self.source_train_op = self.trainstep_op(self.source_loss)
        tf.summary.scalar("source_loss", self.source_loss)


        #dev graph
        shared_conv_output = self.shared_conv_encoder(target_embeddings, "shared_conv_encoder")
        fully_connected_output = self.fully_connected_layer(shared_conv_output)#[batch,num_relations]
        #[batch,]
        self.labels_predict = tf.cast(tf.argmax(fully_connected_output, axis=-1),tf.int32)

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

    def private_conv_encoder(self,origin_embedding_input,scope,reuse = None):
        with tf.variable_scope(scope,reuse = reuse):
            '''
            # [batch,T,embeding_dim,1]
            output = tf.expand_dims(origin_embedding_input, axis=-1)
            #[batch,T,1,filters]
            output = tf.nn.conv2d(input=output, filter=tf.Variable(tf.ones([3, self.all_emb_size, 1, 800])), strides=[1, 1, self.all_emb_size, 1],
                                  padding="SAME",use_cudnn_on_gpu=None,name = "conv")
            #[batch,T,filters,1]
            output = tf.transpose(output,[0,1,3,2])

            '''

            params = {"inputs": origin_embedding_input, "kernel_initializer": tf.contrib.layers.xavier_initializer(),
                      "bias_initializer": tf.zeros_initializer(), "filters": 800, "kernel_size": 3, "strides": 1,
                      "activation": tf.nn.tanh, "use_bias": True, "padding": 'SAME', "name": 'conv'}

            #output:[batch,T,filters]
            output = tf.layers.conv1d(**params)
            #[batch,T,filters,1]#batch,height,weight,channal
            output = tf.expand_dims(output,3)



            #[batch,T,1,1]
            output,args = tf.nn.max_pool_with_argmax(output,ksize = [1,1,800,1],strides = [1,1,800,1],padding = 'SAME',name = 'max_pool')
            output = tf.squeeze(output,2)
            output = tf.squeeze(output,2)

            '''
            #print(args.get_shape())
            args = tf.argmax(output, axis=2)
            args = tf.expand_dims(args,axis=3)
            output = tf.reduce_max(output, axis=2)
            output = tf.squeeze(output,2)
            #print(args.get_shape())
            '''

            #private_conv_output = tf.squeeze(output,2)
            # [batch,T]
            private_conv_output = output
            maxpool_args = args
            return private_conv_output,maxpool_args

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
            #[batch,T,1]
            output = tf.reduce_max(output,axis = 2)
            #[batch,T]
            shared_conv_output = tf.squeeze(output,2)
            #shared_conv_output = tf.squeeze(shared_conv_output,2)
            #print(shared_conv_output.get_shape())
            return shared_conv_output
            # self.source_shared_conv_output = tf.squeeze(output)
            # self.target_shared_conv_output = tf.squeeze(output)

    def fully_connected_layer(self,shared_conv_output):
        with tf.variable_scope("fc_layer",reuse=tf.AUTO_REUSE):
            W = tf.get_variable(name="W",
                                shape=[self.max_seq_len, 300],
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


    def gradient_reversed_layer(self,shared_conv_output):
        '''make the task classifier cannot reliably predict the task based on
            the shared feature
            '''

        with tf.variable_scope("domain_classify_layer",reuse = tf.AUTO_REUSE):
            # [batch,T]
            GRL_output = flip_gradient(shared_conv_output)
            W = tf.get_variable(name="W",
                                shape=[self.max_seq_len, 2],
                                #通过xavier算法通过输入输出神经元的数目自动确定权值矩阵的初始化大小
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[2],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            #[batch,T]*[T,2] = [batch,2]
            domain_classifer_output = tf.matmul(GRL_output,W)+b
            return domain_classifer_output

    def deconvolution_layer(self,shared_conv_output,private_conv_output,maxpool_args,scope,reuse = None):

        with tf.variable_scope(scope,reuse = reuse):

            # if(domain == "source"):
            #     ps = unpool(pool=ps,ind = self.source_maxpool_args,ksize=[1,1,800,1])
            # elif(domain == "target"):
            #     ps = unpool(pool=ps, ind=self.target_maxpool_args, ksize=[1,1,800,1])
            ps = shared_conv_output + private_conv_output
            ps = tf.expand_dims(ps, 2)
            ps = tf.expand_dims(ps, 3)
            #print(ps.get_shape())

            # [batch,T,1,1]
            ps = unpool(pool=ps, ind=maxpool_args, ksize=[1,1,800,1])
            # unpooling #[batch,T,filter,1]
            #deconvolution
            # [batch,T,1,filter]
            unpool_output = tf.transpose(ps, [0, 1, 3, 2])

            # [batch,T,embedd_dim,1]
            deconv_output = tf.nn.conv2d_transpose(unpool_output, filter=tf.get_variable(name = "filters",shape = [3,self.all_emb_size,1,800],initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32), output_shape=[self.batch_size, self.max_seq_len, self.all_emb_size, 1],
                                       strides=[1, 1, self.all_emb_size, 1], padding="SAME",name="deconv")
            # [batch,T,embedding_dim]
            deconv_output = tf.squeeze(deconv_output)
            return deconv_output

    def relation_loss(self,fully_connected_output):
        with tf.variable_scope("relation_loss", reuse=None):
            # logits为[batch_size, num_relations]
            # labels为[batch_size,1](真实标记）
            # [batch,1]
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=fully_connected_output,
                                                                  labels=self.relation_labels)  # labels为真
            # [1],batch average loss
            rel_loss = tf.reduce_mean(loss)
            tf.summary.scalar("relation_loss",rel_loss)
            return rel_loss

    def advers_loss(self,domain_classifer_output,domain):
        with tf.variable_scope(domain+"_adv_loss", reuse=None):
            adv_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.domain_labels, logits=domain_classifer_output))
            tf.summary.scalar(domain+"_adv_loss",adv_loss)
            return adv_loss

    def recons_loss(self,origin_embeddings_input,deconv_output,domain):
        with tf.variable_scope(domain + "_recons_loss", reuse=None):

            #[batch,T,embedding_dim]
            origin_input = origin_embeddings_input
            #norm
            #[batch,T]
            recons_norm = tf.sqrt(tf.reduce_sum(tf.square(deconv_output), axis=2))
            origin_norm = tf.sqrt(tf.reduce_sum(tf.square(origin_input), axis=2))
            #dot product
            dp = tf.reduce_sum(tf.multiply(deconv_output, origin_input), axis=2)
            #cos
            res = dp / (recons_norm * origin_norm)
            #recons loss
            d = 1 - tf.reduce_sum(tf.abs(res), axis=1)
            #batch average
            rec_loss = tf.reduce_mean(d)
            tf.summary.scalar(domain+"_recons_loss", rec_loss)
            return rec_loss

    def diff_loss(self,private_conv_output,shared_conv_output,domain):
        with tf.variable_scope(domain+"_diff_loss",reuse=None):

            p = private_conv_output
            s = shared_conv_output
            output = tf.norm(
                tf.matmul(tf.transpose(p, [1, 0]),s),#[batch,T,1]*[batch,1,T]=[batch,T,T]
                axis=[-2,-1]#batch first
            )
                #[batch,]
            dif_loss = tf.square(output)#[batch,]
            dif_loss = tf.reduce_mean(dif_loss)
            tf.summary.scalar(domain+"_diff_loss",dif_loss)
            return dif_loss

    def accuracy(self, fully_connect_output, catalog):
        with tf.variable_scope(catalog+"_accuracy", reuse=None):
            true_label = tf.argmax(self.relation_labels,-1)
            pre_label = tf.argmax(fully_connect_output,-1)
            correct_predictions = tf.equal(pre_label, true_label)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            tf.summary.scalar(catalog+"_accuracy", accuracy)
            return accuracy

    def trainstep_op(self,loss,scope = "train_op"):
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable(shape=[], initializer=tf.zeros_initializer(), name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr,rho=0.9,epsilon=1e-6,)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

            #train_op = optim.minimize(loss)

            gradient_all = optim.compute_gradients(loss)
            # 计算全部gradient
            grads_and_vars = [(g, v) for (g, v) in gradient_all if g is not None]
            #得到可进行梯度计算的变量
            grads_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for (g, v) in grads_and_vars]
            #生成new grad
            train_op = optim.apply_gradients(grads_clip,global_step=self.global_step)


            #每次梯度更新前计算梯度，返回(梯度，变量)的列表，即每个变量wi的梯度值
            # grads_and_vars = optim.compute_gradients(loss)
            # #梯度修剪
            # grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            # #global_step每次加1，表示迭代次数
            # train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)
            return train_op

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.source_merged = tf.summary.merge(
            [tf.get_collection(tf.GraphKeys.SUMMARIES, 'source_diff_loss'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'source_adv_loss'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'source_recons_loss'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'relation_loss'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'source_loss'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'train_accuracy')]
        )
        self.target_merged = tf.summary.merge(
            [tf.get_collection(tf.GraphKeys.SUMMARIES, 'target_diff_loss'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'target_adv_loss'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'target_recons_loss'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'target_loss')]
        )
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

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
            self.add_summary(sess)

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
                                self.relation_labels:source_y}
            _, loss_train_source, train_accuracy,summary, step_num_ = sess.run([self.source_train_op, self.source_loss,
                                                                 self.train_accuracy, self.source_merged,
                                                                 self.global_step],
                                                                # global_step每次递增1，是一个全局的step计数
                                                                feed_dict=source_feed_dict)
            self.file_writer.add_summary(summary, step_num)

            # target train
            target_feed_dict = {self.target_word_ids: target_x,
                                self.target_position_ids1: target_p1,
                                self.target_position_ids2: target_p2,
                                self.domain_labels: target_domain,
                               }

            _, loss_train_target, summary = sess.run([self.target_train_op, self.target_loss, self.target_merged],
                                                         feed_dict=target_feed_dict)

            self.file_writer.add_summary(summary, step_num)


            #save and print

            if(step + 1 == 1 or (step + 1) % 10 == 0 or step + 1 == num_batches):#log every 10 batch_steps
                self.logger.info(
                '{} epoch {}, batch_step {}/{}, source_train_loss: {:.4}, target_train_loss: {:.4}, total_train_loss: {:.4}, train_accuracy: {:.4}, global_step: {}'
                .format(start_time, epoch + 1, step + 1,num_batches,loss_train_source,loss_train_target,loss_train_source+loss_train_target, train_accuracy,step_num))

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)  # 按训练迭代轮次存储。

        #dev after every epoch
        self.logger.info('===========validation / test===========')
        # 得到预测的label列表
        label_true,label_pre = self.dev_one_epoch(sess, dev)
        self.evaluate(label_pre, label_true, epoch)

    # 在训练过程中测试
    def dev_one_epoch(self, sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        batches = dev_batch_iter(dev, self.batch_size, shuffle=False)
        true_label_list,pre_label_list = [],[]
        for step, dev in enumerate(batches):

            dev_x, dev_p1, dev_p2, dev_y = zip(*dev)
            dev_feed_dict = {self.target_word_ids: dev_x,
                             self.target_position_ids1: dev_p1,
                             self.target_position_ids2: dev_p2,
                             self.relation_labels: dev_y}
            # [batch,]
            label_pre = sess.run(self.labels_predict, feed_dict=dev_feed_dict)
            # [batch,]
            label_true = np.argmax(dev_y, axis=-1)

            true_label_list.extend(label_true)
            pre_label_list.extend(label_pre)
        print("true label:"+str(true_label_list))
        print("pre label:"+str(pre_label_list))

        return true_label_list, pre_label_list

    def evaluate(self, label_pre, label_true, epoch=None):
        """

        :param label_pre:[batch,]
        :param label_true:[batch,]
        :param epoch:
        :return:
        """
        macro_f1 = f1_score(label_true, label_pre, average="macro")
        log = 'epoch {}: macro_f1:{}'.format(epoch + 1, macro_f1)
        self.logger.info(log)


    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            self.logger.info('=========== testing ===========')
            #重新载入模型
            saver.restore(sess, self.model_path)
            label_pre, label_true = self.dev_one_epoch(sess, test)
            self.evaluate(label_pre,label_true)

    # def demo_one(self, sess, sent):
    #     """
    #
    #     :param sess:
    #     :param sent:
    #     :return:
    #     """
    #     label_list = []
    #     for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
    #         label_list_, _ = self.predict_one_batch(sess, seqs)
    #         label_list.extend(label_list_)
    #     label2tag = {}
    #     for tag, label in self.tag2label.items():
    #         label2tag[label] = tag if label != 0 else label
    #     tag = [label2tag[label] for label in label_list[0]]
    #     return tag



