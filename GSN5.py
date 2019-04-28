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
from inputs.data_helper_amazon import batch_iter
from .model_helper import flip_gradient,max_unpool,get_logger,MMD
import tensorflow as tf
from sklearn.metrics import f1_score


class GSN(object):
    def __init__(self, args, paths,embedding,config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch_num
        self.pre_train = args.pre_train
        self.word_embeddings = embedding["word_embedding"]
        self.update_embedding = args.update_embedding
        self.optimizer = args.optimizer
        self.lr = args.learning_rate
        self.keep_prob = args.dropout_keep_prob
        self.clip_grad = args.clip_grad
        self.shuffle = args.shuffle
        self.model_path = paths["model_path"]
        self.summary_path = paths["summary_path"]
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.num_relations = args.num_relations
        self.word_emb_size = args.word_embedding_dim
        self.all_emb_size = self.word_emb_size
        self.max_seq_len = args.max_seq_len
        self.num_filters = args.num_filters
        self.best_f1 = 0.0
        self.config = config


    def build_graph(self):
        self.add_placeholders()
        source_embeddings,target_embeddings = self.add_embedding_layers()

        # target graph
        private_conv_output_target_before_pool, private_conv_output_target, _ = self.private_conv_encoder(target_embeddings, "target_private_conv_encoder")
        shared_conv_output_target_before_pool, shared_conv_output_target = self.shared_conv_encoder(target_embeddings, "shared_conv_encoder")
        # fully_connected_output = self.fully_connected_layer(shared_conv_output)
        domain_classifer_output_target = self.gradient_reversed_layer(shared_conv_output_target)
        deconv_output_target = self.deconvolution_layer(shared_conv_output_target_before_pool, private_conv_output_target_before_pool, "target_deconv_layer")
        # target_loss
        adv_loss_target = self.advers_loss(domain_classifer_output_target,"target")
        rec_loss_target = self.recons_loss(target_embeddings, deconv_output_target,"target")
        dif_loss_target = self.diff_loss(private_conv_output_target, shared_conv_output_target,"target")
        self.target_loss = 0.25 * adv_loss_target + 0.075 * dif_loss_target + 0.01 * rec_loss_target
        tf.summary.scalar("target_loss", self.target_loss)

        #source graph
        private_conv_output_source_before_pool,private_conv_output_source,_ = self.private_conv_encoder(source_embeddings,"source_private_conv_encoder")
        shared_conv_output_source_before_pool,shared_conv_output_source = self.shared_conv_encoder(source_embeddings,"shared_conv_encoder")
        fully_connected_output = self.fully_connected_layer(shared_conv_output_source,private_conv_output_source)
        domain_classifer_output_source = self.gradient_reversed_layer(shared_conv_output_source)
        deconv_output_source = self.deconvolution_layer(shared_conv_output_source_before_pool, private_conv_output_source_before_pool, "source_deconv_layer")
        #source_loss
        rel_loss = self.relation_loss(fully_connected_output,"train")
        adv_loss_source = self.advers_loss(domain_classifer_output_source,"source")
        rec_loss_source = self.recons_loss(source_embeddings,deconv_output_source,"source")
        dif_loss_source = self.diff_loss(private_conv_output_source,shared_conv_output_source,"source")
        self.train_accuracy = self.accuracy(fully_connected_output,"train")
        self.source_loss = rel_loss + 0.25*adv_loss_source + 0.075*dif_loss_source + 0.01*rec_loss_source
        tf.summary.scalar("source_loss", self.source_loss)

        #combine loss
        mmd_loss = self.mmd_loss(private_conv_output_source,private_conv_output_target)
        self.loss = rel_loss + 0.25*(adv_loss_source+adv_loss_target) + 0.075*(dif_loss_source+dif_loss_target) + 0.01*(rec_loss_source+rec_loss_target)+ 0.45*mmd_loss
        self.train_op = self.trainstep_op(self.loss)
        tf.summary.scalar("loss", self.source_loss)

        #dev graph
        _, shared_conv_output = self.shared_conv_encoder(target_embeddings, "shared_conv_encoder")
        _, private_conv_output, _ = self.private_conv_encoder(target_embeddings,"source_private_conv_encoder")
        fully_connected_output = self.fully_connected_layer(shared_conv_output,private_conv_output)#[batch,num_relations]
        rel_loss = self.relation_loss(fully_connected_output,"dev")
        self.dev_loss = rel_loss
        #[batch,]
        self.labels_predict = tf.cast(tf.argmax(fully_connected_output, axis=-1),tf.int32)
        self.dev_accuracy = self.accuracy(fully_connected_output,"dev")
        tf.summary.scalar("dev_loss", self.dev_loss)


        #init
        self.init_op()


    def add_placeholders(self):
        self.source_word_ids = tf.placeholder(tf.int32, shape=[None, self.max_seq_len], name="source_word_ids")#源域word在单词表中的下标，用于embedding_lookup层
        self.target_word_ids = tf.placeholder(tf.int32, shape=[None, self.max_seq_len], name="target_word_ids")#目标域word在单词表中的下标，用于embedding_lookup层

        self.source_domain_labels = tf.placeholder(tf.int32, shape=[None, 2], name="source_domain_labels")#数据属于的域
        self.target_domain_labels = tf.placeholder(tf.int32, shape=[None, 2], name="target_domain_labels")  # 数据属于的域
        self.relation_labels = tf.placeholder(tf.int32, shape=[None,self.num_relations], name="relation_labels")#关系种类
        self.dropout_kp = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")

    def add_embedding_layers(self):
        with tf.variable_scope("embedding"):
            _word_embeddings = tf.get_variable(initializer=self.word_embeddings,
                                                dtype=tf.float32,
                                                trainable=self.update_embedding,
                                                name="_word_embeddings")
            source_word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                            ids=self.source_word_ids,
                                                            name="source_word_embeddings")
            target_word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                            ids=self.target_word_ids,
                                                            name="target_word_embeddings")
            #[batch,T,embedding_dim]
            source_embeddings = source_word_embeddings#tf.concat([source_word_embeddings,source_position_embeddings1,source_position_embeddings2],axis=-1)
            target_embeddings = target_word_embeddings#tf.concat([target_word_embeddings,target_position_embeddings1,target_position_embeddings2],axis=-1)
            #print(source_embeddings.get_shape())
            return source_embeddings, target_embeddings

    def private_conv_encoder(self,origin_embedding_input, scope, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope,reuse=reuse):

            '''
            # [batch,T,embeding_dim,1]
            output = tf.expand_dims(origin_embedding_input, axis=-1)
            #[batch,T,1,filters]
            output = tf.nn.conv2d(input=output, filter=tf.get_variable(name="filters", shape=[3, self.all_emb_size, 1, self.num_filters],initializer=tf.contrib.layers.xavier_initializer()),
                                  strides=[1, 1, self.all_emb_size, 1],
                                  padding="SAME",use_cudnn_on_gpu=None,name = "conv")
            output = tf.nn.tanh(output)
            #[batch,T,filters,1]
            output = tf.transpose(output,[0,1,3,2])
            '''

            params = {"inputs": origin_embedding_input, "kernel_initializer": tf.contrib.layers.xavier_initializer(),
                      "bias_initializer": tf.zeros_initializer(), "filters": self.num_filters, "kernel_size": 3, "strides": 1,
                      "activation": tf.nn.tanh, "use_bias": True, "padding": 'SAME', "name": 'conv'}

            #output:[batch,T,filters]
            output = tf.layers.conv1d(**params)
            #[batch,T,filters,1]#batch,height,weight,channal
            conv_output = tf.expand_dims(output, 3)

            '''
            #[batch,1,filters,1]
            output,args = tf.nn.max_pool_with_argmax(conv_output,ksize = [1,self.max_seq_len,1,1],strides = [1,self.max_seq_len,1,1],padding = 'SAME',name = 'max_pool')
            #print(args.get_shape())
            #[batch,filters,1]
            output = tf.squeeze(output, 1)
            #[batch,filters]
            output = tf.squeeze(output, 2)

            '''

            #print(args.get_shape())
            args = tf.argmax(output, axis=1)
            args = tf.expand_dims(args,axis=1)
            output = tf.reduce_max(output, axis=1)
            #output = tf.squeeze(output,2)
            #print(args.get_shape())



            #private_conv_output = tf.squeeze(output,2)
            # [batch,T]
            private_conv_output = output
            maxpool_args = args
            #池化前，池化后，池化位置
            return conv_output, private_conv_output, maxpool_args

    def shared_conv_encoder(self,origin_embedding_input,scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

            '''
            #[batch,T,embeding_dim,1]
            output = tf.expand_dims(origin_embedding_input,axis=-1)
            # [batch,T,1,filters]
            output = tf.nn.conv2d(input=output,

                                  filter=tf.get_variable(name="filters", shape=[3, self.all_emb_size, 1, self.num_filters],initializer=tf.contrib.layers.xavier_initializer()),
                                  strides=[1, 1, self.all_emb_size, 1],
                                  padding="SAME", use_cudnn_on_gpu=None, name="conv")
            output = tf.nn.tanh(output)
            # [batch,T,filters,1]
            output = tf.transpose(output, [0, 1, 3, 2])
            '''

            params = {"inputs": origin_embedding_input, "kernel_initializer":tf.contrib.layers.xavier_initializer(),
                      "bias_initializer": tf.zeros_initializer(), "filters": self.num_filters,"kernel_size": 3, "strides": 1,
                      "activation": tf.nn.tanh, "use_bias": True, "padding": 'SAME', "name": 'conv'}

            #output:[batch,T,filters]
            output = tf.layers.conv1d(**params)
            #[batch,T,filters,1]
            conv_output = tf.expand_dims(output,3)



            #[batch,T,1,1]
            #output = tf.nn.max_pool(output,ksize = [1,1,800,1],strides = [1,1,800,1],padding = 'SAME',name = 'max_pool')
            #[batch,filters,1]
            output = tf.reduce_max(conv_output,axis=1)
            #[batch,filters]
            shared_conv_output = tf.squeeze(output, 2)
            #shared_conv_output = tf.squeeze(shared_conv_output,2)
            #print(shared_conv_output.get_shape())
            #池化前，池化后
            return conv_output, shared_conv_output


    def fully_connected_layer(self,shared_conv_output,private_conv_output):
        #shared_conv_output:[batch,filters]
        with tf.variable_scope("fc_layer",reuse=tf.AUTO_REUSE):
            #[batch,filters*2]
            #sp = tf.concat([shared_conv_output,private_conv_output],axis=1)
            sp = tf.add(shared_conv_output,private_conv_output)
            W = tf.get_variable(name="W",
                                shape=[self.num_filters, 300],
                                #通过xavier算法通过输入输出神经元的数目自动确定权值矩阵的初始化大小
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[300],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            #[batch,T]*[T,300] = [batch,300]
            output = tf.matmul(sp,W)+b
            hidden_output = tf.nn.tanh(output)#[batch,300]

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
            #dropout
            fully_connected_output = tf.nn.dropout(fully_connected_output,self.dropout_kp)
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
                                shape=[self.num_filters, 2],
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

    def deconvolution_layer(self,shared_conv_output_before_pool,private_conv_output_before_pool,scope,reuse = None):

        with tf.variable_scope(scope,reuse = reuse):

            #未池化之前的private和target表示相加
            ##[batch,T,filters,1]
            ps_b = shared_conv_output_before_pool+private_conv_output_before_pool
            # [batch,1,filters,1]
            # output, args = tf.nn.max_pool_with_argmax(ps_b, ksize=[1, self.max_seq_len, 1, 1],
            #                                           strides=[1, self.max_seq_len, 1, 1], padding='SAME',
            #                                           name='max_pool')
            args = tf.argmax(ps_b, axis=1)
            args = tf.expand_dims(args, axis=1)
            output = tf.reduce_max(ps_b, axis=1)
            output = tf.expand_dims(output, axis=1)

            # [batch,1,filters,1]
            ps = max_unpool(ps_b,output,ksize=[1,self.max_seq_len,1,1],strides=[1,self.max_seq_len,1,1])
            #ps = unpool(pool=ps, ind=maxpool_args, ksize=[1,self.max_seq_len,1,1])
            # after unpooling #[batch,T,filter,1]
            #deconvolution
            # [batch,T,1,filter]
            ps = tf.transpose(ps, [0, 1, 3, 2])

            ps = tf.nn.tanh(ps)


            # [batch,T,embedd_dim,1]
            #
            deconv_output = tf.nn.conv2d_transpose(ps, filter=tf.get_variable(name="filters", shape=[3, self.all_emb_size, 1, self.num_filters],initializer=tf.contrib.layers.xavier_initializer(),
                            dtype=tf.float32), output_shape=[self.batch_size, self.max_seq_len, self.all_emb_size, 1],
                            strides=[1, 1, self.all_emb_size, 1], padding="SAME",name="deconv")
            # [batch,T,embedding_dim]
            deconv_output = tf.squeeze(deconv_output,-1)
            return deconv_output

    def relation_loss(self,fully_connected_output,catalog):
        with tf.variable_scope(catalog+"_relation_loss", reuse=None):
            # logits为[batch_size, num_relations]
            # labels为[batch_size,1](真实标记）
            # [batch,1]
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=fully_connected_output,
                                                                  labels=self.relation_labels)  # labels为真
            # [1],batch average loss
            rel_loss = tf.reduce_mean(loss)
            tf.summary.scalar(catalog+"_relation_loss",rel_loss)
            return rel_loss

    def advers_loss(self,domain_classifer_output,domain):
        with tf.variable_scope(domain+"_adv_loss", reuse=None):
            if(domain=="source"):
                adv_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.source_domain_labels, logits=domain_classifer_output))
                tf.summary.scalar(domain + "_adv_loss", adv_loss)
                return adv_loss
            else:
                adv_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.target_domain_labels, logits=domain_classifer_output))
                tf.summary.scalar(domain + "_adv_loss", adv_loss)
                return adv_loss

    def recons_loss(self,origin_embeddings_input,deconv_output,domain):
        with tf.variable_scope(domain + "_recons_loss", reuse=None):

            #[batch,T,embedding_dim]
            origin_input = origin_embeddings_input
            #norm
            #[batch,T]
            #input = origin_input-deconv_output
            #rec_loss = tf.nn.l2_loss(input)

            recons_norm = tf.sqrt(tf.reduce_sum(tf.square(deconv_output), axis=2))
            origin_norm = tf.sqrt(tf.reduce_sum(tf.square(origin_input), axis=2))
            #dot product
            #[batch,T]
            dp = tf.reduce_sum(tf.multiply(deconv_output, origin_input), axis=2)
            #cos , [batch,T]
            res = dp / (recons_norm * origin_norm+1e-04)
            #recons loss
            rec_loss = 1 - tf.reduce_sum(tf.abs(res), axis=1)

            #batch average
            rec_loss = tf.reduce_mean(rec_loss)
            tf.summary.scalar(domain+"_recons_loss", rec_loss)
            return rec_loss

    def diff_loss(self,private_conv_output,shared_conv_output,domain):
        with tf.variable_scope(domain+"_diff_loss", reuse=None):

            # p = private_conv_output
            # s = shared_conv_output
            # output = tf.norm(
            #     tf.matmul(tf.transpose(p, [1, 0]),s),#[batch,T,1]*[batch,1,T]=[batch,T,T]
            #     axis=[-2,-1]#batch first
            # )
            #     #[batch,]
            # dif_loss = tf.square(output)#[batch,]
            # dif_loss = tf.reduce_mean(dif_loss)
            # tf.summary.scalar(domain+"_diff_loss",dif_loss)
            # return dif_loss

            #[batch,filters,1]
            p = tf.expand_dims(private_conv_output,axis=2)
            s = tf.expand_dims(shared_conv_output,axis=2)
            # [batch,]
            output = tf.norm(
                tf.matmul(p, tf.transpose(s, [0, 2, 1])),#[batch,T,1]*[batch,1,T]=[batch,T,T]
                axis=[-2,-1]#batch first
            )

            dif_loss = tf.square(output)#[batch,]
            dif_loss = tf.reduce_mean(dif_loss)
            tf.summary.scalar(domain+"_diff_loss", dif_loss)
            return dif_loss

    def mmd_loss(self,source_private_conv_output,target_private_conv_output):
        with tf.variable_scope("mmd_loss", reuse=None):
            mmd_loss = MMD(source_private_conv_output, target_private_conv_output)
            tf.summary.scalar("mmd_loss", mmd_loss)
            return mmd_loss

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
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr,rho=0.95,epsilon=1e-08,)
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
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'train_relation_loss'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'source_loss'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'train_accuracy'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'mmd_loss'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'loss')]
        )
        self.target_merged = tf.summary.merge(
            [tf.get_collection(tf.GraphKeys.SUMMARIES, 'target_diff_loss'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'target_adv_loss'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'target_recons_loss'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'target_loss')]
        )
        self.dev_merged = tf.summary.merge(
            [tf.get_collection(tf.GraphKeys.SUMMARIES, 'dev_accuracy'),
             tf.get_collection(tf.GraphKeys.SUMMARIES, 'dev_relation_loss')]
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

            best_f1 = 0.0
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

            source_x, source_y = zip(*source_batch)
            target_x, target_y = zip(*target_batch)
            source_domain = np.array([[1,0]]*len(source_x))
            target_domain = np.array([[0,1]]*len(target_x))

            #source train
            step_num = epoch * num_batches + step + 1
            source_feed_dict = {self.source_word_ids: source_x,
                                self.source_domain_labels: source_domain,

                                self.target_word_ids: target_x,
                                self.target_domain_labels: target_domain,

                                self.relation_labels: source_y,
                                self.dropout_kp: self.keep_prob}
            _, loss,loss_train_target,loss_train_source, train_accuracy, summary_source,summary_target, step_num_ = sess.run([self.train_op, self.loss,
                                                                                 self.target_loss,self.source_loss,
                                                                                self.train_accuracy, self.source_merged,
                                                                                self.target_merged,self.global_step],
                                                                                # global_step每次递增1，是一个全局的step计数
                                                                                feed_dict=source_feed_dict)

            #print(source_ps[0])
            self.file_writer.add_summary(summary_source, step_num)
            self.file_writer.add_summary(summary_target, step_num)


            #save and print

            if(step + 1 == 1 or (step + 1) % 10 == 0 or step + 1 == num_batches):#log every 10 batch_steps
                self.logger.info(
                '{} epoch {}, batch_step {}/{}, loss: {:.4}, source_train_loss: {:.4}, target_train_loss: {:.4}, train_acccuracy: {:.4}, global_step: {}'
                .format(start_time, epoch + 1, step + 1,num_batches, loss, loss_train_source,loss_train_target,train_accuracy, step_num))

        #dev after every epoch
        self.logger.info('===========validation / test===========')
        # dev
        f1 = self.dev_one_epoch(sess, dev, epoch)
        if f1 > self.best_f1:
            self.best_f1 = f1
            saver.save(sess, self.model_path + "-{:.5g}".format(self.best_f1), global_step=epoch)  # 按训练迭代轮次存储。

    # 在训练过程中测试
    def dev_one_epoch(self, sess, dev, epoch):
        """

        :param sess:
        :param dev:
        :return:
        """
        batches = batch_iter(dev, self.batch_size, shuffle=False)
        true_label_list,pre_label_list = [], []
        dev_acc = 0
        dev_los = 0
        num = 0
        num_batches = (len(dev) + self.batch_size - 1) // self.batch_size
        for step, dev in enumerate(batches):
            step_num = epoch * num_batches + step + 1

            _,(dev_x, dev_y), = zip(*dev)
            dev_feed_dict = {self.target_word_ids: dev_x,
                             self.relation_labels: dev_y,
                             self.dropout_kp: 1.0}
            # [batch,]
            label_pre,dev_loss, dev_accuracy, summary = sess.run([self.labels_predict, self.dev_loss, self.dev_accuracy,
                                                                  self.dev_merged],feed_dict=dev_feed_dict)
            # [batch,]
            label_true = np.argmax(dev_y, axis=-1)

            self.file_writer.add_summary(summary, step_num)
            print(label_pre)
            true_label_list.extend(label_true)
            pre_label_list.extend(label_pre)
            dev_acc+=dev_accuracy
            dev_los+=dev_loss
            num+=1
        dev_acc = dev_acc/(num+1.0)
        dev_los = dev_los/(num+1.0)
        print("true label:"+str(true_label_list))
        print("pre label:"+str(pre_label_list))

        macro_f1 = f1_score(true_label_list, pre_label_list, average="macro")
        self.logger.info('epoch {},  dev_loss: {:.4}, dev_accuracy: {:.4},macro_f1: {:.4}'
                         .format(epoch + 1, dev_los, dev_acc,macro_f1))
        return macro_f1


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



