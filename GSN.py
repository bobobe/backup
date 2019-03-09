#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/9 16:20
# @Author  : yanbo
# @Site    : 
# @File    : GSN.py
# @Software: PyCharm
# @python version:
import numpy as np
import os, time, sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield
from utils import get_logger
from eval import conlleval


class GSN(object):
    def __init__(self, args, tag2label, vocab, paths, config):
        self.batch_size = args["batch_size"]
        self.epoch_num = args["epoch_num"]
        self.hidden_dim = args["hidden_dim"]
        self.word_embeddings = args["word_embeddings"]
        self.position_embeddings = args["position_embeddings"]
        self.update_embedding = args["update_embedding"]
        self.dropout_keep_prob = args["dropout"]
        self.optimizer = args["optimizer"]
        self.lr = args["lr"]
        self.clip_grad = args["clip"]
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args["shuffle"]
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config
        self.num_relations = args["num_relations"]
        self.postion_emb_size  = args["postion_emb_size"]
        self.word_emb_size = args["word_emb_size"]

    def build_graph(self):
        self.add_placeholders()
        self.add_embedding_layers()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        self.source_word_ids = tf.placeholder(tf.int32, shape=[None, self.word_emb_size], name="source_word_ids")#源域word在单词表中的下标，用于embedding_lookup层
        self.target_word_ids = tf.placeholder(tf.int32, shape=[None, self.word_emb_size], name="target_word_ids")#目标域word在单词表中的下标，用于embedding_lookup层

        self.source_position_ids1 = tf.placeholder(tf.int32, shape=[None, self.postion_emb_size], name="source_position_ids1")#源域每个单词离实体的距离1
        self.source_position_ids2 = tf.placeholder(tf.int32, shape=[None, self.postion_emb_size], name="source_position_ids2")  # 源域每个单词离实体的距离2
        self.target_position_ids1 = tf.placeholder(tf.int32, shape=[None, self.postion_emb_size],name="target_position_ids1")  # 源域每个单词离实体的距离1
        self.target_position_ids2 = tf.placeholder(tf.int32, shape=[None, self.postion_emb_size],name="target_position_ids2")  # 源域每个单词离实体的距离2

        self.domain_labels = tf.placeholder(tf.int32, shape=[None, 2], name="domain_labels")#数据属于的域
        self.relation_labels = tf.placeholder(tf.int32, shape=[None, self.num_relations], name="relation_labels")#关系种类

    def add_embedding_layers(self):
        with tf.variable_scope("embedding"):
            _word_embeddings = tf.Variable(self.word_embeddings,
                                                dtype=tf.float32,
                                                trainable=self.update_embedding,
                                                name="_word_embeddings")
            _position_embeddings = tf.Variable(self.position_embeddings,
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
            self.source_embeddings = tf.concat([source_word_embeddings,source_position_embeddings1,source_position_embeddings2],axis=-1)
            self.target_embeddings = tf.concat([target_word_embeddings,target_position_embeddings1,target_position_embeddings2],axis=-1)

    def private_cnn_encoder(self):
    def shared_cnn_encoder(self):
    def fully_connected_layer(self):
    def gradient_reversed_layer(self):
    def deconvolution_layer(self):



    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            #类比单向lstm，双向lstm返回两个元组，前向后向的每一步输出，和前向后向的最后一个状态值
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            #output把双向lstm每一步的输出拼接
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            #拼接完之后再接一层dropout（dropout后维数不变，只是有的单元会被置0）
            output = tf.nn.dropout(output, self.dropout_pl)

        #从双向lstm的输出到输出tag概率之间的W，b参数
        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                #通过xavier算法通过输入输出神经元的数目自动确定权值矩阵的初始化大小
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])#第一维大小看第二维
            pred = tf.matmul(output, W) + b

            #logits用于crf的输入和softmax的输入
            self. logits = tf.reshape(pred, [-1, s[1], self.num_tags])

    def loss_op(self):
        if self.CRF:
            #使用tensorflow实现好的crf
            #参数
            # inputs为[batch_size,max_seq_len,num_tags]
            #tag_indices:标签索引[batch_size,max_seq_len]
            #transition_params[num_tags,num_tags]转换矩阵，可以自己提供作为参数，也可以通过训练作为返回值返回
            #返回值
            #对数似然log_likelihood,转移矩阵
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,

                                                              sequence_lengths=self.sequence_lengths)
            #取所有样本对数似然平均值的相反数作为损失。（因为对数似然本身为负)
            self.loss = -tf.reduce_mean(log_likelihood)

        else:#softmax
            # logits为[batch_size,max_seq_len,num_tags]
            #labels为[batch_size,max_seq_len](真实标记）
            #losses是一个向量,每一步的损失
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)#label为真实标记
            #sequence_mask看下面的连接，返回Boolean值列表
            #https://www.w3cschool.cn/tensorflow_python/tensorflow_python-2elp2jns.html
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            #这个mask作用在losses上大致就是不计入那些填充的字的损失
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:#默认SGD
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            #每次梯度更新前计算梯度，返回(梯度，变量)的列表，即每个变量wi的梯度值
            grads_and_vars = optim.compute_gradients(self.loss)
            #梯度修剪
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            #global_step每次加1，表示迭代次数？？
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):#训练集和验证集
        """

        :param train:
        :param dev:
        :return:
        """
        #golbal_variables()返回所有变量域中的变量，如果加scope参数，则返回特定变量域的变量
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op) #也可以.runs()
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            #重新载入模型
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)

    def demo_one(self, sess, sent):
        """

        :param sess:
        :param sent:
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    #一个epoch会分批训练所有样本
    def run_one_epoch(self, sess, train, dev, epoch, saver):
        """

        :param sess:
        :param train:
        :param dev:
        :param epoch:
        :param saver:
        :return:
        """
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        #batches是按batch_size划分的一个迭代器，包含了全部的样本
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):
            #sys.stdout.write就是输出到屏幕
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)#获得网络的所有输入feed_dict
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],#global_step每次递增1，是一个全局的step计数
                                                         feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info('{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)#按训练迭代轮次存储。

        self.logger.info('===========validation / test===========')
        #得到预测的label列表
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)
    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """

        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict（返回参数字典和每个句子的真实长度seq_len_list
        """
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)#填充0

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)#填充0
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    #在训练过程中测试
    def dev_one_epoch(self, sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)#返回预测的标签
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            #logits和transition_params是训练得到的。用于带入s函数求最大得分的标记序列
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):#对每个测试样本
                #tensorflow集成了用于bilstm+crf解码的函数
                #返回参数是最高分的标签索引列表，和最高得分。
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)#对每个样本进行维特比解码
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:#softmax
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """

        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if  len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])#word，真实标签，预测标签
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)

