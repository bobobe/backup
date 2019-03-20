#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/20 11:20
# @Author  : yanbo
# @Site    : 
# @File    : dataloader_test.py
# @Software: PyCharm
# @python version:
import tensorflow as tf
import numpy as np
import pickle
import os
import datetime
import time

from text_cnn import TextCNN
import data_helpers
import utils
from configure import FLAGS

from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
def load_data():
    _,_,all_data = data_helpers.load_data_and_labels(FLAGS.train_path)
    # source_x,source_pos1,source_pos2,source_y = source_data
    # target_x, target_pos1, target_pos2, target_y = target_data
    x,pos1,pos2,y = all_data
    # a = x[0].split()
    # for d in a:
    #     print(d)

    def build_vocab(data,vocab_path,vocab_type):
        data_dict = {}
        data_dict['<PAD>'] = 0
        data_dict['<UNK>'] = 1
        i=2
        for line in data:
            for w in line.split():
                if(w not in data_dict.keys()):
                    data_dict[w] = i
                    i+=1
        print(vocab_type+" vocab_size:"+str(len(data_dict)))
        with open(vocab_path, 'wb') as fw:
            pickle.dump(data_dict, fw)

    def load_vocab(vocab_path,vocab_type):
        with open(vocab_path, 'rb') as fr:
            word2id = pickle.load(fr)
        print('load '+vocab_type)
        return word2id

    # build_vocab(x,'./x_vocab.pkl','x_vocab')
    # build_vocab(pos1+pos2,'./pos_vocab.pkl','pos_vocab')

    x_vocab = load_vocab('./x_vocab.pkl',"x_vocab")
    pos_vocab = load_vocab('./pos_vocab.pkl', "pos_vocab")

    def data2id(data, vocab):
        data2id = []
        for line in data:
            line_list = []
            for w in line.split():
                if w not in vocab.keys():
                    w = '<UNK>'
                line_list.append(vocab[w])
            data2id.append(line_list)
        return data2id

    def pad_data(data, pad_mark=0):
        """

        :param sequences:
        :param pad_mark:
        :return:填充后的seq列表，以及未填充前每个句子的长度
        """
        # map(function,paramter),对每个paramter调用function
        max_len = max(map(lambda x: len(x), data))  # 返回所有句子中最大的长度
        print("max_len:"+str(max_len))
        seq_list, seq_len_list = [], []
        for seq in data:
            seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))
        return np.array(seq_list), np.array(seq_len_list)

    x = data2id(x,x_vocab)
    x,x_len= pad_data(x)
    p1 = data2id(pos1,pos_vocab)
    p1,p1_len = pad_data(p1)
    p2 = data2id(pos2,pos_vocab)
    p2,p2_len = pad_data(p2)

    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))
    print("")

    print("position_1 = {0}".format(p1.shape))
    print("position_2 = {0}".format(p2.shape))
    print("")

    # Randomly shuffle data to split into train and test(dev)
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    p1_shuffled = p1[shuffle_indices]
    p2_shuffled = p2[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    #dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    s_t_split_index = 4000
    dev_percentage = 0.1
    s_x,t_x = x_shuffled[:s_t_split_index],x_shuffled[s_t_split_index:]
    s_p1,t_p1 = p1_shuffled[:s_t_split_index],p1_shuffled[s_t_split_index:]
    s_p2,t_p2 = p2_shuffled[:s_t_split_index],p2[s_t_split_index:]
    s_y,t_y = y_shuffled[:s_t_split_index],y_shuffled[s_t_split_index:]

    s_train_x = s_x
    s_train_p1 = s_p1
    s_train_p2 = s_p2
    s_train_y = s_y

    dev_target_index = 1-int(dev_percentage * float(len(t_x)))
    t_train_x, t_dev_x = t_x[:dev_target_index], t_x[dev_target_index:]
    t_train_p1, t_dev_p1 = t_p1[:dev_target_index], t_p1[dev_target_index:]
    t_train_p2, t_dev_p2 = t_p2[:dev_target_index], t_p2[dev_target_index:]
    t_train_y, t_dev_y = t_y[:dev_target_index], t_y[dev_target_index:]

    print("Source/Target split: {:d}/{:d}\n".format(len(s_x), len(t_x)))
    print("Train/Dev split: {:d}/{:d}\n".format(len(t_train_x), len(t_dev_x)))

def glove_embed_matrix():


load_data()