#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/20 11:20
# @Author  : yanbo
# @Site    : 
# @File    : dataloader_test.py
# @Software: PyCharm
# @python version:
import numpy as np
import pickle
import os
import datetime
import time
import re

from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
import numpy as np
import pandas as pd
import nltk

import utils
from configure import args
import random
import tensorflow as tf
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


#nltk.download('punkt')
#os.environ['CUDA_VISIBLE_DEVICES']='1'
base_dir = "C:/Users/Admin_S_L/Desktop/fudan_mtl_copy"

x_vocab_path = base_dir+'/data/dict/x_vocab.pkl'
#x_vocab_type = 'x_vocab'

pos_vocab_path = base_dir+'/data/dict/pos_vocab.pkl'
#pos_vocab_type = 'pos_vocab'
glove_embed_path = base_dir+'/data/pretrain/glove_embed_matrix.npy'

train_path = base_dir+'/data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
text_path = base_dir+'/data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'

def clean_str(text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def load_origin_data(path):
    def get_relative_position(df, max_sentence_length):
        # Position data
        pos1 = []
        pos2 = []
        for df_idx in range(len(df)):
            sentence = df.iloc[df_idx]['sentence']
            tokens = nltk.word_tokenize(sentence)
            e1 = df.iloc[df_idx]['e1']  # the index of end symbol
            e2 = df.iloc[df_idx]['e2']

            p1 = ""
            p2 = ""
            for word_idx in range(len(tokens)):
                p1 += str((max_sentence_length - 1) + word_idx - e1) + " "
                p2 += str((max_sentence_length - 1) + word_idx - e2) + " "
            pos1.append(p1)
            pos2.append(p2)

        return pos1, pos2

    data = []
    lines = [line.strip() for line in open(path)]
    max_sentence_length = 0
    for idx in range(0, len(lines), 4):
        id = lines[idx].split("\t")[0]
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        sentence = sentence.replace('<e1>', ' _e11_ ')
        sentence = sentence.replace('</e1>', ' _e12_ ')
        sentence = sentence.replace('<e2>', ' _e21_ ')
        sentence = sentence.replace('</e2>', ' _e22_ ')

        sentence = clean_str(sentence)
        tokens = nltk.word_tokenize(sentence)
        if max_sentence_length < len(tokens):
            max_sentence_length = len(tokens)
        e1 = tokens.index("e12") - 1  # the index of every entity end symbol
        e2 = tokens.index("e22") - 1
        sentence = " ".join(tokens)

        data.append([id, sentence, e1, e2, relation])

    print("data_path: "+path)
    print("")
    print("max sentence length = {}".format(max_sentence_length))

    df = pd.DataFrame(data=data, columns=["id", "sentence", "e1", "e2", "relation"])

    pos1, pos2 = get_relative_position(df, args.max_seq_len)

    df['label'] = [utils.class2label[r] for r in df['relation']]

    # Text Data
    x_text = df['sentence'].tolist()

    # Label Data
    y = df['label']

    labels_flat = y.values.ravel()
    labels_count = np.unique(labels_flat).shape[0]
    print("relation_num: "+str(labels_count))

    # convert class labels from scalars to one-hot vectors
    # 0  => [1 0 0 0 0 ... 0 0 0 0 0]
    # 1  => [0 1 0 0 0 ... 0 0 0 0 0]
    # ...
    # 18 => [0 0 0 0 0 ... 0 0 0 0 1]

    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    labels = dense_to_one_hot(labels_flat, labels_count)
    labels = labels.astype(np.uint8)

    return x_text, pos1, pos2, labels

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

def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    #print('load '+vocab_type+"...")
    return word2id

def load_data(train_path,test_path):
    x, pos1, pos2, y = load_origin_data(train_path)
    x_test,pos1_test,pos2_test,y_test = load_origin_data(test_path)


    #build_vocab(x+x_test,x_vocab_path,'x_vocab')
    #build_vocab(pos1+pos2+pos1_test+pos2_test,pos_vocab_path,'pos_vocab')

    x_vocab = load_vocab(x_vocab_path)
    pos_vocab = load_vocab(pos_vocab_path)

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
        #print("max_len:"+str(max_len))
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

    print("position_1 = {0}".format(p1.shape))
    print("position_2 = {0}".format(p2.shape))
    print("")

    # Randomly shuffle data to split into train and test(dev)
    np.random.seed(10)#make true every time shuffle is same
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    p1_shuffled = p1[shuffle_indices]
    p2_shuffled = p2[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    #dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    s_t_split_index = 4000
    s_x,t_x = x_shuffled[:s_t_split_index],x_shuffled[s_t_split_index:]
    s_p1,t_p1 = p1_shuffled[:s_t_split_index],p1_shuffled[s_t_split_index:]
    s_p2,t_p2 = p2_shuffled[:s_t_split_index],p2_shuffled[s_t_split_index:]
    s_y,t_y = y_shuffled[:s_t_split_index],y_shuffled[s_t_split_index:]

    s_train_x = s_x
    s_train_p1 = s_p1
    s_train_p2 = s_p2
    s_train_y = s_y

    source_train = list(zip(s_train_x,s_train_p1,s_train_p2,s_train_y))

    dev_target_index = 3800
    t_train_x, t_dev_x = t_x[:dev_target_index], t_x[dev_target_index:]
    t_train_p1, t_dev_p1 = t_p1[:dev_target_index], t_p1[dev_target_index:]
    t_train_p2, t_dev_p2 = t_p2[:dev_target_index], t_p2[dev_target_index:]
    t_train_y, t_dev_y = t_y[:dev_target_index], t_y[dev_target_index:]

    target_train = list(zip(t_train_x, t_train_p1, t_train_p2, t_train_y))
    #print(np.argmax(t_dev_y,axis=1))
    dev = list(zip(t_dev_x,t_dev_p1,t_dev_p2,t_dev_y))

    print("Source/Target split: {:d}/{:d}\n".format(len(s_x), len(t_x)))
    print("Train/Dev split: {:d}/{:d}\n".format(len(t_train_x), len(t_dev_x)))

    return (source_train,target_train),dev

def glove_embed_matrix():
    '''
        Args:
            x_vocab: dictionary, 第一个元素为word, 第二个元素为索引
        Returns:
            embedding_matrix: 按照dictionary给出的索引组成的词嵌入矩阵
    '''
    x_vocab = load_vocab(x_vocab_path)
    EMBEDDING_FILE = "/home/huadong.wang/bo.yan/fudan_mtl/data/pretrain/glove.6B.100d.txt"
    with open(EMBEDDING_FILE) as f:
        # 用于将embedding的每行的第一个元素word和后面为float类型的词向量分离出来。
        # *表示后面的参数按顺序作为一个元组传进函数
        # ** 表示将调用函数时，有等号的部分作为字典传入。
        def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

        # 将所以的word作为key，numpy数组作为value放入字典
        embeddings_dict = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    #print(embeddings_dict['word'])#100 dim
    #print(len(embeddings_dict))#400000

    # 取出所有词向量，dict.values()返回的是dict_value, 要用np.stack将其转为ndarray
    all_embs = np.stack(embeddings_dict.values())
    # 计算所有元素的均值和标准差
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    # 每个词向量的长度
    embed_size = all_embs.shape[1]#100

    # 得到模型要使用的词的数量（种类）
    # len(x_vocab)是数据集中的不同词的数量
    nb_words = len(x_vocab)

    # 在高斯分布上采样， 利用计算出来的均值和标准差， 生成和预训练好的词向量相同分布的随机数组成的ndarray （假设预训练的词向量符合高斯分布）
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    # 这段循环是为了给embedding_matrix中的一些行用预训练的词向量进行赋值。
    # 因为不能保证数据集中的每个词都出现在了预训练的词向量中，所以利用预训练的词向量的均值和标准差为数据集中的词随机初始化词向量。
    # 然后再使用预训练词向量中的词去替换随机初始化数据集的词向量。
    for word, i in x_vocab.items():

        # 如果dict的key中包括word， 就返回其value。 否则返回none。
        embedding_vector = embeddings_dict.get(word)
        # 如果返回不为none，说明这个词在数据集中和训练词向量的数据集中都出现了，可以使用预训练的词向量替换随机初始化的词向量
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    # 经过上面的循环，返回的 embedding_matrix 中每行都是根据分词器的索引进行赋值的，因此之后可以直接根据词的索引取对应的词向量

    embedding_matrix = np.float32(embedding_matrix)
    np.save(glove_embed_path,embedding_matrix)
    print('save glove_embedding...')
#glove_embed_matrix()

def load_glove_embed_matrix(glove_embed_path):
    print('load glove_embedding...')
    return np.load(glove_embed_path)

def load_random_embed_matrix(vocab_path,embedding_dim,embedding_type):
    """
    :param vocab_path:
    :param embedding_dim:
    :return:
    """
    print("load "+embedding_type+"...")
    vocab = load_vocab(vocab_path)
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat

#trian_data batch_iter
def batch_iter(data, batch_size, shuffle=False):
    """
    :param data:[source,target],source:list of [text,p1,p2,y]
    :param batch_size:
    :param shuffle:
    :return:
    """
    source_train = data[0]
    target_train = data[1]

    if shuffle:
        random.shuffle(source_train)
        random.shuffle(target_train)

    len_source = len(source_train)
    len_target = len(target_train)
    #make sure len_source == len_target
    if(len_source<len_target):
        source_train.extend(source_train[:len_target-len_source])
    else:
        target_train.extend(target_train[:len_source-len_target])


    source_train_batch = []
    target_train_batch = []
    for index,sample in enumerate(source_train):

        if len(source_train_batch) == batch_size:
            yield (source_train_batch,target_train_batch)
            source_train_batch, target_train_batch = [], []

        source_train_batch.append(sample)
        target_train_batch.append(target_train[index])

    last_batch_len = len(source_train_batch)
    if last_batch_len != 0:
        #make true the last batch is the same length
        if(last_batch_len!=batch_size):
            source_train_batch.extend(source_train[:batch_size-last_batch_len])
            target_train_batch.extend(target_train[:batch_size-last_batch_len])
        yield (source_train_batch,target_train_batch)

#dev_data batch_iter
def dev_batch_iter(data, batch_size, shuffle=False):
    """
    :param data:[target],target:list of [text,p1,p2,y]
    :param batch_size:
    :param shuffle:
    :return:
    """
    dev = data

    if shuffle:
        random.shuffle(dev)


    dev_batch = []
    for index,sample in enumerate(dev):

        if len(dev_batch) == batch_size:
            yield dev_batch
            dev_batch = []

        dev_batch.append(sample)

    if len(dev_batch) != 0:
        yield dev_batch


def batch_iter_test():
    train,dev = load_data(train_path)
    bi = batch_iter(train,64)
    i = 1
    for (source_train_batch,target_train_batch) in bi:
        print(i,len(source_train_batch),len(target_train_batch))
        i+=1

#train,dev = load_data('SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT')
#print(len(glove_embed_matrix()))
#a = load_glove_embed_matrix()
