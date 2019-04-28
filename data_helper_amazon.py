import re
import numpy as np
import csv
import nltk
import pandas as pd
import pickle
import random

base_data_dir = 'E:/pycharm/PyCharm2017.2.3/workspace/fudan_mtl_copy/data/Amazon_Review_Dataset/'
source_train_path = base_data_dir+'books.task.train'
source_test_path = base_data_dir+'books.task.test'
target_train_path = base_data_dir+'dvd.task.train'
target_test_path = base_data_dir+'dvd.task.test'
source_path = (source_train_path,source_test_path)
target_path = (target_train_path,target_test_path)
x_vocab_path = 'E:/pycharm/PyCharm2017.2.3/workspace/fudan_mtl_copy/data/dict/amazon_vocab.pkl'
glove_embed_path = 'E:/pycharm/PyCharm2017.2.3/workspace/fudan_mtl_copy/data/pretrain/glove_embed_matrix_amazon.npy'
max_len = 500
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def get_raw_data(full_filename):
    data_train = open(full_filename, 'r')
    reader = csv.reader(data_train, delimiter='\t')
    data = []
    max_len = 0
    for i, row in enumerate(reader):
        try:
            descr = clean_str(row[1])
            tokens = nltk.word_tokenize(descr)
            if max_len < len(tokens):
                max_len = len(tokens)
            sentence = " ".join(tokens)
            #X.append(descr)
            #Y.append(int(row[0]))
            data.append([sentence, int(row[0])])
        except Exception as e:
            print(e)

    print("data_path: " + full_filename)
    print("max sentence length = {}".format(max_len))

    df = pd.DataFrame(data=data, columns=["sentence", "label"])

    # Text Data
    x_text = df['sentence'].tolist()

    # Label Data
    y = df['label']

    labels_flat = y.values.ravel()
    labels_count = np.unique(labels_flat).shape[0]
    print("label_num: " + str(labels_count))
    print(" ")
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

    return x_text, labels

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

def load_data(source_path,target_path):
    (source_train_path, source_test_path) = source_path
    (target_train_path, target_test_path) = target_path

    source_train_x, source_train_y = get_raw_data(source_train_path)
    source_test_x, source_test_y = get_raw_data(source_test_path)
    target_train_x, target_train_y = get_raw_data(target_train_path)
    target_test_x, target_test_y = get_raw_data(target_test_path)

    all_x = source_train_x+source_test_x+target_train_x+target_test_x

    #build_vocab(all_x,x_vocab_path,'x_vocab')

    x_vocab = load_vocab(x_vocab_path)

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

    def pad_or_truncate_data(data, max_len, pad_mark=0):
        """
        大于max_len的被截断，小于max_len的填充pad_mark
        :param sequences:
        :param pad_mark:
        :return:填充后的seq列表，以及未填充前每个句子的长度
        """
        # map(function,paramter),对每个paramter调用function
        #print("max_len:"+str(max_len))
        seq_list, seq_len_list = [], []
        for seq in data:
            seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)#pad or truncate
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))
        return np.array(seq_list), np.array(seq_len_list)

    def max_seq_len(all_x):
        return max(map(lambda x: len(x.split()), all_x))

    true_max_len = max_seq_len(all_x)
    print('true_max_len: '+str(true_max_len))

    source_valid_x, source_valid_y = source_train_x[-200:], source_train_y[-200:]
    source_train_x, source_train_y = source_train_x[:-200], source_train_y[:-200]

    source_train_x = data2id(source_train_x, x_vocab)
    source_train_x, source_train_x_len = pad_or_truncate_data(source_train_x,max_len)
    source_valid_x = data2id(source_valid_x,x_vocab)
    source_valid_x, source_valid_x_len = pad_or_truncate_data(source_valid_x,max_len)
    source_test_x = data2id(source_test_x, x_vocab)
    source_test_x, source_test_x_len = pad_or_truncate_data(source_test_x, max_len)

    target_valid_x, target_valid_y = target_train_x[-200:], target_train_y[-200:]
    target_train_x, target_train_y = target_train_x[:-200], target_train_y[:-200]

    target_train_x = data2id(target_train_x, x_vocab)
    target_train_x, target_train_x_len = pad_or_truncate_data(target_train_x, max_len)
    target_valid_x = data2id(target_valid_x,x_vocab)
    target_valid_x, target_valid_x_len = pad_or_truncate_data(target_valid_x, max_len)
    target_test_x = data2id(target_test_x, x_vocab)
    target_test_x, target_test_x_len = pad_or_truncate_data(target_test_x, max_len)

    print("source_x_train = {0}".format(source_train_x.shape))
    print("source_x_valid = {0}".format(source_valid_x.shape))
    print("source_x_test = {0}".format(source_test_x.shape))

    print("target_x_train = {0}".format(target_train_x.shape))
    print("target_x_valid = {0}".format(target_valid_x.shape))
    print("target_x_test = {0}".format(target_test_x.shape))

    print("source_y_train = {0}".format(source_train_y.shape))
    print("source_y_valid = {0}".format(source_valid_y.shape))
    print("source_y_test = {0}".format(source_test_y.shape))

    print("target_y_train = {0}".format(target_train_y.shape))
    print("target_y_valid = {0}".format(target_valid_y.shape))
    print("target_y_test = {0}".format(target_test_y.shape))

    print("")

    source_train = list(zip(source_train_x, source_train_y))
    source_dev = list(zip(source_valid_x, source_valid_y))
    source_test = list(zip(source_test_x, source_test_y))

    target_train = list(zip(target_train_x, target_train_y))
    target_dev = list(zip(target_valid_x, target_valid_y))
    target_test = list(zip(target_test_x, target_test_y))

    return (source_train,source_dev,source_test), (target_train, target_dev, target_test)

def glove_embed_matrix():
    '''
        Args:
            x_vocab: dictionary, 第一个元素为word, 第二个元素为索引
        Returns:
            embedding_matrix: 按照dictionary给出的索引组成的词嵌入矩阵
    '''
    x_vocab = load_vocab(x_vocab_path)
    EMBEDDING_FILE = "E:/pycharm/PyCharm2017.2.3/workspace/data/glove.6B.100d.txt"
    with open(EMBEDDING_FILE, encoding='UTF-8') as f:
        # 用于将embedding的每行的第一个元素word和后面为float类型的词向量分离出来。
        # *表示后面的参数按顺序作为一个元组传进函数
        # ** 表示将调用函数时，有等号的部分作为字典传入。
        def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

        # 将所以的word作为key，numpy数组作为value放入字典
        embeddings_dict = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,encoding='UTF-8'))
    #print(embeddings_dict['the'])#100 dim
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
def batch_iter(data, batch_size, shuffle=True):
    """
    :param data:[source,target],source:list of [text,y]
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
        yield (source_train_batch,target_train_batch)



# X, y = get_raw_data(base_data_dir+'books.task.train')
# print(np.shape(X))
# print(type(y))
#source,target = load_data(source_path,target_path)
#print(source[0][4])