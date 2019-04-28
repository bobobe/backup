import sys,os
sys.path.append("..")
from models.GSN5 import GSN
#from models.GSN1 import GSN
#from models.GSN2 import GSN
from configure2 import args,paths,config
from inputs.data_helper_amazon import load_data,load_glove_embed_matrix,load_random_embed_matrix
import tensorflow as tf

# default GPU is 1
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# only cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

base_data_dir = 'E:/pycharm/PyCharm2017.2.3/workspace/fudan_mtl_copy/data/Amazon_Review_Dataset/'
source_train_path = base_data_dir+'books.task.train'
source_test_path = base_data_dir+'books.task.test'
target_train_path = base_data_dir+'dvd.task.train'
target_test_path = base_data_dir+'dvd.task.test'
source_path = (source_train_path, source_test_path)
target_path = (target_train_path, target_test_path)
#load data
source, target = load_data(source_path, target_path)
(source_train,source_dev, source_test), (target_train, target_dev, target_test) = source,target
train = (source_train, target_train)
dev = (source_dev, target_dev)

#load embedding
embedding = {}
if(args.pre_train):
    word_embedding = load_glove_embed_matrix(args.word_embedding_path)
else:
    word_embedding = load_random_embed_matrix(args.x_vocab_path,args.word_embedding_dim,"word embedding")
embedding["word_embedding"] = word_embedding


## training model
if args.mode == 'train':

    model = GSN(args, paths, embedding, config)
    model.build_graph()


    ## train model on the whole training data
    print("source_train data size: {}".format(len(train[0])))
    print("target_train data size: {}".format(len(train[1])))
    model.train(train=train, dev=dev)  # use test_data as the dev_data to see overfitting phenomena




## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        while(1):
            print('Please input your sentence:')
            demo_sent = input()
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                tag = model.demo_one(sess, demo_data)
                PER, LOC, ORG = get_entity(tag, demo_sent)
                print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))
