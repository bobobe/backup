import sys,os
sys.path.append("..")
from models.GSN4 import GSN
#from models.GSN1 import GSN
#from models.GSN2 import GSN
from configure import args,paths,config
from inputs.data_helper import load_data,load_glove_embed_matrix,load_random_embed_matrix
import tensorflow as tf

# default GPU is 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# only cpu
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#load data
train, dev = load_data(args.train_path,args.test_path)

#load embedding
embedding = {}
if(args.pre_train):
    word_embedding = load_glove_embed_matrix(args.word_embedding_path)
else:
    word_embedding = load_random_embed_matrix(args.x_vocab_path,args.word_embedding_dim,"word embedding")
embedding["word_embedding"] = word_embedding
pos_embedding = load_random_embed_matrix(args.pos_vocab_path,args.pos_embedding_dim,"position embedding")
embedding["pos_embedding"] = pos_embedding


## training model
if args.mode == 'train':

    model = GSN(args, paths, embedding,config)
    model.build_graph()


    ## train model on the whole training data
    print("train data size: {}".format(len(train[0])))
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
