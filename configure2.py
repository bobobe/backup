import argparse
import sys
import os
import time
#from src.models.model_helper import get_logger
import tensorflow as tf

def parse_args():

    #session config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8

    base_dir = "E:/pycharm/PyCharm2017.2.3/workspace/fudan_mtl_copy"

    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    #path
    parser.add_argument("--source_path", default=base_dir+"/data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT",
                        type=str, help="Path of train data")
    parser.add_argument("--target_path", default=base_dir+"/data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT",
                        type=str, help="Path of test data")
    parser.add_argument("--output_path",
                        default=base_dir+"/saved",
                        type=str, help="path of save models")
    parser.add_argument("--x_vocab_path",
                        default=base_dir+'/data/dict/amazon_vocab.pkl',
                        type=str, help="path of save x_vocab")

    #Data loading params
    parser.add_argument("--max_seq_len", default=500,
                        type=int, help="Max sentence length in data")
    parser.add_argument("--dev_sample_percentage", default=0.1,
                        type=float, help="Percentage of the training data to use for validation")

    # Model Hyper-parameters
    parser.add_argument("--mode", default="train",
                        type=str, help="model mode")
    # Embeddings
    parser.add_argument("--pre_train", default=True,
                        type=bool, help="use pre_train word_embedding or not")
    parser.add_argument("--word_embedding_path", default=base_dir+"/data/pretrain/glove_embed_matrix_amazon.npy",
                        type=str, help="Path of pre-trained word embeddings (glove)")
    parser.add_argument("--word_embedding_dim", default=100,
                        type=int, help="Dimensionality of word embedding (default: 100)")
    parser.add_argument("--pos_embedding_dim", default=25,
                        type=int, help="Dimensionality of relative position embedding (default: 25)")
    parser.add_argument("--update_embedding", default=True,
                       type=bool, help="update word_embedding or not")
    # CNN
    parser.add_argument("--filter_sizes", default=3,
                        type=str, help="filter sizes (Default: 3)")
    parser.add_argument("--num_filters", default=800,
                        type=int, help="Number of filters per filter size (Default: 800)")
    # relation
    parser.add_argument("--num_relations", default=2,
                        type=int, help="Number of relations")

    # Training parameters
    parser.add_argument("--batch_size", default=40,
                        type=int, help="Batch Size (default: 32)")
    parser.add_argument("--learning_rate", default=0.001,
                        type=float, help="Which learning rate to start with (Default: 1.0)")
    parser.add_argument("--dropout_keep_prob", default=0.6,
                        type=float, help="dropout rate (Default: 0.5)")
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')

    parser.add_argument('--clip_grad', type=float, default=5.0, help='gradient clipping')
    parser.add_argument("--shuffle", default=True,
                        type=bool, help="shuffle data or not")
    parser.add_argument("--epoch_num", default=50,
                        type=int, help="epoch num")

    # Testing parameters
    parser.add_argument("--checkpoint_dir", default="",
                        type=str, help="Checkpoint directory from training run")

    # Misc Parameters
    parser.add_argument("--allow_soft_placement", default=True,
                        type=bool, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", default=False,
                        type=bool, help="Log placement of ops on devices")
    parser.add_argument("--gpu_allow_growth", default=True,
                        type=bool, help="Allow gpu memory growth")

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    #model path
    paths = {}
    timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
    output_path = os.path.join(args.output_path, timestamp)
    if not os.path.exists(output_path): os.makedirs(output_path)
    #summary_path
    summary_path = os.path.join(output_path, "summaries")
    paths['summary_path'] = summary_path
    if not os.path.exists(summary_path): os.makedirs(summary_path)
    #model_path
    model_path = os.path.join(output_path, "checkpoints/")
    if not os.path.exists(model_path): os.makedirs(model_path)
    ckpt_prefix = os.path.join(model_path, "model")
    paths['model_path'] = ckpt_prefix
    result_path = os.path.join(output_path, "results")
    paths['result_path'] = result_path
    if not os.path.exists(result_path): os.makedirs(result_path)
    log_path = os.path.join(result_path, "log.txt")
    paths['log_path'] = log_path
    print(str(args)+"\n")

    return args,paths,config

args,paths,config = parse_args()
