import tensorflow as tf
import numpy as np
import os
import datetime
import time
from model import RNN
import data_helpers
import pickle

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("pos_dir", "./cOPN_positive.txt", "Path of positive data")
tf.flags.DEFINE_string("neg_dir", "./cOPN_negative.txt", "Path of negative data")
tf.flags.DEFINE_float("test_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("max_sentence_length", 400, "Max sentence length in train/test data (Default: 100)")

# Model Hyperparameters
#tf.flags.DEFINE_string("cell_type", "lstm", "Type of rnn cell. Choose 'vanilla' or 'lstm' or 'gru' (Default: vanilla)")
tf.flags.DEFINE_boolean("GloVE", False, "Word2vec file with pre-trained embeddings")
tf.flags.DEFINE_integer("vocab_size", 1193515, "The size of vocabulary")
tf.flags.DEFINE_integer("embedding_dim", 25, "Dimensionality of character embedding (Default: 300)")
tf.flags.DEFINE_integer("hidden_size", 32, "Dimensionality of character embedding (Default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (Default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularization lambda (Default: 3.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 1024, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (Default: 100)")
tf.flags.DEFINE_integer("display_every", 10, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")


def train():
    with tf.device('/gpu:0'):
        x, y = data_helpers.load_data_and_labels(FLAGS.pos_dir, FLAGS.neg_dir)

    

    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))
    print("")

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.test_sample_percentage * float(len(y)))
    x_train, x_test = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_test = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_test)))
    del y
    del x
    del x_shuffled
    del y_shuffled
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            rnn = RNN(
                sequence_length=FLAGS.max_sentence_length,
                num_classes=y_train.shape[1],
                vocab_size=FLAGS.vocab_size,
                embedding_size=FLAGS.embedding_dim,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                learning_rate=FLAGS.learning_rate
            )

            # Define Training procedure

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            #loss_summary = tf.summary.scalar("loss", rnn.loss)
            #acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

            # Train Summaries
            #train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            #train_summary_dir = os.path.join(out_dir, "summaries", "train")
            #train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            #dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            #dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            #dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # write log
            log = open(out_dir+'/log.txt','w')

            # Write vocabulary
            #text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Pre-trained word2vec
            if FLAGS.GloVE:
                with open('./glove.twitter.27B.%dd.pkl'%FLAGS.embedding_dim, 'rb') as f:
                    data = pickle.load(f)
                    words_vec = data['words_vec']
                initW = np.array(words_vec)
                sess.run(rnn.W_text.assign(initW))
                print("Success to load pre-trained word2vec model!\n")
                data = []
                words_vec = []

            # Generate batches
            batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                # Train
                feed_dict = {
                    rnn.input_text: x_batch,
                    rnn.input_y: y_batch,
                    rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss, accuracy = sess.run([rnn.train_op, rnn.global_step, rnn.loss, rnn.accuracy], feed_dict)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    log.write("{}: step {}, loss {:g}, acc {:g} \n".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    log.write("\nEvaluation:\n")
                    feed_dict_dev = {
                        rnn.input_text: x_test[:200],
                        rnn.input_y: y_test[:200],
                        rnn.dropout_keep_prob: 1.0
                    }
                    loss, accuracy = sess.run([rnn.loss, rnn.accuracy], feed_dict_dev)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, accuracy))
                    log.write("{}: step {}, loss {:g}, acc {:g}\n\n".format(time_str, step, loss, accuracy))

                # Model checkpoint
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()