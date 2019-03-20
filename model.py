import tensorflow as tf


class RNN:
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, hidden_size, l2_reg_lambda=0.0, learning_rate=0.001):

        # Placeholders for input, output and dropout
        self.input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
        self.input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        l2_loss = tf.constant(0.0)
        text_length = self._length(self.input_text)
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        #self.params = []

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("text-embedding"):
            self.W_text = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_text")
            self.embedded_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)

        # Recurrent Neural Network
        with tf.name_scope("rnn"):
            cell = self._get_cell(hidden_size, "lstm")
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            #self.params.append(cell.trainable_variables)
            all_outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                               inputs=self.embedded_chars,
                                               sequence_length=text_length,
                                               dtype=tf.float32)
            self.h_outputs = self.last_relevant(all_outputs, text_length)

        # Dense ReLU activated layers
        with tf.name_scope("activate"):
            W1 = tf.get_variable("W1", shape=[hidden_size, 128], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[128]), name="b1")
            #self.params.append(W1)
            #self.params.append(b1)
            #l2_loss += tf.nn.l2_loss(W1)
            #l2_loss += tf.nn.l2_loss(b1)
            y1 = tf.nn.xw_plus_b(self.h_outputs, W1, b1, name="activated_1")
            h1 = tf.nn.relu(y1, name="relu1")

            W2 = tf.get_variable("W2", shape=[128, 64], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[64]), name="b2")
            #self.params.append(W2)
            #self.params.append(b2)
            #l2_loss += tf.nn.l2_loss(W2)
            #l2_loss += tf.nn.l2_loss(b2)
            y2 = tf.nn.xw_plus_b(h1, W2, b2, name="activated_2")
            self.h2 = tf.nn.relu(y2, name="relu2")

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[64, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            #self.params.append(W)
            #self.params.append(b)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h2, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

        t_opt = tf.train.AdamOptimizer(self.learning_rate)
        trainable_vars = tf.trainable_variables()
        # trainable_vars.remove(self.W_text)
        self.t_grad = tf.gradients(self.loss, trainable_vars)
        self.train_op = t_opt.apply_gradients(zip(self.t_grad, trainable_vars), global_step=self.global_step)

    @staticmethod
    def _get_cell(hidden_size, cell_type):
        if cell_type == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        elif cell_type == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        else:
            print("ERROR: '" + cell_type + "' is a wrong cell type !!!")
            return None

    # Length of the sequence data
    @staticmethod
    def _length(seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    # Extract the output of last cell of each sequence
    # Ex) The movie is good -> length = 4
    #     output = [ [1.314, -3.32, ..., 0.98]
    #                [0.287, -0.50, ..., 1.55]
    #                [2.194, -2.12, ..., 0.63]
    #                [1.938, -1.88, ..., 1.31]
    #                [  0.0,   0.0, ...,  0.0]
    #                ...
    #                [  0.0,   0.0, ...,  0.0] ]
    #     The output we need is 4th output of cell, so extract it.
    @staticmethod
    def last_relevant(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)