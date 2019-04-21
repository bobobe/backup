import tensorflow as tf
import os
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

class FlipGradientBuilder(object):
    '''Gradient Reversal Layer from https://github.com/pumpikano/tf-dann'''

    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()  # filp_gradient is a callable object

#unpool
#
def unpool(pool, ind, ksize):
    input_shape = pool.get_shape().as_list()
    #print(input_shape)
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

    flat_input_size = np.prod(input_shape)#all element's product(total elements num)
    flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

    pool_ = tf.reshape(pool, [flat_input_size])
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
    b = tf.ones_like(ind) * batch_range
    b = tf.reshape(b, [flat_input_size, 1])
    ind_ = tf.reshape(ind, [flat_input_size, 1])
    ind_ = tf.concat([b, ind_], 1)

    ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
    ret = tf.reshape(ret, output_shape)
    return ret

def max_unpool(pool,pooled,ksize,strides):
    #最大反池化
    unpool = gen_nn_ops._max_pool_grad(pool, #池化前的tensor，即max pool的输入
                                    pooled, #池化后的tensor，即max pool 的输出
                                    pooled, #需要进行反池化操作的tensor，可以是任意shape和pool1一样的tensor
                                    ksize=ksize,
                                    strides=strides,
                                    padding='SAME')
    return unpool

import logging
def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    #make output(default is to screen) to file
    logging.basicConfig(filename = filename,format='%(asctime)s:%(levelname)s: %(message)s', level=logging.DEBUG, filemode='a')#output format

    # handler = logging.FileHandler(filename)
    # handler.setLevel(logging.DEBUG)
    # handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))#file_output format
    # logging.getLogger().addHandler(handler)

    #output to screen
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)

    return logger

# logger = get_logger('log.txt')
# logger.info("jasdfjasjdfjajsdf")
#
# logger.info("jasdfj")

#calc MMD
def MMD(X_source, X_target, max_size=2000, n_iters=10, sigma=1.0):
    def compute_pairwise_distances(x, y):
        """Computes the squared pairwise Euclidean distances between x and y.
        Args:
          x: a tensor of shape [num_x_samples, num_features]
          y: a tensor of shape [num_y_samples, num_features]
        Returns:
          a distance matrix of dimensions [num_x_samples, num_y_samples].
        Raises:
          ValueError: if the inputs do no matched the specified dimensions.
        """

        if not len(x.get_shape()) == len(y.get_shape()) == 2:
            raise ValueError('Both inputs should be matrices.')

        if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
            raise ValueError('The number of features should be the same.')

        norm = lambda x: tf.reduce_sum(tf.square(x), 1)

        # By making the `inner' dimensions of the two matrices equal to 1 using
        # broadcasting then we are essentially substracting every pair of rows
        # of x and y.
        # x will be num_samples x num_features x 1,
        # and y will be 1 x num_features x num_samples (after broadcasting).
        # After the substraction we will get a
        # num_x_samples x num_features x num_y_samples matrix.
        # The resulting dist will be of shape num_y_samples x num_x_samples.
        # and thus we need to transpose it again.
        return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

    def gaussian_kernel_matrix(x, y, sigma=1.0):
        r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
        We create a sum of multiple gaussian kernels each having a width sigma_i.
        Args:
          x: a tensor of shape [num_samples, num_features]
          y: a tensor of shape [num_samples, num_features]
          sigmas: a tensor of floats which denote the widths of each of the
            gaussians in the kernel.
        Returns:
          A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
        """
        beta = tf.expand_dims(np.array([sigma], dtype=np.float32), 1)  # / (2. * (tf.expand_dims(sigmas, 1)))

        dist = compute_pairwise_distances(x, y)

        s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

        return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

    def maximum_mean_discrepancy(x, y, sigma=1.0, kernel=gaussian_kernel_matrix):
        r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
        Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
        the distributions of x and y. Here we use the kernel two sample estimate
        using the empirical mean of the two distributions.
        MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
                    = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
        where K = <\phi(x), \phi(y)>,
          is the desired kernel function, in this case a radial basis kernel.
        Args:
            x: a tensor of shape [num_samples, num_features]
            y: a tensor of shape [num_samples, num_features]
            kernel: a function which computes the kernel in MMD. Defaults to the
                    GaussianKernelMatrix.
        Returns:
            a scalar denoting the squared maximum mean discrepancy loss.
        """
        with tf.name_scope('MaximumMeanDiscrepancy'):
            # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
            cost = tf.reduce_mean(kernel(x, x, sigma))
            cost += tf.reduce_mean(kernel(y, y, sigma))
            cost -= 2 * tf.reduce_mean(kernel(x, y, sigma))

            # We do not allow the loss to become negative.
            cost = tf.where(cost > 0, cost, 0, name='value')
        return cost
    return maximum_mean_discrepancy(X_source,X_target,sigma)




