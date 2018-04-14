import tensorflow as tf



def alpha_zero_residual_block(inputs, filters=64, kernel_size=(3, 3), activation=tf.nn.relu):

    conv_1 = tf.layers.conv2d(inputs=inputs,
                              filters=filters,
                              kernel_size=kernel_size,
                              padding="same",
                              activation=None)

    bn_1 = tf.layers.batch_normalization(conv_1)

    relu_1 = activation(bn_1)

    conv_2 = tf.layers.conv2d(inputs=relu_1,
                              filters=filters,
                              kernel_size=kernel_size,
                              padding="same",
                              activation=None)

    bn_2 = tf.layers.batch_normalization(conv_2)

    skip = tf.add(bn_2, inputs)

    relu_2 = activation(skip)

    return relu_2

def alpha_zero_conv_block(inputs, filters=64, kernel_size=(3, 3), activation=tf.nn.relu):

    conv_1 = tf.layers.conv2d(inputs=inputs,
                              filters=filters,
                              kernel_size=kernel_size,
                              padding="same",
                              activation=None)

    bn_1 = tf.layers.batch_normalization(conv_1)

    relu_1 = activation(bn_1)

    return relu_1


class lr_schedule(object):

    def __init__(self, intervals=None):

        if intervals is not None:
            self.intervals = intervals
        else:
            self.intervals = {1000: 1e-2, 1200: 1e-3, 9001: 1e-4}

    def __call__(self, iter):

        for i, lr in self.intervals.items():
            if iter < i:
                return lr




