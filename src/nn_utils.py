import tensorflow as tf



def alpha_zero_residual_block(inputs, filters=64, kernel_size=3, activation=tf.nn.relu):

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

    skip = bn_2 + inputs

    relu_2 = activation(skip)

    return relu_2

def alpha_zero_conv_block(inputs, filters=64, kernel_size=3, activation=tf.nn.relu):

    conv_1 = tf.layers.conv2d(inputs=inputs,
                              filters=filters,
                              kernel_size=kernel_size,
                              padding="same",
                              activation=None)

    bn_1 = tf.layers.batch_normalization(conv_1)

    relu_1 = activation(bn_1)

    return relu_1


