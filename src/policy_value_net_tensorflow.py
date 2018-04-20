# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Tensorflow
Tested in Tensorflow 1.4 and 1.5

@author: Xiang Zhong
"""


import numpy as np
import tensorflow as tf
from src.nn_utils import alpha_zero_conv_block, alpha_zero_residual_block


class PolicyValueNet():

    def __init__(self,
                 board_width,
                 board_height,
                 name='alpha0',
                 model_file=None,
                 arch='resnet',
                 n_res_blocks=9):

        # TODO: add savedir
        # TODO: refactor the class:
        #  - add summaries
        #  - check all the methods, are all of them necessary?


        self.board_width = board_width
        self.board_height = board_height

        assert arch in ['resnet', 'orginal']

        self.arch = arch
        self.n_res_blocks = n_res_blocks

        self.l2_penalty_beta = 1e-4

        self.learning_rate = tf.placeholder(tf.float32)
        self.input_states = tf.placeholder(tf.float32, shape=[None, self.board_height, self.board_width, 4])

        if self.arch == 'resnet':
            self._res_net_builder()
        else:
            self._original_builder()

        # Define the Loss function
        # 1. Label: the array containing if the game wins or not for each state
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])

        # 3-1. Value Loss function
        self.value_loss = tf.losses.mean_squared_error(self.labels, self.value_out)

        # 3-2. Policy Loss function
        self.mcts_probs = tf.placeholder(tf.float32, shape=[None, self.board_height * self.board_width])
        # self.policy_loss = tf.negative(
        #     tf.reduce_mean(tf.reduce_sum(tf.multiply(self.mcts_probs, self.probs_out), 1)))

        self.policy_loss = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.multiply(self.mcts_probs, self.probs_out), 1)))

        # 3-3. L2 penalty (regularization)
        vars = tf.trainable_variables()
        l2_penalty = self.l2_penalty_beta * tf.add_n(
            [tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name.lower()])
        # 3-4 Add up to be the Loss function
        self.loss = self.value_loss + self.policy_loss + l2_penalty

        # calc policy entropy, for monitoring only
        self.entropy = tf.negative(tf.reduce_mean(tf.reduce_sum(tf.exp(self.probs_out) * self.probs_out, 1)))

        if self.arch == 'resnet':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = self.optimizer.minimize(self.loss)
        else:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)


        # summaries for tensorboard
        # TODO: add summaries

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        init = tf.global_variables_initializer()
        self.session.run(init)

        # For saving and restoring
        self.saver = tf.train.Saver(max_to_keep=50)
        if model_file is not None:
            self.restore_model(model_file)


    def _res_net_builder(self):

        # 2-1. Convolutiinal block
        out = alpha_zero_conv_block(self.input_states)

        # 2-2. Residual blocks
        for _ in range(self.n_res_blocks):
            out = alpha_zero_residual_block(out)

        # 3-1 Action Networks
        action_conv = tf.layers.conv2d(inputs=out,
                                       filters=2,
                                       kernel_size=[1, 1],
                                       padding="same",
                                       activation=None)

        self.action_conv = tf.nn.relu(tf.layers.batch_normalization(action_conv))

        # Flatten the tensor
        self.action_conv_flat = tf.reshape(self.action_conv, [-1, 2 * self.board_height * self.board_width])
        # 3-2 Full connected layer, the output is the log probability of moves
        # on each slot on the board
        self.probs_out = tf.layers.dense(inputs=self.action_conv_flat,
                                          units=self.board_height * self.board_width,
                                          activation=tf.nn.log_softmax)

        # 4 Evaluation Networks
        value_conv = tf.layers.conv2d(out,
                                      filters=1,
                                      kernel_size=[1, 1],
                                      padding="same",
                                      activation=None)

        self.value_conv = tf.nn.relu(tf.layers.batch_normalization(value_conv))

        self.value_conv_flat = tf.reshape(self.value_conv, [-1, self.board_height * self.board_width])
        self.value_fc1 = tf.layers.dense(inputs=self.value_conv_flat, units=64, activation=tf.nn.relu)

        # output the score of evaluation on current state
        self.value_out = tf.layers.dense(inputs=self.value_fc1,
                                         units=1,
                                         activation=tf.nn.tanh)


    def _original_builder(self):

        # 2. Common Networks Layers
        self.conv1 = tf.layers.conv2d(inputs=self.input_states,
                                      filters=32, kernel_size=[3, 3],
                                      padding="same", activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64,
                                      kernel_size=[3, 3], padding="same",
                                      activation=tf.nn.relu)
        self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128,
                                      kernel_size=[3, 3], padding="same",
                                      activation=tf.nn.relu)
        # 3-1 Action Networks
        self.action_conv = tf.layers.conv2d(inputs=self.conv3, filters=4,
                                            kernel_size=[1, 1], padding="same",
                                            activation=tf.nn.relu)
        # Flatten the tensor
        self.action_conv_flat = tf.reshape(
            self.action_conv, [-1, 4 * self.board_height * self.board_width])
        # 3-2 Full connected layer, the output is the log probability of moves
        # on each slot on the board
        self.logits_out = tf.layers.dense(inputs=self.action_conv_flat,
                                          units=self.board_height * self.board_width,
                                          activation=None)

        self.probs_out = tf.nn.softmax(self.logits_out)

        # 4 Evaluation Networks
        self.evaluation_conv = tf.layers.conv2d(inputs=self.conv3, filters=2,
                                                kernel_size=[1, 1],
                                                padding="same",
                                                activation=tf.nn.relu)
        self.evaluation_conv_flat = tf.reshape(
            self.evaluation_conv, [-1, 2 * self.board_height * self.board_width])
        self.evaluation_fc1 = tf.layers.dense(inputs=self.evaluation_conv_flat,
                                              units=64, activation=tf.nn.relu)
        # output the score of evaluation on current state
        self.evaluation_fc2 = tf.layers.dense(inputs=self.evaluation_fc1,
                                              units=1, activation=tf.nn.tanh)


    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        log_act_probs, value = self.session.run(
            [self.probs_out, self.value_out],
            feed_dict={self.input_states: state_batch}
        )

        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = board.current_state().reshape(1, self.board_width, self.board_height, 4)
        act_probs, value = self.policy_value(current_state)
        act_probs = (legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        winner_batch = np.reshape(winner_batch, (-1, 1))
        loss, entropy, _ = self.session.run(
            [self.loss, self.entropy, self.train_op],
            feed_dict={self.input_states: state_batch,
                       self.mcts_probs: mcts_probs,
                       self.labels: winner_batch,
                       self.learning_rate: lr})
        return loss, entropy

    def save_model(self, model_path):
        self.saver.save(self.session, model_path)

    def restore_model(self, model_path):
        self.saver.restore(self.session, model_path)
