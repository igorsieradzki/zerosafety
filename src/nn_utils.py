
from src import data_dir

import tensorflow as tf
import numpy as np
import os

import pdb

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


def load_data(file_name):
    data = np.load(os.path.join(data_dir, file_name))

    return data['args'], data['states'], data['probs'], data['winners']


class Dataset(object):
    def __init__(self,
                 n_samples : int,
                 file_name: str,
                 default_bs: int,
                 augument : bool = True,
                 shuffle: bool = True,
                 seed: int = 42):

        self.n_samples = n_samples
        self.file_name = file_name

        self.default_bs = default_bs
        self.shuffle = shuffle
        self.augument = augument

        self._load()

        self.current_epoch = 0
        self.index_in_epoch = 0

        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @property
    def batch_size(self):
        return self.default_bs

    @property
    def batches_in_epoch(self):
        return (self.size // self.default_bs) + 1

    @property
    def current_batch_in_epoch(self):
        return (self.index_in_epoch // self.default_bs) + 1

    def get_progress(self):
        return "Epoch: [{0:2d}] [{1:4d}/{2:4d}]".format(self.current_epoch,
                                                        self.current_batch_in_epoch,
                                                        self.batches_in_epoch, )

    def _load(self):
        params, states, probs, winners = load_data(self.file_name)

        if self.augument:
            states, probs, winners = augument_data(states=states[:self.n_samples],
                                                   probs=probs[:self.n_samples],
                                                   winners=winners[:self.n_samples],
                                                   board_height=params.item().board_height,
                                                   board_width=params.item().board_width)
        else:
            states = states[:self.n_samples]
            probs = probs[:self.n_samples]
            winners = winners[:self.n_samples]

        self.states = states
        self.probs = probs
        self.winners = winners

    @property
    def size(self):
        return 1 * self.n_samples


    def _get_data(self, ids):
        return self.states[ids], self.probs[ids], self.winners[ids]

    def next_batch(self, batch_size=None):
        """
        Get next batch
        :param batch_size:
        :return:
        """

        if batch_size is None:
            batch_size = self.default_bs

        # first batch
        if self.current_epoch == 0 and self.index_in_epoch == 0:
            self._reset()

        # last batch in epoch:
        if self.index_in_epoch + batch_size > self.size:
            self.current_epoch += 1
            n_rest_sampels = self.size - self.index_in_epoch
            rest_indices = self.indices[-n_rest_sampels:]

            self._reset()
            n_new_samples = batch_size - n_rest_sampels
            new_indices = self.indices[:n_new_samples]
            self.index_in_epoch = n_new_samples

            rest_states, rest_probs, rest_winners = self._get_data(rest_indices)
            new_states, new_probs, new_winners = self._get_data(new_indices)

            if n_rest_sampels == 0:
                x = (new_states, new_probs, new_winners)
            else:
                states = np.concatenate([rest_states, new_states], axis=0)
                probs = np.concatenate([rest_probs, new_probs], axis=0)
                winners = np.concatenate([rest_winners, new_winners], axis=0)
                x = (states, probs, winners)

        else:
            start = self.index_in_epoch
            stop = self.index_in_epoch + batch_size
            batch_indices = self.indices[start:stop]
            x = self._get_data(batch_indices)
            self.index_in_epoch += batch_size

        return x

    def _reset(self):
        """
        Reset after an epoch
        :return:
        """
        self.index_in_epoch = 0
        self.indices = np.arange(self.size)

        if self.shuffle:
            self.rng.shuffle(self.indices)



def augument_data(states, probs, winners, board_height, board_width):
    """augment the data set by rotation and flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    # state rotation
    states = np.concatenate([np.rot90(states, k=i, axes=(1,2)) for i in range(4)], axis=0)
    # state horizontal flip
    states = np.concatenate([states, np.flip(states, axis=2)], axis=0)

    # reshape
    reshaped_probs = probs.reshape(-1, board_height, board_width)
    # probs rotation
    probs = np.concatenate([np.rot90(reshaped_probs, k=i, axes=(1,2)) for i in range(4)], axis=0)
    # probs horizontal flip
    probs = np.concatenate([probs, np.flip(probs, axis=2)], axis=0)
    # reshape the probs back
    probs = np.reshape(probs, (-1, board_height*board_width))

    winners = np.tile(winners, 8)

    return states, probs, winners



def get_test_x():
    x = np.array([[[0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.]],

                  [[0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.]],

                  [[0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 1., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.]],

                  [[0., 0., 0., 1.],
                   [0., 1., 1., 1.],
                   [1., 0., 0., 1.],
                   [1., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.]],

                  [[0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.]],

                  [[0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., 1.]]])

    x = x.reshape(1, 6, 6, 4)

    # state rotation
    x = np.concatenate([np.rot90(x, k=i, axes=(1, 2)) for i in range(4)], axis=0)
    # state horizontal flip
    x = np.concatenate([x, np.flip(x, axis=2)], axis=0)

    # net = PolicyValueNet(board_width=6,
    #                      board_height=6,
    #                      model_file=os.path.join(results_dir, 'zero_17_4_15:36', 'policy_1500.model'))

    return x

def d_kl(p, q):
    return np.mean(np.sum(p * (np.log(p) - np.log(q)), axis=1))