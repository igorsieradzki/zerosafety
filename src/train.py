# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from datetime import datetime
from time import time
import os
from src import logger, results_dir

import random
import numpy as np
from collections import defaultdict, deque
from src.game import Board, Game
from src.mcts_pure import MCTSPlayer as MCTS_Pure
from src.mcts_alphaZero import MCTSPlayer
from src.policy_value_net_tensorflow import PolicyValueNet
from src.nn_utils import lr_schedule


class TrainPipeline():
    def __init__(self,
                 init_model=None,
                 board_width=6,
                 board_height=6,
                 n_in_row=4,
                 learning_rate=2e-3,
                 n_playouts=400,
                 batch_size=512,
                 train_steps=5,
                 check_freq=50,
                 n_iters=1500):

        # params of the board and the game
        self.board_width = board_width
        self.board_height = board_height
        self.n_in_row = n_in_row

        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # training params
        self.learning_rate = learning_rate
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.initial_temp = 1.0  # the temperature param

        self.n_playouts = n_playouts  # num of simulations for each move

        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = batch_size  # mini-batch size for training

        self.states_buffer = deque(maxlen=self.buffer_size)
        self.probs_buffer = deque(maxlen=self.buffer_size)
        self.winners_buffer = deque(maxlen=self.buffer_size)

        self.play_batch_size = 1
        self.train_steps = train_steps  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = check_freq
        self.game_batch_num = n_iters
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000

        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playouts,
                                      is_selfplay=1)

    def get_equi_data(self, states, probs, winners):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        # state rotation
        states = np.concatenate([np.rot90(states, k=i, axes=(2,3)) for i in range(1,5)], axis=0)
        # state horizontal flip
        states = np.concatenate([states, np.flip(states, axis=3)], axis=0)

        # reshape with flipping due to dumb original move encoding, see game.py:33
        reshaped_probs = np.flip(probs.reshape(-1, self.board_height, self.board_width), axis=1)
        # probs rotation
        probs = np.concatenate([np.rot90(reshaped_probs, k=i, axes=(1,2)) for i in range(1,5)], axis=0)
        # probs horizontal flip
        probs = np.concatenate([probs, np.flip(probs, axis=2)], axis=0)
        # reshape the probs back
        probs = np.reshape(np.flip(probs, axis=1), (-1, self.board_height*self.board_width))

        winners = np.tile(winners, 8)

        return states, probs, winners

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, (states, probs, winners) = self.game.start_self_play(self.mcts_player,
                                                          temp=self.initial_temp)
            self.episode_len = states.shape[0]
            # augment the data

            states, probs, winners = self.get_equi_data(states, probs, winners)
            self.states_buffer.extend(states)
            self.probs_buffer.extend(probs)
            self.winners_buffer.extend(winners)

    def policy_update(self, learning_rate):
        """update the policy-value net"""

        # TODO: maybe change this to single train step with bigger batch size?

        start = time()

        assert len(self.states_buffer) == len(self.probs_buffer)
        assert len(self.states_buffer) == len(self.winners_buffer)

        for _ in range(self.train_steps):

            batch_ids = np.random.choice(len(self.states_buffer), self.batch_size, replace=False)
            state_batch = np.array(self.states_buffer)[batch_ids]
            mcts_probs_batch = np.array(self.probs_buffer)[batch_ids]
            winner_batch = np.array(self.winners_buffer)[batch_ids]

            old_probs, old_v = self.policy_value_net.policy_value(state_batch)

            loss, entropy = self.policy_value_net.train_step(state_batch=state_batch,
                                                             mcts_probs=mcts_probs_batch,
                                                             winner_batch=winner_batch,
                                                             lr=learning_rate)

            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))

        #     if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
        #         break
        #
        # # adaptively adjust the learning rate
        # if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
        #     self.lr_multiplier /= 1.5
        # elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
        #     self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))

        update_time = time()- start

        logger.debug(("kl:{:.3f}, "
                      "loss:{:.3f}, "
                      "entropy:{:.3f}, "
                      "explained_var_old:{:.3f}, "
                      "explained_var_new:{:.3f}, "
                      "time: {:.2f}"
                      ).format(kl,
                               loss,
                               entropy,
                               explained_var_old,
                               explained_var_new,
                               update_time))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playouts)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        logger.info("Evaluation: n_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):

        # create save dir for this run
        now = datetime.now()
        save_dir = os.path.join(results_dir, "{day}_{m}_{h}:{min}".format(day=now.day,
                                                                          m=now.month,
                                                                          h=now.hour,
                                                                          min=now.minute))

        os.makedirs(save_dir)
        mean_iter_time = 0

        schedule = lr_schedule()

        # run the training pipeline
        try:
            for i in range(self.game_batch_num):
                start = time()
                self.collect_selfplay_data(self.play_batch_size)
                mcts_time = time() - start
                logger.info("iter: {}, episode_len:{}, mean time: {:.2f}".format(i + 1, self.episode_len, mean_iter_time))

                if len(self.states_buffer) > self.batch_size:
                    loss, entropy = self.policy_update(learning_rate=schedule(i))

                mean_iter_time += ((time() - start) - mean_iter_time) / (i + 1)

                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    logger.info("current self-play batch: {}, evaluating...".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model(os.path.join(save_dir, 'current_policy.model'))
                    if win_ratio > self.best_win_ratio:
                        logger.info("Found new best policy, saving")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model(os.path.join(save_dir, 'best_policy.model'))
                        if (self.best_win_ratio == 1.0 and
                                    self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0

        except KeyboardInterrupt:
            logger.warning('Got keyboard interrupt, saving and quiting')
            try:
                self.policy_value_net.save_model(os.path.join(save_dir, 'current_policy.model'))
            except:
                logger.error("[!] Error while saving policy net on keyboard interrupt, quiting")




