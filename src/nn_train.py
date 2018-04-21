

import os
from src.policy_value_net_tensorflow import PolicyValueNet
from src import data_dir, logger, results_dir
from time import time
import numpy as np
from src.nn_utils import Dataset, get_test_x, d_kl
from src.mcts_alphaZero import MCTSPlayer
from src.mcts_pure import MCTSPlayer as MCTS_Pure
from src.game import Board, Game
from tqdm import tqdm, trange
import tensorflow as tf





def train_nn(data_filename,
             board_height=6,
             board_width=6,
             n_in_row=4,
             batch_size=32,
             epochs=15,
             learning_rate=5e-3,
             check_freq=200):

    train = True

    dataset = Dataset(file_name=data_filename, default_bs=batch_size, n_samples=2000, augument=True)


    if train:
        teacher = PolicyValueNet(board_width=board_width,
                                 board_height=board_height,
                                 model_file=os.path.join(results_dir, 'zero_17_4_15:36', 'policy_1500.model'))

        test_x = get_test_x()

        teacher_probs, teacher_value = teacher.policy_value(test_x)
        # teacher_move = np.random.choice(np.arange(36), p=teacher_probs)

        tf.reset_default_graph()

        student = PolicyValueNet(board_width=board_width, board_height=board_height)

        save_dir = os.path.join(results_dir, "student")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        student = PolicyValueNet(board_width=board_width,
                                 board_height=board_height,
                                 #model_file=os.path.join(results_dir, 'zero_17_4_15:36', 'policy_1500.model'))
                                 model_file=os.path.join(results_dir, "student", 'current_policy.model'))

    board = Board(width=board_width,
                       height=board_height,
                       n_in_row=n_in_row)
    game = Game(board)

    pure_mcts_playout_num = 1000
    c_puct = 5
    n_playouts = 400
    n_games = 10
    log_freq = 50

    counter = 0
    correct_counter = 0
    start_time = time()

    if train:

        while dataset.current_epoch < epochs:

            states, probs, winners = dataset.next_batch()

            if dataset.current_epoch >= 10:
                learning_rate = 1e-3
            elif dataset.current_epoch >= 30:
                learning_rate = 5e-4

            loss, entropy = student.train_step(states, probs, winners, lr=learning_rate)

            if (counter + 1) % log_freq == 0:
                logger.info("{0:} time: {1:4.3f}, loss: {2:.4f}, entropy: {3:.4f}".format(
                    dataset.get_progress(), time() - start_time, loss, entropy))

            counter += 1

            if (counter + 1) % check_freq == 0:
                student_probs, student_value = student.policy_value(test_x)
                # student_move = np.random.choice(np.arange(36), p=student_probs)



                kl = d_kl(student_probs, teacher_probs)
                mse = np.mean(np.square(student_value - teacher_value))
                prob_mse = np.mean(np.square(student_probs - teacher_probs))

                max_abs = np.max(np.abs(student_probs - teacher_probs))

                logger.info("evaluation: Dkl: {:.4}, MSE: {:.4}, prob_mse: {:.5}, max: {:.3}".format(kl, mse, prob_mse, max_abs))


        student.save_model(os.path.join(save_dir, 'current_policy.model'))

    winners = []

    current_mcts_player = MCTSPlayer(student.policy_value_fn,
                                     c_puct=c_puct,
                                     n_playout=n_playouts,
                                     is_selfplay=False)

    pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=pure_mcts_playout_num)

    with trange(n_games) as t:
        for i in t:
            winner = game.start_play(pure_mcts_player,
                                     current_mcts_player,
                                     start_player=1,
                                     is_shown=0)

            winners.append(winner)

            t.set_postfix(wins=sum(np.array(winners) == 2))

    logger.info("Evaluation: n_playouts:{}, wins: {}, ties:{}".format(pure_mcts_playout_num,
                                                                      sum(np.array(winners) == 2),
                                                                      sum(np.array(winners) == -1)))

if __name__ == '__main__':
    train_nn(data_filename=os.path.join(data_dir, "games.npz"))



