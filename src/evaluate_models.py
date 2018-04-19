from src import logger, results_dir, data_dir
from src.game import Board, Game
from src.mcts_pure import MCTSPlayer as MCTS_Pure
from src.mcts_alphaZero import MCTSPlayer
from src.policy_value_net_tensorflow import PolicyValueNet
from src.nn_utils import lr_schedule
from collections import defaultdict

import argparse
import os

import numpy as np

parser = argparse.ArgumentParser(description='Evaluate two models against each other.')

parser.add_argument('--n_games', '-ng', type=int, default=100, help='A number games in a turnament.')
parser.add_argument('--first_player', '-fp', type=str, default='best_policy.model',
                    help='A path to the first model data file. Default = the most recent best_policy model.')
parser.add_argument('--second_player', '-sp', type=str, default='Pure MCTS',
                    help='A path to the second model data file. Default=pure MCTS player.')
parser.add_argument('--board_width', '-bw', type=int, default=6, help='A board width')
parser.add_argument('--board_height', '-bh', type=int,  default=6, help='A board height')
parser.add_argument('--n_in_row', '-nr', type=int, default=4, help='A game objective')
parser.add_argument('--c_puct_f', '-cpf', type=float, default=5., help='A c_puct hyperparameter for the first player')
parser.add_argument('--c_puct_s', '-cps', type=float, default=5., help='A c_puct hyperparameter for the second player')
parser.add_argument('--n_playout_f', '-npf', type=int, default=400, help='A number of playouts in MCTS for the first player.')
parser.add_argument('--n_playout_s', '-nps', type=int, default=400, help='A number of playouts in MCTS for the second player.')
parser.add_argument('--temperature', '-t', type=float, default=3, help='An initial temperature')


args = parser.parse_args()

    
logger.info(args)

board = Board(width=args.board_width, height=args.board_height, n_in_row=args.n_in_row)
game = Game(board)


if args.first_player == 'best_policy.model':
    files = os.listdir(results_dir)
    files.sort()
    most_recent = files[-1]
    init_model = os.path.join(results_dir, most_recent, 'best_policy.model')
else:
    init_model = os.path.join(results_dir, args.first_player)
    if not os.path.exists(init_model):
        logger.error('the file {} does not exist'.format(init_model))
print(init_model)
policy_value_net = PolicyValueNet(args.board_width, args.board_height, model_file=init_model)
first_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=args.c_puct_f, n_playout=args.n_playout_f)
    
 
if args.second_player == 'Pure MCTS':
    second_player = MCTS_Pure(c_puct=args.c_puct_s, n_playout=args.n_playout_s)
else:
    init_model = os.path.join(results_dir, args.first_player)
    policy_value_net = PolicyValueNet(args.board_width, args.board_height, model_file=init_model)
    second_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=args.c_puct_s, n_playout=args.n_playout_s)
 

win_cnt = defaultdict(int)
for i in range(args.n_games):
    winner = game.start_play(first_player, second_player, start_player=i % 2, is_shown=0)
    win_cnt[winner] += 1
    if (i+1) % 50 == 0:
        ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / (i+1)
        logger.info("Evaluation checkpoint {}/{}: win: {}, lose: {}, tie: {}, ratio: {}".format((i+1)//50, args.n_games//50, 
                                                                                                win_cnt[1], win_cnt[2], win_cnt[-1], ratio))
ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / args.n_games
logger.info("Evaluation: win: {}, lose: {}, tie: {}, ratio: {}".format(win_cnt[1], win_cnt[2], win_cnt[-1], ratio))
