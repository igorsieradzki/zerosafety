from src import logger, results_dir, data_dir
from src.game import Board, Game
from src.policy_value_net_tensorflow import PolicyValueNet # Tensorflow
from src.mcts_pure import MCTSPlayer as MCTS_Pure
from src.mcts_alphaZero import MCTSPlayer
from datetime import datetime

import argparse
import os

import numpy as np

parser = argparse.ArgumentParser(description='Generate training examples for learning to teach.')

parser.add_argument('--n_examples', '-ne', nargs=1, type=int, required=True, help='A number of training examples to generate')
parser.add_argument('--init_model', '-im', nargs=1, type=str, default='best_policy.model',
                    help='A name of a model to use. If not specified, then the "best_policy.model" is uded')
parser.add_argument('--board_width', '-bw', nargs=1, type=int, default=6, help='A board width')
parser.add_argument('--board_height', '-bh', nargs=1, type=int,  default=6, help='A board height')
parser.add_argument('--n_in_row', '-nr', nargs=1, type=int, default=4, help='A game objective')
parser.add_argument('--c_puct', '-cp', nargs=1, type=float, default=5., help='A c_puct hyperparameter')
parser.add_argument('--n_playout', '-np', nargs=1, type=int, default=400, help='A number of playouts in MCTS')
parser.add_argument('--temperature', '-t', nargs=1, type=float, default=3, help='An initial temperature')


args = parser.parse_args()
if args.init_model == 'best_policy.model':
    files = os.listdir(results_dir)
    files.sort()
    most_recent = files[-1]
    args.init_model = os.path.join(results_dir, most_recent, 'best_policy.model')
    
logger.info(args)
board = Board(width=args.board_width, height=args.board_height, n_in_row=args.n_in_row)
game = Game(board)
policy_value_net = PolicyValueNet(args.board_width, args.board_height, model_file=args.init_model)
mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=args.c_puct, n_playout=args.n_playout, is_selfplay=1)

states_buf, probs_buf, winners_buf = [], [], []
n_examples = args.n_examples[0]
while len(states_buf) < n_examples:
    _, (states, probs, winners) = game.start_self_play(mcts_player, temp=args.temperature)
    states_buf.extend(states)
    probs_buf.extend(probs)
    winners_buf.extend(winners)
    logger.info('generated {}/{} examples'.format(len(states_buf), n_examples))

now = datetime.now()
save_file = os.path.join(data_dir, "{day}_{m}_{h}:{min}".format(day=now.day, m=now.month, h=now.hour, min=now.minute))
np.savez(save_file, **{'args':args, 'states':states_buf[:n_examples], 'probs':probs_buf[:n_examples], 'winners':winners_buf[:n_examples]})
