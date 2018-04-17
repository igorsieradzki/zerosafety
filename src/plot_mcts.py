from src import results_dir
from src.game import Board, Game
from src.policy_value_net_tensorflow import PolicyValueNet # Tensorflow
from src.mcts_pure import MCTSPlayer as MCTS_Pure

import os

import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

import matplotlib.pyplot as plt
import matplotlib.cbook
import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)



board_width, board_height, n_in_row = 6, 6, 4
c_puct, n_playout = 5., 100

files = os.listdir(results_dir)
files.sort()
most_recent = files[-1]
init_model = os.path.join(results_dir, most_recent, 'best_policy.model')

board = Board(width=board_width, height=board_height, n_in_row=n_in_row)
game = Game(board)
policy_value_net = PolicyValueNet(board_width, board_height, model_file=init_model)
p1 = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=c_puct, n_playout=n_playout, is_selfplay=1)

board.init_board()
move, probs = p1.get_action(board, temp=1, return_prob=True)

G = p1.mcts.create_nx_graph()

print(p1.mcts.get_size(), p1.mcts.get_height(), G.size())

pos=graphviz_layout(G, prog='dot')
plt.figure(figsize=(30,30)) 
nx.draw(G, pos, with_labels=True, arrows=False, hold=True, node_size=1000)
plt.savefig('mcts')
