
import os
import numpy as np

from . import Player

def node_value(node, parent):
    if node[2] > 0:
        return (
            node[1] / node[2]
            + 0.5 * np.sqrt(np.log(parent[2]) / node[2])
        )
    else:
        # Try new moves in random order
        return 100.0 + np.random.rand() * 100.0

def playout_winner(game):
    while game.winner is None:
        available = np.array(game.available_moves, dtype=np.float32)
        available /= available.sum()
        game = game.move(np.random.choice(available.size, p=available))

    return game.winner

def add_playout(node):
    if node[0].winner is not None:
        winner = node[0].winner

    elif node[3] is None:
        # Expand this node
        node[3] = [
            [node[0].move(i), 0, 0, None]
            for i, is_available in enumerate(node[0].available_moves)
            if is_available
        ]

        # Single playout from random child
        child = node[3][np.random.choice(len(node[3]))]
        winner = playout_winner(child[0])

        # Update child statistics
        if winner * child[0].next_player < 0:
            child[1] += 1
        child[2] += 1

    else:
        # Recursively expand the tree
        winner = add_playout(
            max(node[3], key=lambda c: node_value(c, node))
        )

    # Update this node's statistics
    if winner * node[0].next_player < 0:
        node[1] += 1
    node[2] += 1

    return winner

def print_board_info(game, data):
    width = len(game.board[0])
    print()
    for y in range(len(game.board)):
        for x in range(width):
            print("%9s" % str(data[y * width + x])[:9], end=" ")
        print("\n")

class RandomMCTS(Player):
    def __init__(self, *, n_playouts=1600):
        debug = "DEBUG" in os.environ
        game = None

        def load_game(new_game):
            nonlocal game
            assert new_game.winner is None

            game = new_game

        def add_move(move_id):
            nonlocal game
            assert game.winner is None

            game = game.move(move_id)

        def make_move(): # -> move_id
            nonlocal game
            assert game.winner is None

            # Each node is: [game state, wins, playouts, children]
            tree = [game, 0, 0, None]

            for _ in range(n_playouts):
                add_playout(tree)

            if debug:
                stats = [""] * game.board.size
                for child in tree[3]:
                    if child[2] > 0:
                        stats[child[0].last_move] = "%d%%/%d" % (
                            100 * child[1] / child[2],
                            child[2]
                        )
                print_board_info(game, stats)

            # Pick the move with the most playouts
            move_id = max(tree[3], key=lambda c: c[2])[0].last_move
            game = game.move(move_id)

            return move_id

        self.load_game = load_game
        self.add_move = add_move
        self.make_move = make_move
