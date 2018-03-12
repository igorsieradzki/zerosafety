
import numpy as np

from ._dumb import playout_winner, print_board_info

def node_value(node, parent):
    if node[2] > 0:
        return (
            node[1] / node[2]
            + 0.5 * np.sqrt(np.log(parent[2]) / node[2])
        )
    else:
        # Explore new moves
        return 100.0 + np.random.rand() * 100.0

def add_playout(node, *, base_policy):
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
        winner = playout_winner(child[0], base_policy=base_policy)

        # Update child statistics
        if winner * child[0].next_player < 0:
            child[1] += 1
        child[2] += 1

    else:
        # Recursively expand the tree
        winner = add_playout(
            max(node[3], key=lambda c: node_value(c, node)),
            base_policy=base_policy
        )

    if winner * node[0].next_player < 0:
        node[1] += 1
    node[2] += 1

    return winner

def mcts_policy(game, *, base_policy,
        n_playouts=1600, debug=False):
    assert game.winner is None

    tree = [game, 0, 0, None] # [game state, wins, playouts, children]

    for _ in range(n_playouts):
        add_playout(tree, base_policy=base_policy)

    if debug:
        stats = [""] * game.board.size
        for child in tree[3]:
            if child[2] > 0:
                stats[child[0].last_move] = "%d%%/%d" % (
                    100 * child[1] / child[2],
                    child[2]
                )
        print_board_info(game, stats)

    # Return move with the most playouts
    return max(tree[3], key=lambda c: c[2])[0].last_move
