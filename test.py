#!/usr/bin/env python3

from gomoku import gomoku

def random_move(game):
    import numpy as np

    available = np.array(game.available_moves, dtype=np.float32)
    available /= available.sum()

    return np.random.choice(available.size, p=available)

def playout(start_game, player1, player2):
    game = [start_game()]

    # Make moves until the game is finished
    while game[-1].winner is None:
        game.append(game[-1].move(player1(game[-1])))
        player1, player2 = player2, player1

    # Propagate game result to all intermediate positions
    for state in game:
        state.winner = game[-1].winner

    return game[1:]

p = playout(gomoku, random_move, random_move)
print(p)
