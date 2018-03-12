#!/usr/bin/env python3

import numpy as np

from gomoku import gomoku
from policy import dumb_policy, random_policy, mcts_policy

players = {
    "Random": random_policy,
    "Dumb": lambda game: dumb_policy(game, base_policy=random_policy),
    "MCTS": lambda game: mcts_policy(game, base_policy=random_policy),
}

def game_result(player1, player2):
    game = gomoku(board_size=5)

    while game.winner is None:
        game = game.move(player1(game))
        print(game)

        player1, player2 = player2, player1

    return game.winner

player1 = "Random" ### Random / Dumb / MCTS
player2 = "Dumb" ### Random / Dumb / MCTS
print("X:", player1)
print("O:", player2)

result = game_result(players[player1], players[player2])
print("Winner:  ", {-1: player2, 0: "(draw)", 1: player1}[result])
