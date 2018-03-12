#!/usr/bin/env python3

import numpy as np

from gomoku import gomoku
from policy import dumb_policy, random_policy, mcts_policy

players = {
    "random": random_policy,
    "dumb": lambda game: dumb_policy(game, base_policy=random_policy),
    "mcts": lambda game: mcts_policy(game, base_policy=random_policy),
}

player1 = players["dumb"] ### random / dumb / mcts
player2 = players["dumb"] ### random / dumb / mcts

game = gomoku()

while game.winner is None:
    game = game.move(player1(game))
    print(game)

    player1, player2 = player2, player1
