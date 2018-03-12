#!/usr/bin/env python3

from gomoku import gomoku
from policy import dumb_policy, random_policy, mcts_policy

game = gomoku(board_size=5)

player1 = lambda game: dumb_policy(game, base_policy=random_policy, debug=True)
player2 = lambda game: mcts_policy(game, base_policy=random_policy, debug=True)

while game.winner is None:
    game = game.move(player1(game))
    print(game)

    player1, player2 = player2, player1
