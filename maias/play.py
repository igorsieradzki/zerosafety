#!/usr/bin/env python3

import numpy as np

from gomoku import gomoku
from players import *

player1 = DumbPlayouts() ### RandomMoves / DumbPlayouts / RandomMCTS
player2 = RandomMCTS() ### RandomMoves / DumbPlayouts / RandomMCTS

game = gomoku()
player1.load_game(game)
player2.load_game(game)

while game.winner is None:
    move = player1.make_move()
    player2.add_move(move)
    game = game.move(move)
    print(game)

    player1, player2 = player2, player1
