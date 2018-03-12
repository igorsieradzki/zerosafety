#!/usr/bin/env python3

from gomoku import gomoku
from policy import dumb_policy, random_policy

game = gomoku()

while game.winner is None:
    game = game.move(dumb_policy(game, base_policy=random_policy))
    print(game)
