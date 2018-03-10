#!/usr/bin/env python3

from gomoku import gomoku

def random_move(game):
    import numpy as np

    available = np.array(game.available_moves, dtype=np.float32)
    available /= available.sum()

    return np.random.choice(available.size, p=available)

def dumb_playouts_move(game):
    best_score = None
    best_move = -1

    for i, is_available in enumerate(game.available_moves):
        if is_available:
            state_after = game.move(i)
            if state_after.winner is None:
                score = 0.0
                for _ in range(200):
                    p = playout(state_after, random_move, random_move)
                    score += 0.005 * p[-1].winner * game.next_player
            else:
                score = state_after.winner * game.next_player

            if best_score is None or score > best_score:
                best_score = score
                best_move = i

    return best_move

def playout(game, player1, player2, *, print=lambda *_: None):
    game = [game]

    # Make moves until the game is finished
    while game[-1].winner is None:
        game.append(game[-1].move(player1(game[-1])))
        player1, player2 = player2, player1
        print(game[-1])

    return game[1:]

playout(gomoku(), dumb_playouts_move, dumb_playouts_move, print=print)
