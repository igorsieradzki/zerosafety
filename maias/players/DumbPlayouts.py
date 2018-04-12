
import os
import numpy as np

from . import Player

def playout_winner(game):
    while game.winner is None:
        available = np.array(game.available_moves, dtype=np.float32)
        available /= available.sum()
        game = game.move(np.random.choice(available.size, p=available))

    return game.winner

def dumb_score(game, *, n_playouts):
    if game.winner is not None:
        return game.winner

    score = 0
    for _ in range(n_playouts):
        score += playout_winner(game)

    return score / n_playouts

def print_board_info(game, data):
    width = len(game.board[0])
    print()
    for y in range(len(game.board)):
        for x in range(width):
            print("%9s" % str(data[y * width + x])[:9], end=" ")
        print("\n")

class DumbPlayouts(Player):
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

            n = np.array(game.available_moves, dtype=np.int32).sum()
            n = max(1, int(n_playouts) // n)

            scores = [
                dumb_score(game.move(i), n_playouts=n)
                    * game.next_player
                if is_available else -1000.0
                for i, is_available in enumerate(game.available_moves)
            ]

            if debug:
                print_board_info(
                    game,
                    ["%d%%/%d" % (50.0*(s+1.0), n)
                        if s > -100.0 else "" for s in scores]
                )

            move_id = np.argmax(scores)
            game = game.move(move_id)
            return move_id

        self.load_game = load_game
        self.add_move = add_move
        self.make_move = make_move
