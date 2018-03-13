
import numpy as np

from . import Player

class RandomMoves(Player):
    def __init__(self):
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

            available = np.array(game.available_moves, dtype=np.float32)
            available /= available.sum()

            move_id = np.random.choice(available.size, p=available)
            game = game.move(move_id)

            return move_id

        self.load_game = load_game
        self.add_move = add_move
        self.make_move = make_move

