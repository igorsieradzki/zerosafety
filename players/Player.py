
class Player(object):
    def load_game(self, new_game):
        raise NotImplementedError()

    def add_move(self, move_id):
        raise NotImplementedError()

    def make_move(self): # -> move_id
        raise NotImplementedError()
