
def playout_winner(game, *, base_policy):
    while game.winner is None:
        game = game.move(base_policy(game))

    return game.winner

def dumb_score(game, *, base_policy, n_playouts=100):
    if game.winner is not None:
        return game.winner

    score = 0
    for _ in range(n_playouts):
        score += playout_winner(game, base_policy=base_policy)

    return score / n_playouts

def dumb_policy(game, *, base_policy):
    import numpy as np

    scores = [
        dumb_score(game.move(i), base_policy=base_policy)
            * game.next_player
        if is_available else -1000.0
        for i, is_available in enumerate(game.available_moves)
    ]

    return np.argmax(scores)
