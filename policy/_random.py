
def random_policy(game):
    import numpy as np

    assert game.winner is None

    available = np.array(game.available_moves, dtype=np.float32)
    available /= available.sum()

    return np.random.choice(available.size, p=available)

