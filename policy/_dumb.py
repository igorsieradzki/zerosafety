
def playout_winner(game, *, base_policy):
    while game.winner is None:
        game = game.move(base_policy(game))

    return game.winner

def dumb_score(game, *, base_policy, n_playouts):
    if game.winner is not None:
        return game.winner

    score = 0
    for _ in range(n_playouts):
        score += playout_winner(game, base_policy=base_policy)

    return score / n_playouts

def print_board_info(game, data):
    width = len(game.board[0])
    print()
    for y in range(len(game.board)):
        for x in range(width):
            print("%9s" % str(data[y * width + x])[:9], end=" ")
        print("\n")

def dumb_policy(game, *, base_policy,
        n_playouts=5000, debug=False):
    import numpy as np

    assert game.winner is None

    scores = [
        dumb_score(
            game.move(i),
            base_policy=base_policy,
            n_playouts=n_playouts // len(game.available_moves)
        ) * game.next_player
        if is_available else -1000.0
        for i, is_available in enumerate(game.available_moves)
    ]

    if debug:
        print_board_info(game, scores)

    return np.argmax(scores)
