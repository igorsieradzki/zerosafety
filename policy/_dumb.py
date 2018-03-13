
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
        n_playouts=1600, debug=False):
    import numpy as np

    n_playouts //= np.array(game.available_moves, dtype=np.int32).sum()
    n_playouts = max(1, int(n_playouts))

    scores = [
        dumb_score(game.move(i), base_policy=base_policy,
            n_playouts=n_playouts) * game.next_player
        if is_available else -1000.0
        for i, is_available in enumerate(game.available_moves)
    ]

    if debug:
        print_board_info(game, ["%d%%/%d" % (50.0*(s+1.0), n_playouts)
            if s > -100.0 else "" for s in scores])

    return np.argmax(scores)
