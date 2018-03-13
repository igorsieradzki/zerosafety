
import numpy as np

# How many repeated elements around this position in 1d array?
def streak_length(array, around):
    assert array[around] != 0
    a = around - 1
    while a >= 0 and array[a] == array[around]:
        a -= 1
    b = around + 1
    while b < len(array) and array[b] == array[around]:
        b += 1
    return b - a - 1

# Get top-left to bottom-right diagonal from a 2d array
def diagonal_slice(array, y, x):
    idx = np.arange(max(0, x-y), len(array)-max(0, y-x))
    return array[idx + (y-x), idx], x-idx[0]

def gomoku(win_size=4, board_size=6):
    class Node(object):
        def __init__(self, board, next_player, *,
                winner=None, last_move=None):
            # Make sure this ndarray stays read-only
            board.setflags(write=False)
            board = board[:]

            def move(n):
                # Validate move
                n = int(n)
                assert n >= 0 and n < board.size
                y, x = divmod(n, board_size)

                # Mark new move
                new_board = np.array(board)
                assert new_board[y, x] == 0
                new_board[y, x] = next_player

                # Check win condition
                if (
                    streak_length(new_board[y, :], x) == win_size
                    or streak_length(new_board[:, x], y) == win_size
                    or streak_length(*diagonal_slice(new_board,
                        y, x)) == win_size
                    or streak_length(*diagonal_slice(new_board[::-1,:],
                        board_size - (y+1), x)) == win_size
                ):
                    return Node(new_board, -next_player,
                        winner=next_player, last_move=n)

                # Check for draws (no free space)
                if (new_board == 0).astype(np.int32).sum() == 0:
                    return Node(new_board, -next_player,
                        winner=0, last_move=n)

                return Node(new_board, -next_player, last_move=n)

            self.game_id = "gomoku_%d_%d" % (win_size, board_size)
            self.board = board
            self.next_player = next_player
            self.winner = winner
            self.last_move = last_move

            if winner is None:
                self.move = move
                self.available_moves = (board == 0).reshape(-1)
            else:
                # The game is finished
                self.move = None
                self.available_moves = (board == 1000).reshape(-1)

        def __str__(self):
            board_marks = {-1: "O", 0: " ", 1: "X"}

            s = "\n,-" + "+-" * len(self.board[0]) + ",\n"
            for i, row in enumerate(self.board):
                s += "| "
                for node in row:
                    s += board_marks[node] + " "
                s += "|"
                if i == 0:
                    s += (" Next player: "
                        + board_marks[self.next_player])
                if i == 1 and self.winner is not None:
                    if self.winner == 0:
                        s += " Game result: draw"
                    else:
                        s += (" Game result: won by "
                            + board_marks[self.winner])
                s += "\n"
            s += "`-" + "+-" * len(self.board[0]) + "`\n"
            return s

        __repr__ = __str__

    return Node(np.zeros((board_size, board_size), dtype=np.int32), 1)
