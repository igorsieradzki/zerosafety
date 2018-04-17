# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

from src import logger
import numpy as np
import copy
import networkx as nx


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p, self_play=True):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

        self.self_play = self_play

    def expand(self, moves, probs):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """

        # correct place to add dirichlet noise for exploration, a bit hack'y right now
        explor_noise = np.random.dirichlet(0.43 * np.ones(len(probs)))

        for eps, action, prob in zip(explor_noise, moves, probs):
            if action in self._children:
                raise Exception('Possible second expand on the same node')
            if self.is_root() and self.self_play:
                prob = 0.75 * prob + 0.25 * eps

            self._children[action] = TreeNode(parent=self, prior_p=prob, self_play=self.self_play)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000, self_play=True):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(parent=None, prior_p=1.0, self_play=self_play)
        self._prev_root = None
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.self_play = self_play

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        (moves, probs), leaf_value = self._policy(state)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(moves, probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]

        acts, visits = zip(*act_visits)
        act_probs = softmax((1.0/temp) * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._prev_root = self._root
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(parent=None, prior_p=1.0, self_play=self.self_play)
            self._prev_root = None

    def __str__(self):
        return "MCTS"

    def get_height(self):
        queue = [(self._prev_root, 0)]
        tree_height = 0
        while queue:
            node, height = queue.pop(0)
            if height > tree_height:
                tree_height = height
            for child in node._children.values():
                queue.append((child, height+1))
        return tree_height

    def get_size(self):
        queue = [self._prev_root]
        tree_size = 1
        while queue:
            node = queue.pop(0)
            for child in node._children.values():
                if not child.is_leaf():
                    tree_size += 1
                queue.append(child)
        return tree_size

    def create_nx_graph(self):
        # parent, node are use for mcts tree
        # nxparent, nxnode are used for networkx tree
        G = nx.Graph()
        label = 0
        nxnode = (-1, int(self._prev_root._P*100)/100, label)
        label += 1
        G.add_node(nxnode)
        queue = [(self._prev_root, nxnode)]
        while queue:
            parent, nxparent = queue.pop(0)
            for action, child in parent._children.items():
                if not child.is_leaf() or child._Q == 1.:
                    nxnode = (action, int(child._P*100)/100, label)
                    label += 1
                    G.add_node(nxnode)
                    G.add_edge(nxparent, nxnode)
                    queue.append((child, nxnode))
        return G





class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self,
                 policy_value_function,
                 c_puct=5,
                 n_playout=2000,
                 is_selfplay=False):

        self._is_selfplay = is_selfplay

        self.mcts = MCTS(policy_value_fn=policy_value_function,
                         c_puct=c_puct,
                         n_playout=n_playout,
                         self_play=is_selfplay)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            actions, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(actions)] = probs

            move = np.random.choice(actions, p=probs)

            if self._is_selfplay:
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # reset the root node
                self.mcts.update_with_move(-1)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            logger.warning("[!] Game board is full!")

    def __str__(self):
        return "MCTS {}".format(self.player)
