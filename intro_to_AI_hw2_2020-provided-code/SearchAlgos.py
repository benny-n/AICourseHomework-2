"""Search Algos: MiniMax, AlphaBeta
"""
from dataclasses import dataclass

from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT, get_directions
from time import time
import numpy as np
#TODO: you can import more modules, if needed

class SearchAlgos:
    def __init__(self, utility, succ, perform_move, goal):
        """The constructor for all the search algos.
        You can code these functions as you like to,
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move
        self.is_goal = goal

    def search(self, state, depth, maximizing_player):
        pass


@dataclass(frozen=True)
class State:

    board: np.array
    players_score: tuple
    player_positions: tuple
    curr_player: int
    penalty: int
    direction: tuple
    lifetime: int
    initial_pos: tuple = None


def is_goal(state):
    pos = state.player_positions[state.curr_player]
    for d in get_directions():
        i = pos[0] + d[0]
        j = pos[1] + d[1]

        if 0 <= i < len(state.board) and 0 <= j < len(state.board[0]) and (
            state.board[i][j] not in [-1, 1, 2]):  # then move is legal
            return False
    return True


class Interrupted(Exception):
    pass


class MiniMax(SearchAlgos):

    developed_whole_tree = True

    def __init__(self, utility, succ, perform_move, start_time, time_limit, heuristic):
        SearchAlgos.__init__(self, utility, succ, perform_move, is_goal)
        self.start_time = start_time
        self.time_limit = time_limit
        self.heuristic = heuristic

    def search(self, state, depth, maximizing_player) -> (float, State):
        """Start the MiniMax algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """

        if time() - self.start_time > self.time_limit - 0.01:
            raise Interrupted

        if self.is_goal(state):
            return self.utility(state), state.direction

        if depth == 0:
            self.developed_whole_tree = False
            return self.heuristic(state), state.direction

        if maximizing_player:
            curr_max = float("-inf")
            best_direction = None
            for child in self.succ(state):
                value, direction = self.search(child, depth - 1, 1 - maximizing_player)
                if value > curr_max:
                    curr_max, best_direction = value, child.direction
            return curr_max, best_direction
        else:
            curr_min = float("inf")
            for child in self.succ(state):
                value, none_direction = self.search(child, depth - 1, 1 - maximizing_player)
                if value < curr_min:
                    curr_min, none_direction = value, none_direction
            return curr_min, None


class AlphaBeta(SearchAlgos):

    developed_whole_tree = True

    def __init__(self, utility, succ, perform_move, start_time, time_limit, heuristic):
        SearchAlgos.__init__(self, utility, succ, perform_move, is_goal)
        self.start_time = start_time
        self.time_limit = time_limit
        self.heuristic = heuristic

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """

        if time() - self.start_time > self.time_limit - 0.01:
            raise Interrupted

        if self.is_goal(state):
            return self.utility(state), state.direction

        if depth == 0:
            self.developed_whole_tree = False
            return self.heuristic(state), state.direction

        if maximizing_player:
            curr_max = float("-inf")
            best_direction = None
            for child in self.succ(state):
                value, direction = self.search(child, depth - 1, 1 - maximizing_player, alpha, beta)
                if value > curr_max:
                    curr_max, best_direction = value, child.direction
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return float('inf'), None
            return curr_max, best_direction
        else:
            curr_min = float("inf")
            for child in self.succ(state):
                value, none_direction = self.search(child, depth - 1, 1 - maximizing_player, alpha, beta)
                if value < curr_min:
                    curr_min, none_direction = value, none_direction
                beta = min(curr_min, beta)
                if curr_min <= alpha:
                    return float('-inf'), None
            return curr_min, None
