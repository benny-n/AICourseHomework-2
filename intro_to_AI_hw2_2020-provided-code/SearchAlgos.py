"""Search Algos: MiniMax, AlphaBeta
"""
from dataclasses import dataclass

from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
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
    # def __init__(self, board, players_score, player_positions, curr_player, penalty_score):
    #     self.board = board
    #     self.players_score = players_score
    #     self.player_positions = player_positions
    #     self.curr_player = curr_player
    #     self.penalty = penalty_score
    board: np.array
    players_score: tuple
    player_positions: tuple
    curr_player: int
    penalty: int
    direction: tuple


def is_goal(children):
    return len(children) == 0


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
        #global counter
        #counter += 1
        #print("counter:" + str(counter))
        # if time() - self.start_time > self.time_limit:
        #     return -1, state.direction

        if depth == 0:
            self.developed_whole_tree = False
            return self.utility(state), state.direction #TODO: change this later

        children = self.succ(state)

        if self.is_goal(children):
            scores = list(state.players_score)
            scores[state.curr_player] = state.players_score[state.curr_player] - state.penalty
            state = State(state.board.copy(), tuple(scores), state.player_positions, state.curr_player, state.penalty, state.direction)
            return self.utility(state), state.direction

        if maximizing_player:
            curr_max = float("-inf")
            best_direction = state.direction
            for child in children:
                value, direction = self.search(child, depth - 1, 1 - maximizing_player)
                if value > curr_max:
                    #print("update max")
                    curr_max, best_direction = value, child.direction
                    #print("now curr max is:" + str(curr_max) + " and the direction TO RETURN is:" + str(child.direction))
            return curr_max, best_direction
        else:
            curr_min = float("inf")
            for child in children:
                value, none_direction = self.search(child, depth - 1, 1 - maximizing_player)
                if value < curr_min:
                    curr_min, none_direction = value, none_direction
            return curr_min, None



class AlphaBeta(SearchAlgos):

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        #TODO: erase the following line and implement this function.
        raise NotImplementedError
