"""
MiniMax Player with AlphaBeta pruning with heavy heuristic
"""
from players.AbstractPlayer import AbstractPlayer
import numpy as np
from time import time
import utils
import SearchAlgos

class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time, penalty_score)  # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.lifetime = 0
        self.board = None
        self.pos = None
        self.initial_pos = None

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        self.board = board
        pos = np.where(board == 1)
        # convert pos to tuple of ints
        self.pos = tuple(ax[0] for ax in pos)
        self.initial_pos = self.pos

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """

        def find_player_positions(player_index):
            if player_index == 1:
                return self.pos
            pos = np.where(self.board == player_index)
            # convert pos to tuple of ints
            return tuple(ax[0] for ax in pos)

        start_time = time()
        player_positions = (find_player_positions(1), find_player_positions(2))
        self.board[self.pos] = -1

        curr_state = SearchAlgos.State(self.board.copy(), tuple(players_score), player_positions, 0, self.penalty_score,
                                       (0, 0), self.lifetime, self.initial_pos)

        heavy_player = SearchAlgos.AlphaBeta(minimax_utility, minimax_succ, None, start_time, time_limit, minimax_heuristic)
        depth = 2
        value, direction = heavy_player.search(curr_state, depth, 1)

        i = self.pos[0] + direction[0]
        j = self.pos[1] + direction[1]
        self.pos = (i, j)
        self.board[self.pos] = 1

        self.lifetime += 1
        return direction

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        old_pos = np.where(self.board == 2)
        self.board[tuple(ax[0] for ax in old_pos)] = -1
        self.board[pos] = 2
        self.lifetime += 1

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        for pos, fruit in fruits_on_board_dict.items():
            self.board[pos] = fruit

    ########## helper functions in class ##########
    # TODO: add here helper functions in class, if needed

########## helper functions for MiniMax algorithm ##########
# TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm

def other_player_stuck(board, pos):
    for d in utils.get_directions():
        i = pos[0] + d[0]
        j = pos[1] + d[1]
        if 0 <= i < len(board) and 0 <= j < len(board[0]) and (board[i][j] not in [-1, 1, 2]):
            return False
    return True

def minimax_utility(state):
    player_scores = list(state.players_score)
    player_scores[state.curr_player] -= state.penalty
    if other_player_stuck(state.board, state.player_positions[1 - state.curr_player]):
        player_scores[1 - state.curr_player] -= state.penalty
    if player_scores[0] == player_scores[1]:
        return 0.5
    else:
        return int(player_scores[0] > player_scores[1])

def minimax_heuristic(state):
    delta_score = 0.4 * heuristic_score_delta(state)
    distance_from_enemy = 0.3 * heuristic_distance_from_enemy(state)
    distance_from_initial_pos = 0.2 * heuristic_distance_from_initial_pos(state)
    available_steps = 0.1 * heuristic_num_steps(state.board, state.player_positions[0])
    return delta_score + distance_from_enemy + distance_from_initial_pos + available_steps

def heuristic_score_delta(state):
    delta_score = abs(state.players_score[0] - state.players_score[1])
    if delta_score == 0:
        return 0.5  # exactly 0.5, Tie
    score = (0.24 * state.penalty) / delta_score if delta_score >= 0.5 * state.penalty else 0.5 - (
                0.04 / state.penalty) * delta_score
    if state.players_score[0] > state.players_score[1]:
        return 1 - score  # good for us, always bigger than 0.5
    else:
        return score  # bad for us, always lower than 0.5

def heuristic_distance_from_initial_pos(state):
    self_pos = state.player_positions[0]
    initial_pos = state.initial_pos
    return 1 - (1 / (abs(initial_pos[0] - self_pos[0]) + abs(initial_pos[1] - self_pos[1])))

def heuristic_distance_from_enemy(state):
    self_pos = state.player_positions[0]
    rival_pos = state.player_positions[1]
    return 1 / (abs(rival_pos[0] - self_pos[0]) + abs(rival_pos[1] - self_pos[1]))

def heuristic_num_steps(board, pos):
    num_steps_available = 0
    for d in utils.get_directions():
        i = pos[0] + d[0]
        j = pos[1] + d[1]

        # check legal move
        if 0 <= i < len(board) and 0 <= j < len(board[0]) and (board[i][j] not in [-1, 1, 2]):
            num_steps_available += 1
    if num_steps_available == 0:
        return 0
    return 1 / num_steps_available

def minimax_succ(state):
    pos = state.player_positions[state.curr_player]
    succ_states = []
    board = state.board.copy()
    board[pos] = -1
    min_fruit_time = min(len(board[0]), len(board))
    if state.lifetime >= 2 * min_fruit_time:
        board = np.where(board >= 3, 0, board)
    for d in utils.get_directions():
        i = pos[0] + d[0]
        j = pos[1] + d[1]

        if 0 <= i < len(board) and 0 <= j < len(board[0]) and (
                board[i][j] not in [-1, 1, 2]):  # then move is legal

            new_pos = (i, j)
            fruit_score = board[new_pos]
            players_score = list(state.players_score)
            players_score[state.curr_player] += fruit_score

            player_positions = list(state.player_positions)
            player_positions[state.curr_player] = new_pos

            old_board_value = board[new_pos]
            board[new_pos] = (state.curr_player + 1)
            yield SearchAlgos.State(board.copy(), tuple(players_score), tuple(player_positions),
                                    1 - state.curr_player,
                                    state.penalty, d, state.lifetime + 1, state.initial_pos)

            # reset the board: positions + scores
            board[new_pos] = old_board_value
            players_score[state.curr_player] -= fruit_score
