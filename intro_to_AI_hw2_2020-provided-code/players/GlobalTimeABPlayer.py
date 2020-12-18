"""
MiniMax Player with AlphaBeta pruning and global time
"""
from players.AbstractPlayer import AbstractPlayer
import numpy as np
#TODO: you can import more modules, if needed
import SearchAlgos
import utils



class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time, penalty_score) # keep the inheritance of the parent's (AbstractPlayer) __init__()
        #TODO: initialize more fields, if needed, and the AlphaBeta algorithm from SearchAlgos.py
        self.lifetime = 0
        self.board = None
        self.pos = None
        self.A = None
        self.B = None
        self.C = None
        self.game_time = game_time
        self.fruit_disappear_turn = None

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
        self.fruit_disappear_turn = min(len(board), len(board[0]))
        self.calculate_coordinates_for_turns_function()
    

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

        player_positions = (find_player_positions(1), find_player_positions(2))
        self.board[self.pos] = -1

        curr_state = SearchAlgos.State(self.board.copy(), tuple(players_score), player_positions, 0, self.penalty_score,
                                       (0, 0), self.lifetime)

        global_alphabeta = SearchAlgos.AlphaBeta(
            minimax_utility, minimax_succ, None, 0, self.calculate_turn_time_limit(self.lifetime), minimax_heuristic)
        depth = 1
        value, direction = global_alphabeta.search(curr_state, depth, 1)

        while not global_alphabeta.developed_whole_tree:
            try:
                global_alphabeta.developed_whole_tree = True
                # curr_value, curr_direction = minimax.search(curr_state, depth, 1)
                value, direction = global_alphabeta.search(curr_state, depth, 1)
                depth += 1
            except SearchAlgos.Interrupted:
                break
        # print("minimax value: " + str(value))
        print("depth: " + str(depth))
        i = self.pos[0] + direction[0]
        j = self.pos[1] + direction[1]
        self.pos = (i, j)
        self.board[self.pos] = 1

        self.lifetime += 1
        # print("number of nodes:" + str(counter))
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
    #TODO: add here helper functions in class, if needed
    def calculate_coordinates_for_turns_function(self):
        allocated_time_after_fruits_disappear = 0.5 * self.game_time
        allocated_time_before_fruits_disappear = 0.5 * self.game_time
        max_turns_possible = len(np.where(self.board != -1)[0]) - 2
        self.C = (max_turns_possible, 0)
        max_turn_time = allocated_time_after_fruits_disappear / (max_turns_possible - self.fruit_disappear_turn)
        self.B = (self.fruit_disappear_turn, max_turn_time)
        initial_turn_time = (2 * allocated_time_before_fruits_disappear / self.fruit_disappear_turn) - max_turn_time
        self.A = (0, initial_turn_time)

    def calculate_turn_time_limit(self, turn):
        if turn <= self.fruit_disappear_turn:
            m = (self.B[1] - self.A[1])/(self.B[0] - self.A[0])
            n = self.A[1]
            return m * turn + n
        else:
            m = (self.C[1] - self.B[1]) / (self.C[0] - self.B[0])
            n = -m * self.C[0]
            return m * turn + n
    ########## helper functions for AlphaBeta algorithm ##########
    #TODO: add here the utility, succ, and perform_move functions used in AlphaBeta algorithm


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
    delta_score = 1 * heuristic_score_delta(state)
    # distance_from_enemy = 0 * heuristic_distance_from_goal(state.board, state.player_positions[0], lambda x: x == 2)
    # available_steps = 0 * heuristic_num_steps(state.board, state.player_positions[0])
    # distance_from_fruit = 0 * heuristic_distance_from_goal(state.board, state.player_positions[0], lambda x: x >= 3)
    return delta_score #+ distance_from_fruit + distance_from_enemy + available_steps


def heuristic_score_delta(state):
    # print("player score:" + str(state.players_score))
    delta_score = abs(state.players_score[0] - state.players_score[1])
    if delta_score == 0:
        return 0.5                      # exactly 0.5, Tie
    score = (0.24 * state.penalty) / delta_score if delta_score >= 0.5 * state.penalty else 0.5 - (0.04 / state.penalty) * delta_score
    if state.players_score[0] > state.players_score[1]:
        return 1 - score                # good for us, always bigger than 0.5
    else:
        return score                    # bad for us, always lower than 0.5


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
    if state.lifetime >= min_fruit_time:
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
            lifetime = state.lifetime if state.curr_player == 0 else state.lifetime + 1
            yield SearchAlgos.State(board.copy(), tuple(players_score), tuple(player_positions), 1 - state.curr_player,
                                  state.penalty, d, lifetime)

            # reset the board: positions + scores
            board[new_pos] = old_board_value
            players_score[state.curr_player] -= fruit_score





