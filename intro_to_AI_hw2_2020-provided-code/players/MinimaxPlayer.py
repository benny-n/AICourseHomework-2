"""
MiniMax Player
"""
from players.AbstractPlayer import AbstractPlayer
import numpy as np
from time import time
import utils
import SearchAlgos


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time, penalty_score) # keep the inheritance of the parent's (AbstractPlayer) __init__()
        #TODO: initialize more fields, if needed, and the Minimax algorithm from SearchAlgos.py


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
        player_positions = [find_player_positions(1), find_player_positions(2)]
        #state(board, player_scores, player_positions, 1 - state.curr_player, state.penalty)
        curr_state = state(self.board, players_score, player_positions, 1, self.penalty_score)
        minimax = SearchAlgos.MiniMax(minimax_utility, minimax_succ, None, start_time, time_limit, None) #TODO: change last argument!
        depth = 1
        value = 0
        next_state = None
        while time() - start_time < time_limit:
            curr_value, curr_state = minimax.search(curr_state, depth, 1)
            if value < curr_value:
                value, next_state = curr_value, curr_state
            depth += 1

        self.board[player_positions[0]] = -1
        self.board[next_state.player_positions[0]] = 1

        self.pos = next_state.player_positions[0]

        return next_state.player_positions[0] - player_positions[0]


    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        self.board[pos] = 2


    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        for pos, fruit in fruits_on_board_dict:
            self.board[pos] = fruit

    ########## helper functions in class ##########
    #TODO: add here helper functions in class, if needed


    ########## helper functions for MiniMax algorithm ##########
    #TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm

def minimax_utility(state, maximizing_player):
    score = state.players_score[maximizing_player] > state.players_score[1 - maximizing_player]
    if maximizing_player == 1:
        return int(score), state
    else:
        return (1 - int(score)), state

def minimax_succ(state):
    pos = state.player_positions[state.curr_player]
    succ_states = []
    board = state.board
    board[pos] = -1
    for d in utils.get_directions():
        i = pos[0] + d[0]
        j = pos[1] + d[1]

        if 0 <= i < len(board) and 0 <= j < len(board[0]) and (
                board[i][j] not in [-1, 1, 2]):  # then move is legal
            new_pos = (i, j)
            fruit_score = board[new_pos]
            player_scores = state.score
            player_scores[state.curr_player] += fruit_score

            player_positions = state.player_positions
            player_positions[state.curr_player] = new_pos

            board[new_pos] = state.curr_player + 1
            succ_states.append(state(board, player_scores, player_positions, 1-state.curr_player, state.penalty))

    if len(succ_states) == 0:
        state.score[state.curr_player] = max(state.score[state.curr_player] - state.penalty, 0)

    return succ_states