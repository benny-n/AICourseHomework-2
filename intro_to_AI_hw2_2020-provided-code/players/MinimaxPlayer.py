"""
MiniMax Player
"""
from players.AbstractPlayer import AbstractPlayer
import numpy as np
from time import time
import random
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
        #print("initial state of board")
        #print(self.board)
        #print(players_score)

        def find_player_positions(player_index):
            if player_index == 1:
                return self.pos
            pos = np.where(self.board == player_index)
            # convert pos to tuple of ints
            return tuple(ax[0] for ax in pos)
        start_time = time()
        player_positions = (find_player_positions(1), find_player_positions(2))
        #print(self.pos)
        self.board[self.pos] = -1

        curr_state = SearchAlgos.State(self.board.copy(), tuple(players_score), player_positions, 0, self.penalty_score, (0, 0))
        minimax = SearchAlgos.MiniMax(minimax_utility, minimax_succ, None, start_time, time_limit, None) #TODO: change last argument!
        depth = 1
        value = 0
        legal_directions = get_legal_directions(self.board, self.pos)
        print(legal_directions)
        best_direction = legal_directions[random.randint(0, len(legal_directions) - 1)]
        # next_state = None
        # value, direction = minimax.search(curr_state, depth, True)
        # print("actual direction chosen:" + str(direction))
        while time() - start_time < time_limit - 0.01:
            #print(time() - start_time)
            curr_value, curr_direction = minimax.search(curr_state, depth, 1)
            #print(curr_value)
            if value < curr_value != -1:
                value, best_direction = curr_value, curr_direction
                #print(best_direction)
            depth += 1
            print(depth)

        i = self.pos[0] + best_direction[0]
        j = self.pos[1] + best_direction[1]
        self.pos = (i, j)
        #print(self.pos)
        self.board[self.pos] = 1
        #print("board from make_move:" + str(self.board))
        #print(player_positions)
        #print(next_state.player_positions)
        #print(type(np.subtract(next_state.player_positions[0], player_positions[0])))
        # print(player_positions[0])
        # # i = next_state.player_positions[0][0] - self.pos[0]
        # # j = next_state.player_positions[0][1] - self.pos[1]
        # # print(i, j)
        return best_direction


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


def get_legal_directions(board, pos):
    legal_directions = []
    for d in utils.get_directions():
        i = pos[0] + d[0]
        j = pos[1] + d[1]

        if 0 <= i < len(board) and 0 <= j < len(board[0]) and (
                board[i][j] not in [-1, 1, 2]):  # then move is legal
            legal_directions.append(d)
    return legal_directions

    ########## helper functions for MiniMax algorithm ##########
    #TODO: add here the utility, succ, and perform_move functions used in MiniMax algorithm


def minimax_utility(state):
    #(maximizing_player)
    normalizing_factor = max(state.players_score[0], state.players_score[1])
    if normalizing_factor == 0:
        #print("tie for direction:" + str(state.direction))
        return 0.5                                      # exactly 0.5, Tie
    score = abs(state.players_score[0] - state.players_score[1]) * 0.5
    if state.players_score[0] >= state.players_score[1]:
        #print("good state:" + str(min(score/normalizing_factor + 0.5, 1)) + " for direction:" + str(state.direction))
        return min(score/normalizing_factor + 0.5, 1)   # good for us, always bigger than 0.5
    else:
        #print("bad state:" + str(max(0.5 - score/normalizing_factor, 0)) + " for direction:" + str(state.direction))
        return max(0.5 - score/normalizing_factor, 0)   # bad for us, always lower than 0.5


def minimax_succ(state):
    pos = state.player_positions[state.curr_player]
    succ_states = []
    board = state.board.copy()
    #print(pos)
    board[pos] = -1
    #print("new turn")
    for d in utils.get_directions():
        i = pos[0] + d[0]
        j = pos[1] + d[1]

        if 0 <= i < len(board) and 0 <= j < len(board[0]) and (
                board[i][j] not in [-1, 1, 2]):  # then move is legal
            #print(board)
            new_pos = (i, j)
            #print(new_pos)
            fruit_score = board[new_pos]
            players_score = list(state.players_score)
            players_score[state.curr_player] += fruit_score

            player_positions = list(state.player_positions)
            player_positions[state.curr_player] = new_pos

            old_board_value = board[new_pos]
            board[new_pos] = (state.curr_player + 1)
            succ_states.append(SearchAlgos.State(board.copy(), tuple(players_score), tuple(player_positions), 1-state.curr_player, state.penalty, d))

            # reset the board: positions + scores
            board[new_pos] = old_board_value
            #players_score[state.curr_player] -= fruit_score

    return succ_states
