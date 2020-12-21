"""
MiniMax Player with AlphaBeta pruning
"""
from players.AbstractPlayer import AbstractPlayer
import SearchAlgos
import players.MinimaxPlayer as minimax


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time, penalty_score) # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.minimax_player = minimax.Player(game_time, penalty_score)
        self.minimax_player.initiate = alphabeta_initiate

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        self.minimax_player.set_game_params(board)
    

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        print("ALPHABETA TURN:")
        #print(players_score)
        return self.minimax_player.make_move(time_limit, players_score)


    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        self.minimax_player.set_rival_move(pos)


    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        self.minimax_player.update_fruits(fruits_on_board_dict)


    ########## helper functions in class ##########
    #TODO: add here helper functions in class, if needed


    ########## helper functions for AlphaBeta algorithm ##########
    #TODO: add here the utility, succ, and perform_move functions used in AlphaBeta algorithm


def alphabeta_initiate(start_time, time_limit):
    return SearchAlgos.AlphaBeta(minimax.minimax_utility, minimax.minimax_succ, None, start_time, time_limit, minimax.minimax_heuristic)