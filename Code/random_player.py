#%% Import

import sys

sys.path.append('E:\\Projects\\01_Patchwork\\Code')

from patchwork import patchwork   
from draw import draw_game
from draw import play_game
import random

#%% Class code

class random_player: 
    def __init__(self, name = "random"):
        self.name = name
        
    def choose_move(self, game_state):
        available_moves = patchwork.available_moves(game_state)
        return(random.choice(available_moves))
    
#%% Sample game

player1 = random_player("p1")
#player2 = random_player("p2")

play_game(player1)

#game = patchwork(player1, player2)
#draw_game(game)
#play_game(game)
#game.game_states[1].available_elements[0].to_xml()
#game.game_states[2].available_elements[1].to_xml()

#game_async = patchwork(player1, player2, synchronous = False)
#while game_async.is_end_game() == False:
#    game_async.make_move()
#len(game_async.game_states)

#play_game(game_async)