import sys

sys.path.append('E:\\Projects\\01_Patchwork\\Code')

import patchwork as patch
from draw import draw_game
from draw import play_game
import competition
import random
import time
from joblib import Parallel, delayed
import pandas as pd
import xgboost as xgb
import copy
import numpy as np
import joblib

class gbm1_player:
    def __init__(self, name = "gbm1"):
        self.model = None
        self.name = name
        
    def choose_move(self, state):
        moves = patch.patchwork.available_moves(state)
        if self.model is None:
            return(random.choice(moves))
        else: 
            possible_states = [state]
            for am in moves:
                possible_states.append(copy.deepcopy(state))
                patch.patchwork.make_move(possible_states[-1], am)
            table_game = self.get_table_game(possible_states)
            pred = self.model.predict(table_game.iloc[:, :-1])
            best_move = np.argmax(pred)
            return(moves[best_move])
        
    def learn(self, game_states, train_fraction = 0.8):
        random.shuffle(game_states)
        game_states_full = pd.concat(game_states)
        game_states_train = pd.concat(game_states[:int(n * train_fraction)])
        game_states_test = pd.concat(game_states[int(n * train_fraction):])
        
        game_states_full.reset_index(drop=True)
        game_states_train.reset_index(drop=True)
        game_states_test.reset_index(drop=True)
        
        X_train, y_train = game_states_train.iloc[:,:-1], game_states_train.iloc[:,-1]
        X_test, y_test = game_states_test.iloc[:,:-1], game_states_test.iloc[:,-1]
        X_full, y_full = game_states_full.iloc[:,:-1], game_states_full.iloc[:,-1]
        
        xg_reg = xgb.XGBRegressor(
            objective ='reg:squarederror', 
            colsample_bytree = 1, 
            learning_rate = 0.1,
            max_depth = 5, 
            alpha = 10, 
            n_estimators = 1000)
        
        eval_set = [(X_train, y_train), (X_test, y_test)]
        xg_reg.fit(X_train, y_train, early_stopping_rounds = 10, eval_set=eval_set, verbose = False)
        
        xg_reg = xgb.XGBRegressor(
            objective ='reg:squarederror', 
            colsample_bytree = 1, 
            learning_rate = 0.1,
            max_depth = 5, 
            alpha = 10, 
            n_estimators = xg_reg.best_iteration)
        
        xg_reg.fit(X_full, y_full)
        self.model = xg_reg
        
    def get_table_game(self, game_progress):
        p1_final = game_progress[-1].player1_stats.points
        p2_final = game_progress[-1].player2_stats.points
        which_player = [x.player_last_move for x in game_progress[1:]]
        money = [x.player1_stats.money if x.player_last_move == 1 else x.player2_stats.money for x in game_progress[1:]]
        buttons = [x.player1_stats.buttons if x.player_last_move == 1 else x.player2_stats.buttons for x in game_progress[1:]]
        space = [x.player1_stats.space if x.player_last_move == 1 else x.player2_stats.space for x in game_progress[1:]]
        time = [x.player1_stats.time if x.player_last_move == 1 else x.player2_stats.time for x in game_progress[1:]]
        final =  [p1_final if x.player_last_move == 1 else p2_final for x in game_progress[1:]]       
        data = {'which_player': which_player, 'money': money, 'time': time, 'buttons': buttons, 'space': space, 'final': final}
        
        df = pd.DataFrame(data) 
        return(df)        
    
    
game_tables = []
gbm_players = [gbm1_player("random")]

n = 1000
num_cores = 7

for i in range(len(gbm_players) - 1, 20):
    print(i)
    start_time = time.time()
    inputs = [gbm_players[-1]] * n
    game_states = Parallel(n_jobs = num_cores)(delayed(self_match)(i) for i in inputs)
    game_tables = game_tables + [gbm_players[-1].get_table_game(x) for x in game_states]
    gbm_players.append(gbm1_player("gbm1_{}".format(len(game_tables))))
    gbm_players[-1].learn(game_states = game_tables)
    print(round((time.time() - start_time) / 60, 1))
    print("-------------------------------")

start_time = time.time()
result = competition.competition2(gbm_players[1:], 100)
print((time.time() - start_time))
print(result)

joblib.dump(gbm_players[18], "E:\\Projects\\01_Patchwork\\Models\\gbm1_18000.z")


# mdl = joblib.load("E:\\Projects\\CODES\\4_Patchwork\\MODELS\\gbm1_18000.z")
