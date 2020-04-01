import random
import copy
import math
import string
import pygame    
import pygame.gfxdraw
import pandas as pd
import itertools
import time
import xgboost as xgb
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

START_MONEY = 5
MAX_TIME = 53
THRESHOLDS_TIME = [4.5, 10.5, 16.5, 22.5, 28.5, 34.5, 40.5, 46.5, 52.5]


class stats:
    def __init__(self, money = 0, time = 0, buttons = 0, space = 0):
        self.money = money
        self.time = time
        self.buttons = buttons
        self.space = space

        
    def add(self, x):
        self.money = self.money + x.money
        self.time = self.time + x.time
        self.buttons = self.buttons + x.buttons
        self.space = self.space + x.space
        
    def print(self):
        print("Money", self.money, "| Time", self.time, "| Buttons", self.buttons, "| Space", self.space)
        
    def to_xml(self):
        result = "<money>{}</money><time>{}</time><buttons>{}</buttons><space>{}</space>".format(self.money, self.time, self.buttons, self.space)
        return(result)
        
        

class element(stats):
    def __init__(self, id, money, time, buttons, space, shape = ""):
        stats.__init__(self, money = money, time = time, buttons = buttons, space = space)
        self.id = id
        self.shape = shape
        
    def to_xml(self):
        result = "<element><id>{}</id>{}</element>".format(self.id, stats.to_xml(self))
        return(result)
        
class player_stats(stats):
    def __init__(self, id, money = START_MONEY, time = 0, buttons = 0, space = 0):
        stats.__init__(self, money = money, time = time, buttons = buttons, space = space)
        self.id = id
        self.points = self.calculate_points()
        
    def make_move(self, e):
        time_start = self.time
        self.money = self.money - e.money
        self.time = self.time + e.time
        self.buttons = self.buttons + e.buttons
        self.space = self.space + e.space
        
        for t in THRESHOLDS_TIME:
            if ((time_start < t) & (self.time > t)):
                self.money = self.money + self.buttons
        
        self.calculate_points()
        
    def can_make_move(self, e):
        if ((e.money <= self.money) & (e.time + self.time <= MAX_TIME)):
            return(True)
        else:
            return(False)
            
    def calculate_points(self):
        self.points = -162 + self.money + 2 * self.space
            
    def to_xml(self):
        result = "<player><id>{}</id><points>{}</points>{}</player>".format(self.id, self.points, stats.to_xml(self))
        return(result)            

class game:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.current_state = game_state()
        self.game_progress = []        
        self.add_game_progress()
                
        while(self.current_state.is_end_game() == False):
            if (self.current_state.current_player() == 1):
                move = self.player1.choose_move(self.current_state)
            else:
                move = self.player2.choose_move(self.current_state)
            self.current_state.make_move(move)
            self.add_game_progress()
        
    def add_game_progress(self):
        self.game_progress.append(copy.deepcopy(self.current_state)) 
        
    
    def winner(self):
        if self.game_progress[-1].player1_stats.points > self.game_progress[-1].player2_stats.points:
            return(1)
        if self.game_progress[-1].player1_stats.points < self.game_progress[-1].player2_stats.points:
            return(2)
        return(0)
        
                                                         

class game_state:
    def __init__(self, 
                 player1_stats = player_stats(id = 1), 
                 player2_stats = player_stats(id = 2), 
                 available_elements = None,
                 last_move = element(id = 0, money = 0, time = 0, buttons = 0, space = 0), 
                 player_last_move = random.randint(1, 2)):
        
        self.player1_stats = copy.deepcopy(player1_stats)
        self.player2_stats = copy.deepcopy(player2_stats)
        if available_elements is None:
            self.available_elements = self.initialize_elements()
        else:
            self.available_elements = available_elements
        self.last_move = last_move
        self.player_last_move = player_last_move
        
    def create_money_element(self):
        if (self.player1_stats.time <= self.player2_stats.time):
            money = self.player2_stats.time - self.player1_stats.time + 1
            if (self.player2_stats.time == MAX_TIME):
                money = money - 1
        else:
            money = self.player1_stats.time - self.player2_stats.time + 1
            if (self.player1_stats.time == MAX_TIME):
                money = money - 1
        e = element(id = 0, money = -money, time = money, buttons = 0, space = 0)
        return(e)                                             
        
    def available_moves(self):
        if (self.current_player() == 1):
            s = self.player1_stats
        else:
            s = self.player2_stats
        max_e = min(len(self.available_elements), 3)
        
        result = [self.create_money_element()]
        if (max_e == 0):
            return(result)
        for i in range(max_e):
            if s.can_make_move(self.available_elements[i]):
                result.append(self.available_elements[i])
        return(result)
    
    def is_end_game(self):
        if ((self.player1_stats.time >= MAX_TIME) & (self.player2_stats.time >= MAX_TIME)):
            return(True)
        else:
            return(False)
        
    def current_player(self):
        if (self.player1_stats.time < self.player2_stats.time):
            return(1)
        if (self.player1_stats.time > self.player2_stats.time):
            return(2)
        return(self.player_last_move)
        
    def make_move(self, e):
        self.last_move = e
        if (self.current_player() == 1):
            self.player1_stats.make_move(e)
            self.player_last_move = 1
        else:
            self.player2_stats.make_move(e)
            self.player_last_move = 2
        if(e.id == 0):
            return
        else:
            x = self.available_elements.pop(0)
            if (x.id != e.id):
                self.available_elements.append(x)
                x = self.available_elements.pop(0)
                if (x.id != e.id):
                    self.available_elements.append(x)
                    x = self.available_elements.pop(0)
                    if (x.id != e.id):
                        raise Exception('Move not valid')
        
    def initialize_elements(self):
        ae = []
        ae.append(element(id = 2, money = 1, time = 3, buttons = 0, space = 3))
        ae.append(element(id = 3, money = 2, time = 2, buttons = 0, space = 3))
        ae.append(element(id = 4, money = 3, time = 1, buttons = 0, space = 3))
        ae.append(element(id = 5, money = 2, time = 2, buttons = 0, space = 4))
        ae.append(element(id = 6, money = 3, time = 2, buttons = 1, space = 4))
        ae.append(element(id = 7, money = 3, time = 3, buttons = 1, space = 4))
        ae.append(element(id = 8, money = 4, time = 2, buttons = 1, space = 4))
        ae.append(element(id = 9, money = 4, time = 6, buttons = 2, space = 4))
        ae.append(element(id = 10, money = 6, time = 5, buttons = 2, space = 4))
        ae.append(element(id = 11, money = 7, time = 6, buttons = 3, space = 4))
        ae.append(element(id = 12, money = 1, time = 2, buttons = 0, space = 5))
        ae.append(element(id = 13, money = 2, time = 2, buttons = 0, space = 5))
        ae.append(element(id = 14, money = 2, time = 3, buttons = 1, space = 5))
        ae.append(element(id = 15, money = 3, time = 4, buttons = 1, space = 5))
        ae.append(element(id = 16, money = 5, time = 4, buttons = 2, space = 5))
        ae.append(element(id = 17, money = 5, time = 5, buttons = 2, space = 5))
        ae.append(element(id = 18, money = 7, time = 1, buttons = 1, space = 5))
        ae.append(element(id = 19, money = 10, time = 3, buttons = 2, space = 5))
        ae.append(element(id = 20, money = 10, time = 4, buttons = 3, space = 5))
        ae.append(element(id = 21, money = 0, time = 3, buttons = 1, space = 6))
        ae.append(element(id = 22, money = 1, time = 2, buttons = 0, space = 6))
        ae.append(element(id = 23, money = 1, time = 5, buttons = 1, space = 6))
        ae.append(element(id = 24, money = 2, time = 1, buttons = 0, space = 6))
        ae.append(element(id = 25, money = 3, time = 6, buttons = 2, space = 6))
        ae.append(element(id = 26, money = 4, time = 2, buttons = 0, space = 6))
        ae.append(element(id = 27, money = 7, time = 2, buttons = 2, space = 6))
        ae.append(element(id = 28, money = 7, time = 4, buttons = 2, space = 6))
        ae.append(element(id = 29, money = 8, time = 6, buttons = 3, space = 6))
        ae.append(element(id = 30, money = 10, time = 5, buttons = 3, space = 6))
        ae.append(element(id = 31, money = 1, time = 4, buttons = 1, space = 7))
        ae.append(element(id = 32, money = 2, time = 3, buttons = 0, space = 7))
        ae.append(element(id = 33, money = 5, time = 3, buttons = 1, space = 8))
        random.shuffle(ae)
        ae.append(element(id = 1, money = 2, time = 1, buttons = 0, space = 2))
        return(ae)
        
    def to_xml(self):
        p1_xml = self.player1_stats.to_xml()
        p2_xml = self.player2_stats.to_xml()
        av_e_xml = "<available_elements><number>{}</number>{}</available_elements>".format(len(self.available_elements),''.join([x.to_xml() for x in self.available_elements]))
        curr_p_xml = "<current_player>{}</current_player>".format(self.current_player())
        av_m_xml = "<available_moves>{}</available_moves>".format(''.join([x.to_xml() for x in self.available_moves()]))
        last_move_xml = "<last_move><player>{}</player>{}</last_move>".format(self.player_last_move, self.last_move.to_xml())
        final_xml = "<game_state>{}{}{}{}{}{}</game_state>".format(p1_xml, p2_xml, av_e_xml, curr_p_xml, av_m_xml, last_move_xml)
        return(final_xml)
        
class random_player: 
    def __init__(self, name = "random"):
        self.name = name
        
    def choose_move(self, state):
        return(random.choice(state.available_moves()))
        
class gbm1_player:
    def __init__(self, name = "gbm1"):
        self.model = None
        self.name = name
        
    def choose_move(self, state):
        moves = state.available_moves()
        if self.model is None:
            return(random.choice(moves))
        else: 
            possible_states = [state]
            for am in moves:
                possible_states.append(copy.deepcopy(state))
                possible_states[-1].make_move(am)
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
            n_estimators = 1000,
            n_jobs = 7)
        
        eval_set = [(X_train, y_train), (X_test, y_test)]
        xg_reg.fit(X_train, y_train, early_stopping_rounds = 10, eval_set=eval_set, verbose = False)
        
        xg_reg = xgb.XGBRegressor(
            objective ='reg:squarederror', 
            colsample_bytree = 1, 
            learning_rate = 0.1,
            max_depth = 5, 
            alpha = 10, 
            n_estimators = xg_reg.best_iteration,
            n_jobs = 7)
        
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
 

def competition(players, n):
    points = [0 for x in players]
    names = [x.name for x in players]
    games = [0 for x in players]
    for k in range(n):
        for i in range(len(players) - 1):
            for j in range(i+1, len(players)):
                g = game(players[i], players[j])
                if g.winner() == 1:
                    points[i] = points[i] + 1
                if g.winner() == 2:
                    points[j] = points[j] + 1
                if g.winner() == 0:
                    points[i] = points[i] + 0.5
                    points[j] = points[j] + 0.5
                games[i] = games[i] + 1
                games[j] = games[j] + 1
    result = {'names': names, 'points': points, 'games': games}
    df = pd.DataFrame(result)
    #df = df.sort_values(['points'], ascending=[0])
    return(df)

def competition2(players, n):
    iterations = len(players) * len(players) - 1
    result = []
    for i in range(len(players) - 1):
        for j in range(i+1, len(players)):
            winners = Parallel(n_jobs = num_cores)(delayed(game_results)(x) for x in [[players[i], players[j]]] * n)
            #winners = [game_results([players[i], players[j]])]
            win1 = len(list(filter(lambda x: x == 1, winners)))
            win2 = len(list(filter(lambda x: x == 2, winners)))
            tie = len(list(filter(lambda x: x == 0, winners)))
            result.append(pd.DataFrame({'name_1': [players[i].name], 'name_2': [players[j].name], 'win_1': [win1], 'win_2': [win2], 'tie': [tie]}))
            print(result[-1])
    return(pd.concat(result))
    
def self_match(p):
    return(game(p, p).game_progress)


def game_results(p):
    return(game(p[0], p[1]).winner())




game_tables = []
gbm_players = [gbm1_player("random")]

n = 5000
num_cores = 7

for i in range(len(gbm_players) - 1, 20):
    print(i)
    start_time = time.time()
    inputs = [gbm_players[-1]] * n
    game_states = Parallel(n_jobs = num_cores)(delayed(self_match)(i) for i in inputs)
    # game_states = [game(gbm_players[-1], gbm_players[-1]).game_progress for x in range(n)]
    game_tables = game_tables + [gbm_players[-1].get_table_game(x) for x in game_states]
    gbm_players.append(gbm1_player("gbm1_{}".format(len(game_tables))))
    gbm_players[-1].learn(game_states = game_tables)
    print(results)
    print(round((time.time() - start_time) / 60, 1))
    print("-------------------------------")

start_time = time.time()
result = competition(gbm_players[1:], 1000)
print((time.time() - start_time))
print(result)

joblib.dump(gbm_players[], "E:\\Projects\\CODES\\4_Patchwork\\MODELS\\gbm1_45000.z")


xxx = joblib.load("E:\\Projects\\CODES\\4_Patchwork\\MODELS\\gbm1_45000.z")

gbm_player_new = gbm1_player('gbm1_45000')
gbm_player_new.model = gbm_players[9].model

g = game(gbm_players[-1], gbm_players[-1])

new_gbm_players = []
for x in gbm_players:
	new_player = gbm1_player(x.name)
	new_player.model = x.model
	new_gbm_players.append(new_player)
    
    




def get_shape(n):
    x = ["" for x in range(34)]
    x[0] = ""
    x[1] = "XX"
    x[2] = "XX\nX "
    x[3] = "XXX"
    x[4] = "XX\nX "
    x[5] = "XXX\n X "
    x[6] = "XX \n XX"
    x[7] = "XXXX"
    x[8] = "XXX\nX  "
    x[9] = "  X\nXXX"
    x[10] = "XX\nXX"
    x[11]= "XX \n XX"
    x[12] = "X X\nXXX"
    x[13] = "XXX\nXX "
    x[14] = " XXX\nXX  "
    x[15] = "  X \nXXXX"
    x[16] = " X \nXXX\n X "
    x[17] = "XXX\n X \n X "
    x[18] = "XXXXX"
    x[19] = "X   \nXXXX"
    x[20] = "  X\n XX\nXX "
    x[21] = "  X \nXXXX\n  X "
    x[22] = "   X\nXXXX\nX   "
    x[23] = "XXXX\nX  X"
    x[24] = "  X \nXXXX\n X  "
    x[25] = " X \nXXX\nX X"
    x[26] = " XXX\nXXX "
    x[27] = "X   \nXXXX\nX   "
    x[28] = "XXXX\n XX "
    x[29] = " XX\n XX\nXX "
    x[30] = "XXXX\nXX  "
    x[31] = "  X  \nXXXXX\n  X  "
    x[32] = "XXX\n X \nXXX"
    x[33] = " XX \nXXXX\n XX "
    e = x[n].split("\n")
    return(e)
    

    
COLOR_BACKGROUND = (200, 200, 200)
COLOR_SQUARE = (0, 0, 0)
COLOR_MONEY = (255, 0, 0)
COLOR_BUTTONS = (0, 200, 0)
COLOR_TIME = (0, 0, 255)
COLOR_SPACE = (255, 25, 255)
SIZE_X = 1600
SIZE_Y = 1000
SIZE_SQUARE = 16



def draw_shape(win, e, x, y):
    if e.id > 0:
        shape = get_shape(e.id)
        for i in range(len(shape)):
            for j in range(len(shape[i])):
                if shape[i][j] == "X":
                   pygame.draw.rect(win, COLOR_SQUARE , (x + i * SIZE_SQUARE, y + j * SIZE_SQUARE, SIZE_SQUARE, SIZE_SQUARE), 2) 
               
def draw_player(win, p_stats, p_num, x, y):
    
    textsurface = myfont.render('Player {}'.format(p_num), False, COLOR_SQUARE)
    win.blit(textsurface, (x, y))     
    
    textsurface = myfont.render('Money {}'.format(p_stats.money), False, COLOR_MONEY)
    win.blit(textsurface, (x, y + 1.5 * SIZE_SQUARE))
        
    textsurface = myfont.render('Time {}'.format(p_stats.time), False, COLOR_TIME)
    win.blit(textsurface, (x, y + 3 * SIZE_SQUARE))
        
    textsurface = myfont.render('Buttons {}'.format(p_stats.buttons), False, COLOR_BUTTONS)
    win.blit(textsurface, (x, y + 4.5 * SIZE_SQUARE))
    
    textsurface = myfont.render('Space {}'.format(p_stats.space), False, COLOR_SPACE)
    win.blit(textsurface, (x, y + 6 * SIZE_SQUARE))
               
def draw_multiple_shapes(win, elements, x, y, shapes_in_row = 6):
    for i in range(len(elements)):
        pos_x = i % shapes_in_row
        pos_y = i // shapes_in_row
        pygame.draw.rect(win, COLOR_SQUARE , (x + pos_x * 6 * SIZE_SQUARE - SIZE_SQUARE, y + pos_y * 7 * SIZE_SQUARE - SIZE_SQUARE, SIZE_SQUARE * 6, SIZE_SQUARE * 7), 2) 
        draw_shape(win, elements[i], x + pos_x * 6 * SIZE_SQUARE, y + pos_y * 7 * SIZE_SQUARE)
        
        textsurface = myfont.render(str(elements[i].money), False, COLOR_MONEY)
        win.blit(textsurface, (x + pos_x * 6 * SIZE_SQUARE - SIZE_SQUARE * 0.75, y + pos_y * 7 * SIZE_SQUARE - SIZE_SQUARE))    
        
        textsurface = myfont.render(str(elements[i].time), False, COLOR_TIME)
        win.blit(textsurface, (x + pos_x * 6 * SIZE_SQUARE + SIZE_SQUARE * 4.25, y + pos_y * 7 * SIZE_SQUARE - SIZE_SQUARE)) 
        
        textsurface = myfont.render(str(elements[i].buttons), False, COLOR_BUTTONS)
        win.blit(textsurface, (x + pos_x * 6 * SIZE_SQUARE - SIZE_SQUARE * 0.75, y + pos_y * 7 * SIZE_SQUARE + 4.75 * SIZE_SQUARE)) 
        
        textsurface = myfont.render(str(elements[i].space), False, COLOR_SPACE)
        win.blit(textsurface, (x + pos_x * 6 * SIZE_SQUARE + SIZE_SQUARE * 4.25, y + pos_y * 7 * SIZE_SQUARE + 4.75 * SIZE_SQUARE)) 
        
def draw_game_state(win, state, move_num, x, y):
    draw_multiple_shapes(win, state.available_elements, x + 50, y + 200)
    draw_player(win, state.player1_stats, 1,  x + 50, y + 50)
    draw_player(win, state.player2_stats, 2,  x + 150, y + 50)
    draw_multiple_shapes(win, [state.last_move], x + 300, y + 70)
    textsurface = myfont.render('Last player {}'.format(state.player_last_move), False, COLOR_SQUARE)
    win.blit(textsurface, (x + 280, y + 30))
    textsurface = myfont.render('Move number {}'.format(move_num), False, COLOR_SQUARE)
    win.blit(textsurface, (x + 200, y + 10))

pygame.init()
myfont = pygame.font.SysFont('Comic Sans MS', SIZE_SQUARE) 

def draw_game(g):
    
               
    win = pygame.display.set_mode((SIZE_X, SIZE_Y), pygame.RESIZABLE)    
    
    run = True
    
    current_move = 0
    while run:
        pygame.time.delay(100)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                            
            if event.type == pygame.VIDEORESIZE:
                win = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_move = current_move - 1
                if event.key == pygame.K_RIGHT:
                    current_move = current_move + 1
        
            current_move = max(current_move, 0)
            current_move = min(current_move, len(g.game_progress) - 1)
            next_move = min(current_move + 1, len(g.game_progress) - 1)
            pygame.draw.rect(win, COLOR_BACKGROUND , (0, 0, SIZE_X, SIZE_Y))  
            draw_game_state(win, g.game_progress[current_move], current_move, 0, 0)
            draw_game_state(win, g.game_progress[next_move], next_move, 800, 0)
        
        pygame.display.update()
                
    pygame.quit() 

draw_game(g)
