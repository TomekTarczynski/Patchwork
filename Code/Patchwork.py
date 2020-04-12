#%% Import

import random
import copy

#%% CONSTANTS

START_MONEY = 5
MAX_TIME = 53
THRESHOLDS_TIME = [4.5, 10.5, 16.5, 22.5, 28.5, 34.5, 40.5, 46.5, 52.5]


#%% class stats

class stats:
    def __init__(self, money = 0, time = 0, buttons = 0, space = 0):
        self.money = money
        self.time = time
        self.buttons = buttons
        self.space = space
        
    def print(self):
        print("Money", self.money, "| Time", self.time, "| Buttons", self.buttons, "| Space", self.space)
        
    def to_xml(self):
        result = "<money>{}</money><time>{}</time><buttons>{}</buttons><space>{}</space>".format(self.money, self.time, self.buttons, self.space)
        return(result)
        
#%% class elements        
        
class element(stats):
    def __init__(self, id, money, time, buttons, space):
        stats.__init__(self, money = money, time = time, buttons = buttons, space = space)
        self.id = id
        
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
        
    def to_xml(self):
        result = "<element><id>{}</id>{}</element>".format(self.id, stats.to_xml(self))
        return(result)
    
#%% class player_stats    
        
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

#%% class patchwork

class patchwork:
    def __init__(self, player1, player2, synchronous = True):
        self.player1 = player1
        self.player2 = player2
        self.synchronous = synchronous
        self.game_states = [patchwork_state()]        
                
        if self.synchronous:
            while(self.is_end_game() == False):
                self.make_move()
                    
    def save_game_state(self, state):
        self.game_states.append(copy.deepcopy(state)) 
    
    # Possible inputs:
    #   o self - move for current player is choosen and then make_move(self, move) is called
    #   o self, move - move is made for current game_state
    #   o game_state, move - move is made for given game_state
    # Returns:
    #   o game_state - state after the move is made
    def make_move(*args):
        if __debug__:
            if len(args) == 1:
                if args[0].__class__ not in [patchwork]:
                    raise Exception("If make_move have one argument then it must be of type patchwork")
            elif len(args) > 2:
                raise Exception("Make_move must have exactly one or two arguments")
            elif args[0].__class__ not in [patchwork, patchwork_state]:
                raise Exception("First parameter of make move must be of class either patchwork or patchwork_state")
            elif args[1].__class__ not in [element]:
                raise Exception("Second parameter of make move must be of class element")
        
        if len(args) == 1:
            if (args[0].current_player() == 1):
                move = args[0].player1.choose_move(args[0].current_state())
            else:
                move = args[0].player2.choose_move(args[0].current_state())
            return(patchwork.make_move(args[0], move))
    
        if args[0].__class__ == patchwork:
            args[0].save_game_state(args[0].current_state())
            game_state = args[0].current_state()
        else:
            game_state = args[0]
        e = args[1]
        
        game_state.last_move = e
        if (game_state.current_player() == 1):
            game_state.player1_stats.make_move(e)
            game_state.player_last_move = 1
        else:
            game_state.player2_stats.make_move(e)
            game_state.player_last_move = 2
        game_state.available_elements.pop(0) # Money element is removed
        if(e.id > 0):
            x = game_state.available_elements.pop(0)
            if (x.id != e.id):
                game_state.available_elements.append(x)
                x = game_state.available_elements.pop(0)
                if (x.id != e.id):
                    game_state.available_elements.append(x)
                    x = game_state.available_elements.pop(0)
                    if (x.id != e.id):
                        raise Exception('Move not valid')
        game_state.available_elements.insert(0, patchwork.create_money_element(game_state))
        return(game_state)
        
    # -1 : game is still on
    #  0 : draw
    #  1 : player1 won
    #  2 : player2 won
    def winner(*args):
        if len(args) != 1:
            raise Exception("Winner must have exactly one argument")
        elif args[0].__class__ not in [patchwork, patchwork_state]:
            raise Exception("First argument of winner must be of type either patchwork or patchwork_state")
         
        if args[0].__class__ == patchwork:
            game_state = args[0].current_state()
        else:
            game_state = args[0]
            

        if patchwork.is_end_game(game_state) == False:
            return(-1)
        if game_state.player1_stats.points > game_state.player2_stats.points:
            return(1)
        if game_state.player1_stats.points < game_state.player2_stats.points:
            return(2)
        return(0)
    
    def create_money_element(*args):
        if len(args) != 1:
            raise Exception("create_money_element must have exactly one argument")
        elif args[0].__class__ not in [patchwork, patchwork_state]:
            raise Exception("First argument of create_money_element must be of type either patchwork or patchwork_state")       
        
        if args[0].__class__ == patchwork:
            game_state = args[0].current_state()
        else:
            game_state = args[0]        
        
        game_max_time = max(game_state.player1_stats.time, game_state.player2_stats.time)
        game_min_time = min(game_state.player1_stats.time, game_state.player2_stats.time)
        money = game_max_time - game_min_time + 1
        if (game_max_time == MAX_TIME):
            money = money - 1
        e = element(id = 0, money = -money, time = money, buttons = 0, space = 0)
        return(e)         
    
    def available_moves(*args):
        if len(args) != 1:
            raise Exception("available_moves must have exactly one argument")
        elif args[0].__class__ not in [patchwork, patchwork_state]:
            raise Exception("First argument of available_moves must be of type either patchwork or patchwork_state")     
        
        if args[0].__class__ == patchwork:
            game_state = args[0].current_state()
        else:
            game_state = args[0]          
        
        if game_state.current_player() == 1:
            s = game_state.player1_stats
        else:
            s = game_state.player2_stats
        max_e = min(len(game_state.available_elements), 4)
        
        result = []
        if (max_e == 0):
            return(result)
        for i in range(max_e):
            if s.can_make_move(game_state.available_elements[i]):
                result.append(game_state.available_elements[i])
        return(result)
    
    def is_end_game(*args):
        if len(args) != 1:
            raise Exception("is_end_game must have exactly one argument")
        elif args[0].__class__ not in [patchwork, patchwork_state]:
            raise Exception("First argument of is_end_game must be of type either patchwork or patchwork_state")       
        
        if args[0].__class__ == patchwork:
            game_state = args[0].current_state()
        else:
            game_state = args[0]          
        
        if ((game_state.player1_stats.time >= MAX_TIME) & (game_state.player2_stats.time >= MAX_TIME)):
            return(True)
        else:
            return(False)
    
    def current_state(self):
        return(self.game_states[-1])
    
    def current_player(self):
        return(self.current_state().current_player())
    
    def initialize_elements():
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
        
#%% class game_state                                                        

class patchwork_state:
    def __init__(self, 
                 player1_stats = player_stats(id = 1), 
                 player2_stats = player_stats(id = 2), 
                 available_elements = None,
                 last_move = element(id = 0, money = 0, time = 0, buttons = 0, space = 0), 
                 player_last_move = random.randint(1, 2)):
        
        self.player1_stats = copy.deepcopy(player1_stats)
        self.player2_stats = copy.deepcopy(player2_stats)
        if available_elements is None:
            self.available_elements = patchwork.initialize_elements()
            self.available_elements.insert(0, element(id = 0, money = -1, time = 1, buttons = 0, space = 0))
        else:
            self.available_elements = available_elements
        self.last_move = last_move
        self.player_last_move = player_last_move
                  
    def current_player(self):
        if (self.player1_stats.time < self.player2_stats.time):
            return(1)
        if (self.player1_stats.time > self.player2_stats.time):
            return(2)
        return(self.player_last_move)
                        
    def to_xml(self):
        p1_xml = self.player1_stats.to_xml()
        p2_xml = self.player2_stats.to_xml()
        av_e_xml = "<available_elements><number>{}</number>{}</available_elements>".format(len(self.available_elements),''.join([x.to_xml() for x in self.available_elements]))
        curr_p_xml = "<current_player>{}</current_player>".format(self.current_player())
        av_m_xml = "<available_moves>{}</available_moves>".format(''.join([x.to_xml() for x in patchwork.available_moves(self)]))
        last_move_xml = "<last_move><player>{}</player>{}</last_move>".format(self.player_last_move, self.last_move.to_xml())
        final_xml = "<game_state>{}{}{}{}{}{}</game_state>".format(p1_xml, p2_xml, av_e_xml, curr_p_xml, av_m_xml, last_move_xml)
        return(final_xml)
