#%% Import

import pygame
import patchwork
import copy

#%% Constants

COLOR_BACKGROUND = (200, 200, 200)
COLOR_SQUARE = (0, 0, 0)
COLOR_MONEY = (255, 255, 0)
#COLOR_BUTTONS = (0, 200, 0)
COLOR_BUTTONS = (0, 0, 0)
COLOR_TIME = (0, 0, 255)
COLOR_SPACE = (255, 25, 255)
COLOR_NO = (255, 0, 0)
COLOR_YES = (0, 200, 0)

SIZE_BASE = 16
SIZE_CELL_X = SIZE_BASE * 6
SIZE_CELL_Y = SIZE_BASE * 7
CELLS_IN_ROW = 6

SIZE_X = 1600
SIZE_Y = 1200

BORDER_WIDTH_SQARE = max(round(0.1 * SIZE_BASE), 1)
BORDER_WIDTH_CELL = max(round(0.1 * SIZE_BASE), 1)


#%% Functions

def draw_text(win, text, color, pos_x, pos_y):
    textsurface = myfont.render(text, False, color)
    win.blit(textsurface, (pos_x, pos_y))     

def draw_shape(win, e, x, y):
    if e.id > 0:
        shape =  patchwork.element.get_shape(e.id)
        for i in range(len(shape)):
            for j in range(len(shape[i])):
                if shape[i][j] == "X":
                   pygame.draw.rect(win, COLOR_SQUARE , (x + i * SIZE_BASE, y + j * SIZE_BASE, SIZE_BASE, SIZE_BASE), BORDER_WIDTH_SQARE) 
               
def draw_player(win, p_stats, p_num, x, y):
    draw_text(win, 'Player {}'.format(p_num), COLOR_SQUARE, x, y)
    draw_text(win, 'Money {}'.format(p_stats.money), COLOR_MONEY, x, y + 1.5 * SIZE_BASE)
    draw_text(win, 'Time {}'.format(p_stats.time), COLOR_TIME, x, y + 3 * SIZE_BASE)
    draw_text(win, 'Buttons {}'.format(p_stats.buttons), COLOR_BUTTONS, x, y + 4.5 * SIZE_BASE)
    draw_text(win, 'Space {}'.format(p_stats.space), COLOR_SPACE, x, y + 6 * SIZE_BASE)
                       
def draw_multiple_shapes(win, elements, is_available, x, y):
    for i in range(len(elements)):
        pos_x = i % CELLS_IN_ROW
        pos_y = i // CELLS_IN_ROW
        cell_pos_x = x + pos_x * 6 * SIZE_BASE
        cell_pos_y = y + pos_y * 7 * SIZE_BASE
        cell_pos_x_end = x + (pos_x + 1) * 6 * SIZE_BASE
        cell_pos_y_end = y + (pos_y + 1) * 7 * SIZE_BASE
        
        if is_available[i] == True:
            color = COLOR_YES
        elif is_available[i] == False:
            color = COLOR_NO
        elif is_available[i] is None:
            color = COLOR_BACKGROUND
        pygame.draw.rect(win, color , (cell_pos_x, cell_pos_y, SIZE_CELL_X, SIZE_CELL_Y)) 
        pygame.draw.rect(win, COLOR_SQUARE , (cell_pos_x, cell_pos_y, SIZE_CELL_X, SIZE_CELL_Y), BORDER_WIDTH_CELL) 
        draw_shape(win, elements[i], cell_pos_x + SIZE_BASE, cell_pos_y + SIZE_BASE)
        
        if (elements[i].id > 0):
            draw_text(win, str(elements[i].money), COLOR_MONEY, cell_pos_x + SIZE_BASE * 0.25, cell_pos_y)
            draw_text(win, str(elements[i].time), COLOR_TIME, cell_pos_x_end - SIZE_BASE * 0.75, cell_pos_y)
            draw_text(win, str(elements[i].buttons), COLOR_BUTTONS, cell_pos_x + SIZE_BASE * 0.25, cell_pos_y_end - SIZE_BASE * 1.25)
            draw_text(win, str(elements[i].space), COLOR_SPACE, cell_pos_x_end - SIZE_BASE * 0.75, cell_pos_y_end - SIZE_BASE * 1.25)
        else:
            draw_text(win, "Money {}".format(-elements[i].money), COLOR_SQUARE, cell_pos_x + SIZE_BASE * 0.25, cell_pos_y + SIZE_BASE * 3)
        
def draw_game_state(win, state, move_num, x, y):
    draw_text(win, 'PLAYER STATISTICS', COLOR_SQUARE, x + SIZE_BASE, y)
    draw_text(win, 'Move number {}'.format(move_num), COLOR_SQUARE, x + 17 * SIZE_BASE, y)
    draw_text(win, 'Last player {}'.format(state.player_last_move), COLOR_SQUARE, x + 17 * SIZE_BASE, y + 1.5 * SIZE_BASE)
    draw_player(win, state.player1_stats, 1,  x, y + 1.5 * SIZE_BASE)
    draw_player(win, state.player2_stats, 2,  x + 7 * SIZE_BASE, y + 1.5 * SIZE_BASE)
    draw_multiple_shapes(win, [state.last_move], [None], x + 17 * SIZE_BASE, y + 3 * SIZE_BASE)
    available_moves = [x.id for x in patchwork.patchwork.available_moves(state)]
    is_available = [None] * len(state.available_elements)
    for i in range(4):
        if state.available_elements[i].id in available_moves:
            is_available[i] = True
        else:
            is_available[i] = False
    draw_multiple_shapes(win, state.available_elements, is_available, x, y + 11 * SIZE_BASE)

    
def draw_board(win, p1_time, p2_time, x, y):
    for i in range(patchwork.MAX_TIME + 1):
        pygame.draw.rect(win, COLOR_SQUARE , (x + i * SIZE_BASE, y, SIZE_BASE, SIZE_BASE), BORDER_WIDTH_SQARE) 
    draw_text(win, "P1", COLOR_MONEY, x + p1_time * SIZE_BASE, y - 1.5 * SIZE_BASE)
    draw_text(win, "P2", COLOR_MONEY, x + p2_time * SIZE_BASE, y + SIZE_BASE)  

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
            current_move = min(current_move, len(g.game_states) - 1)
            next_move = min(current_move + 1, len(g.game_states) - 1)
            pygame.draw.rect(win, COLOR_BACKGROUND , (0, 0, SIZE_X, SIZE_Y))  
            draw_game_state(win, g.game_states[current_move], current_move, 0, 0)
            draw_game_state(win, g.game_states[next_move], next_move, 800, 0)
            draw_board(win, g.game_states[current_move].player1_stats.time, g.game_states[current_move].player2_stats.time, 0, 50 * SIZE_BASE)
        
        pygame.display.update()
                
    pygame.quit() 
    
def play_game(player):
    win = pygame.display.set_mode((SIZE_X, SIZE_Y), pygame.RESIZABLE)
    game = patchwork.patchwork(player, None, synchronous = False)
    
    run = True
    mouse_pos = (0, 0)
    
    current_move = 0
    while run:
        pygame.time.delay(100)

        
        current_move = max(current_move, 0)
        current_move = min(current_move, len(game.game_states) - 1)
        next_move = min(current_move + 1, len(game.game_states) - 1)
        
        pygame.draw.rect(win, COLOR_BACKGROUND , (0, 0, SIZE_X, SIZE_Y))
        state = copy.deepcopy(game.game_states[current_move])
        draw_game_state(win, state, current_move, 0, 0)
        draw_board(win, state.player1_stats.time, state.player2_stats.time, 0, 55 * SIZE_BASE)
        
        for event in pygame.event.get():
            if game.current_player() == 1:
                game.make_move()
                current_move = current_move + 1
                
            if event.type == pygame.QUIT:
                run = False
                            
            if event.type == pygame.VIDEORESIZE:
                win = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_move = current_move - 1
                if event.key == pygame.K_RIGHT:
                    current_move = current_move + 1
                    
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                mouse_pos = (mouse_pos[0] // SIZE_CELL_X, (mouse_pos[1] - 11 * SIZE_BASE) // SIZE_CELL_Y)
                
                if (game.is_end_game() == False) & (game.current_player() == 2) & (current_move + 1 == len(game.game_states)):
                    if (mouse_pos[0] <= 3) & (mouse_pos[1] == 0):
                        if game.current_state().available_elements[mouse_pos[0]] in game.available_moves():
                            game.make_move(game.current_state().available_elements[mouse_pos[0]])
                    current_move = current_move + 1
                
        textsurface = myfont.render("x={} y={}".format(mouse_pos[0], mouse_pos[1]), False, COLOR_MONEY)
        win.blit(textsurface, (600, 200)) 
        pygame.display.update()
                
    pygame.quit() 

#%% Initialisation

pygame.init()
myfont = pygame.font.SysFont('Comic Sans MS', SIZE_BASE) 