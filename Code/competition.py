import patchwork as patch
import pandas as pd
from joblib import Parallel, delayed

num_cores = 7

def competition(players, n):
    points = [0 for x in players]
    names = [x.name for x in players]
    games = [0 for x in players]
    for k in range(n):
        for i in range(len(players) - 1):
            for j in range(i+1, len(players)):
                g = patch.patchwork(players[i], players[j])
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
    df = df.sort_values(['points'], ascending=[0])
    return(df)

def competition2(players, n):
    iterations = len(players) * len(players) - 1
    result = []
    for i in range(len(players) - 1):
        for j in range(i+1, len(players)):
            winners = Parallel(n_jobs = num_cores)(delayed(game_results)(x) for x in [[players[i], players[j]]] * n)
            win1 = len(list(filter(lambda x: x == 1, winners)))
            win2 = len(list(filter(lambda x: x == 2, winners)))
            tie = len(list(filter(lambda x: x == 0, winners)))
            result.append(pd.DataFrame({'name_1': [players[i].name], 'name_2': [players[j].name], 'win_1': [win1], 'win_2': [win2], 'tie': [tie]}))
            print(result[-1])
    return(pd.concat(result))
    
def self_match(p):
    return(patch.patchwork(p, p).game_states)


def game_results(p):
    return(patch.patchwork(p[0], p[1]).winner())