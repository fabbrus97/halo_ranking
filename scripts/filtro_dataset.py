import pymongo
from datetime import datetime



myclient = pymongo.MongoClient('')
myclient2 = pymongo.MongoClient("")
db = myclient["halo_infinite"]
db2= myclient2["FinaleHalo"]
Matches = db["matches"]
Matches2=db2["match_filtrato"]

def formattaData(x):
  x=x.replace("T"," ")
  x = x.split(".", 1)[0]
  return x

totale_match_idonei=[]
problema=0
i=0

for _match in Matches.find({}):

    joined_at = []
    joined_in_progress = []
    left_at = []
    kills = []
    deaths = []
    name = []
    team = []
    rank = []
    outcome = []
    player_type=[]

    if(type(_match['data'][0]['match'])!=type(None)):
        tempo_giocato_match=_match['data'][0]['match']['duration']['seconds']
        inizio_partita_match=int(datetime.timestamp(datetime.strptime( formattaData(_match['data'][0]['match']['players'][0]['participation']['joined_at']), '%Y-%m-%d %H:%M:%S')))
        fine_partita_match=inizio_partita_match+tempo_giocato_match
        totale_players_idonei=0
        tempo_minimo=90*tempo_giocato_match/100
        for _player in _match['data'][0]['match']['players']:

            inizio_partita_player = int(datetime.timestamp(datetime.strptime(formattaData(_player['participation']['joined_at']), '%Y-%m-%d %H:%M:%S')))
            if(_player['participation']['left_at']!=None or _player['outcome']!='left'):

                if(_player['participation']['left_at']==None):
                    tempo_giocato_player=fine_partita_match-inizio_partita_player
                else:
                    fine_partita_player=int(datetime.timestamp(datetime.strptime(formattaData(_player['participation']['left_at']), '%Y-%m-%d %H:%M:%S')))
                    tempo_giocato_player=fine_partita_player-inizio_partita_player
                if (tempo_giocato_player >= tempo_minimo):
                    totale_players_idonei = totale_players_idonei + 1
                    name.append(_player['details']['name'])
                    team.append(_player['team']['id'])
                    rank.append(_player['rank'])
                    outcome.append(_player['outcome'])
                    joined_at.append(_player['participation']['joined_at'])
                    joined_in_progress.append(_player['participation']['joined_in_progress'])
                    left_at.append(_player['participation']['left_at'])
                    player_type.append(_player['details']['type'])
                    if(_player['performances']!=None):
                        kills.append(_player['performances']['kills']['count'])
                        deaths.append(_player['performances']['deaths']['count'])
                    else:
                        kills.append(0)
                        deaths.append(0)

        if(totale_players_idonei==8 and sum(team)==4):#cioè 4 nel team 1 e 4 nel team 0
            totale_match_idonei.append(_match['data'][0]['match']['id'])
            if(_match['data'][0]['match']['teams']['details'][0]['outcome']=='win'): team_winner=1
            elif(_match['data'][0]['match']['teams']['details'][0]['outcome']=='loss'): team_winner=2
            else: team_winner=0
            db2.match_filtrato_v2.insert_one(
                {
                    'id':_match['data'][0]['match']['id'],
                    'name_arena':_match['data'][0]['match']['details']['gamevariant']['name'],
                    'team_winner':team_winner,
                    'played_at':_match['data'][0]['match']['played_at'],
                    'duration':_match['data'][0]['match']['duration']['seconds'],
                    'players':[
                        {
                            'name':name[0],
                            'type':player_type[0],
                            'team':team[0],
                            'rank':rank[0],
                            'skill': 0,
                            'outcome':outcome[0],
                            'duration':{
                                'joined_in_progress':joined_in_progress[0],
                                'joined_at':joined_at[0],
                                'left_at':left_at[0]
                            },
                            'performances':{
                                'kills':kills[0],
                                'death':deaths[0]
                            }
                        },

                        {
                            'name': name[1],
                            'type': player_type[1],
                            'team': team[1],
                            'rank': rank[1],
                            'skill': 0,
                            'outcome': outcome[1],
                            'duration': {
                                'joined_in_progress': joined_in_progress[1],
                                'joined_at': joined_at[1],
                                'left_at': left_at[1]
                            },
                            'performances': {
                                'kills': kills[1],
                                'death': deaths[1]
                            }
                        },
                        {
                            'name': name[2],
                            'type': player_type[2],
                            'team': team[2],
                            'rank': rank[2],
                            'skill': 0,
                            'outcome': outcome[2],
                            'duration': {
                                'joined_in_progress': joined_in_progress[2],
                                'joined_at': joined_at[2],
                                'left_at': left_at[2]
                            },
                            'performances': {
                                'kills': kills[2],
                                'death': deaths[2]
                            }
                        },
                        {
                            'name': name[3],
                            'type': player_type[3],
                            'team': team[3],
                            'rank': rank[3],
                            'skill': 0,
                            'outcome': outcome[3],
                            'duration': {
                                'joined_in_progress': joined_in_progress[3],
                                'joined_at': joined_at[3],
                                'left_at': left_at[3]
                            },
                            'performances': {
                                'kills': kills[3],
                                'death': deaths[3]
                            }
                        },
                        {
                            'name': name[4],
                            'type': player_type[4],
                            'team': team[4],
                            'rank': rank[4],
                            'skill': 0,
                            'outcome': outcome[4],
                            'duration': {
                                'joined_in_progress': joined_in_progress[4],
                                'joined_at': joined_at[4],
                                'left_at': left_at[4]
                            },
                            'performances': {
                                'kills': kills[4],
                                'death': deaths[4]
                            }
                        },
                        {
                            'name': name[5],
                            'type': player_type[5],
                            'team': team[5],
                            'rank': rank[5],
                            'skill': 0,
                            'outcome': outcome[5],
                            'duration': {
                                'joined_in_progress': joined_in_progress[5],
                                'joined_at': joined_at[5],
                                'left_at': left_at[5]
                            },
                            'performances': {
                                'kills': kills[5],
                                'death': deaths[5]
                            }
                        },
                        {
                            'name': name[6],
                            'type': player_type[6],
                            'team': team[6],
                            'rank': rank[6],
                            'skill': 0,
                            'outcome': outcome[6],
                            'duration': {
                                'joined_in_progress': joined_in_progress[6],
                                'joined_at': joined_at[6],
                                'left_at': left_at[6]
                            },
                            'performances': {
                                'kills': kills[6],
                                'death': deaths[6]
                            }
                        },
                        {
                            'name': name[7],
                            'type': player_type[7],
                            'team': team[7],
                            'rank': rank[7],
                            'skill':0,
                            'outcome': outcome[7],
                            'duration': {
                                'joined_in_progress': joined_in_progress[7],
                                'joined_at': joined_at[7],
                                'left_at': left_at[7]
                            },
                            'performances': {
                                'kills': kills[7],
                                'death': deaths[7]
                            }
                        }
                    ]

                })

