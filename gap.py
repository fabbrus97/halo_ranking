
import pymongo



myclient = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = myclient["dbTestHalo"]
Matches = db["Collection"]
#{"data.0.match.details.gamevariant.name": "Arena:CTF"}
totale=0
skill_gap_team0=[]
skill_gap_team1=[]
team_winner=[]
team_sbilanciato=[]
ipotesi=[]



for _match in Matches.find({"data.0.match.details.gamevariant.name": "Arena:Slayer"}):
    min_team0=1000
    min_team1=1000
    max_team0=-1000
    max_team1=-1000
    if(len(_match['data'][0]['match']['players'])==8):
        for _player in _match['data'][0]['match']['players']:
            if(_player['team']['id']==0):
                if(_player['rank']>max_team0):max_team0=_player['rank']
                if(_player['rank']<min_team0):min_team0=_player['rank']
            else:
                if (_player['rank'] > max_team1): max_team1 = _player['rank']
                if (_player['rank'] < min_team1): min_team1 = _player['rank']
        if((((max_team0-min_team0)*100/max_team0)>60 and ((max_team1-min_team1)*100/max_team1)<20)
                or (((max_team0-min_team0)*100/max_team0)<20 and ((max_team1-min_team1)*100/max_team1)>60)):
            skill_gap_team0.append((max_team0-min_team0)*100/max_team0)
            skill_gap_team1.append((max_team1-min_team1)*100/max_team1)
            if(((max_team0-min_team0)*100/max_team0)>((max_team1-min_team1)*100/max_team1)):team_sbilanciato.append(0)
            else: team_sbilanciato.append(1)
            if (_match['data'][0]['match']['teams']['details'][0]['outcome'] == 'win'):
                team_winner.append(0)
            else:
                team_winner.append(1)




for i in range(len(team_sbilanciato)):
    if(team_sbilanciato[i]==team_winner[i]): ipotesi.append("no")
    else: ipotesi.append("si")

print(len(ipotesi))
print(ipotesi.count("si"))
print(len(skill_gap_team0))