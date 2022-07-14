import pymongo

def aggiustamento_skill(mod):
    myclient = pymongo.MongoClient('mongodb://hds:2wuzA5fRdGVwU2@2.224.244.126:27017/halo_infinite')
    db = myclient["halo_infinite"]
    Matches = db["filteredmatches"]
    agg=0
    min=0
    for _match in Matches.find({"name_arena": mod}):
        for _player in _match['players']:
                if (_player['skill'] < min):
                    min = _player['skill']

    if(min<0):agg=1.1-min
    return agg

def calcologap(mod,n):

    myclient = pymongo.MongoClient('mongodb://hds:2wuzA5fRdGVwU2@2.224.244.126:27017/halo_infinite')
    db = myclient["halo_infinite"]
    Matches = db["filteredmatches"]

    skill_gap_team0=[]
    skill_gap_team1=[]
    team_winner=[]
    team_sbilanciato=[]
    match_in_questione=[]
    ipotesi=[]
    agg=aggiustamento_skill(mod)

    for _match in Matches.find({"name_arena": mod}):
        min_team0=0
        min_team1=0
        max_team0=0
        max_team1=0
        inizio_0=True
        inizio_1=True

        for _player in _match['players']:
            if(_player['team']==0):
                if(inizio_0):
                    max_team0=_player['skill']+agg
                    min_team0=_player['skill']+agg
                    inizio_0=False
                else:
                    if (_player['skill'] > max_team0): max_team0 = _player['skill']+agg
                    if (_player['skill'] < min_team0): min_team0 = _player['skill']+agg
            else:
                if(inizio_1):
                    max_team1 = _player['skill'] + agg
                    min_team1 = _player['skill'] + agg
                    inizio_1 = False
                else:
                    if (_player['skill'] > max_team1): max_team1 = _player['skill']+agg
                    if (_player['skill'] < min_team1): min_team1 = _player['skill']+agg

        if (((1 - (min_team0 / max_team0)) * 100 > n - 10 and (1 - (min_team1 / max_team1)) * 100 < 10 and (
                1 - (min_team0 / max_team0)) * 100 < n)
                or ((1 - (min_team0 / max_team0)) * 100 < 10 and (1 - (min_team1 / max_team1)) * 100 < n and (
                        1 - (min_team1 / max_team1)) * 100 > n - 10)):

            skill_gap_team0.append((1-(min_team0/max_team0))*100)
            skill_gap_team1.append((1-(min_team1/max_team1))*100)

            if((1-(min_team0/max_team0))*100>(1-(min_team1/max_team1))*100):
                team_sbilanciato.append(0)
            else:
                team_sbilanciato.append(1)
            if (_match['team_winner'] == 1):
                team_winner.append(0)
            elif(_match['team_winner'] == 2):
                team_winner.append(1)
            else: team_winner.append(2)
            match_in_questione.append(_match["id"])


    for i in range(len(team_sbilanciato)):
        if(team_sbilanciato[i]==team_winner[i]): ipotesi.append("si")
        elif(2==team_winner[i]):ipotesi.append("pareggio")
        else: ipotesi.append("si")
    return ipotesi

def test(freq_oss,freq_att,eventi):
    somma=0
    for i in range(len(freq_att)):
        somma+=pow((freq_oss[i]-freq_att[i]),2)/freq_att[i]
    eve=[3.84,5.99,7.82,9.49,11.07,12.59,14.07,15.51,16.92,18.31,19.68]
    if(somma>eve[eventi-2]):bool_evento=True
    else: bool_evento=False
    return somma,bool_evento

nome_arena=['Arena:Oddball','Arena:CTF','Arena:Slayer','Arena:Strongholds','Arena:One Flag CTF','Arena:Attrition','Arena:King of the Hill']


for j in range(len(nome_arena)):
    print("Sto calcolando per "+nome_arena[j])
    freq_oss=[]
    freq_att=[]
    for i in range(20,101,10):
        ip= calcologap(nome_arena[j], i)
        if(len(ip)!=0):
            freq_oss.append(ip.count("no"))
            freq_att.append((len(ip)-ip.count("pareggio"))/2)
    print("Test per "+nome_arena[j]+" ",test(freq_oss,freq_att,len(freq_att)))
