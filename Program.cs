namespace ts2
{
    using System;
    using System.Linq;
    using System.Text.Json;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Models;
    using Newtonsoft.Json.Linq;
    using MongoDB.Bson;
    using MongoDB.Driver;
    using System.Text.RegularExpressions;
    using System.Collections.Generic;

    using Range = Microsoft.ML.Probabilistic.Models.Range;


    class Program
    {

        static string MONGO_STRING = "mongodb://hds:2wuzA5fRdGVwU2@192.168.1.16:27017/halo_infinite"; 
        static string MONGO_DB = "halo_infinite"; 
        static string MONGO_COLLECTION = "matches"; 

        static List<List<Match>> SplitToSublists(List<Match> source,int parametro)
        {
            return source
                     .Select((x, i) => new { Index = i, Value = x })
                     .GroupBy(x => x.Index / parametro)
                     .Select(x => x.Select(v => v.Value).ToList())
                     .ToList();
        }

        static List<List<Match>> split_match_mode(List<Match> Matches,string[] mode_array){
          List<List<Match>> match_per_mode=  new List<List<Match>>();
          for (int i = 0; i < mode_array.Length; i++){
            match_per_mode.Add(new List<Match>());
          }

          foreach (var _mode in mode_array){
            foreach(var _match in Matches){
              if (_match.mode.Equals(_mode)){
                match_per_mode[IndexOf(mode_array,_mode)].Add(_match);
              }
            }
          }
          return match_per_mode;
        }

        static int IndexOf(string[] mod,string cercare){
          int index=0;
          while(mod[index]!= cercare && index<mod.Length+1){
            index=index+1;
                    }
          if(index==mod.Length+1) index=-1;
          return index;
        }


        static List<Match> getMatches(string play_mode, int skip)
        {

          Console.WriteLine("DEBUG Chiedo i match al dataset...");

          var dbClient = new MongoClient("mongodb://hds:2wuzA5fRdGVwU2@192.168.1.16:27017/halo_infinite");
          IMongoDatabase db = dbClient.GetDatabase("halo_infinite");
          var Matches = db.GetCollection<BsonDocument>("matches");
          // var dbClient = new MongoClient("mongodb://127.0.0.1:27017");
          // IMongoDatabase db = dbClient.GetDatabase("dbTestHalo");
          // var Matches = db.GetCollection<BsonDocument>("Collection");

          // var o = Matches.Find(new BsonDocument()).Limit(10).ToList();
          var o = Matches.Find("{data: { $elemMatch : { 'match.details.gamevariant.name': 'Arena:" + play_mode + "'}}}").Limit(1000).Skip(skip).ToList();
          var matches = new List<Match>();


          DateTime origin = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);

          int max = 10000;

          foreach (var _match in o)
          {

              if (max-- < 0)
                  break;
              string play_mode_mongo=(string)(_match["data"][0]["match"]["details"]["gamevariant"]["name"]);
              play_mode_mongo= play_mode_mongo[6..];
              if(play_mode==play_mode_mongo)
              {
                var team1players = new List<TeamPlayer>();
                var team2players = new List<TeamPlayer>();
                var secondsPlayed = (int)_match["data"][0]["match"]["duration"]["seconds"];
                var players_string = _match["data"][0]["match"]["players"].ToJson();
                int num_players=Regex.Matches(players_string, "details").Count;
                for (int i=0;i<num_players;i=i+1)
              {
                  var tag = (string)_match["data"][0]["match"]["players"][i]["details"]["name"];
                  var isBot = Equals(((string)_match["data"][0]["match"]["players"][i]["details"]["type"]), "bot");
                  var date_join=(string)_match["data"][0]["match"]["players"][i]["participation"]["joined_at"];

                  DateTime datetimeJoin = DateTime.Parse( date_join.Substring(1, date_join.Length-2));//JsonSerializer.Deserialize<DateTime>((string)_player["participation"]["joined_at"])!;
                  TimeSpan diff = datetimeJoin.ToUniversalTime() - origin;

                  var joinTime = Math.Floor(diff.TotalSeconds);

                  var endTime = 0.0;



                  int kcount = (int)_match["data"][0]["match"]["players"][i]["stats"]["core"]["summary"]["kills"];
                  int dcount = (int)_match["data"][0]["match"]["players"][i]["stats"]["core"]["summary"]["deaths"];

                  if (!(_match["data"][0]["match"]["players"][i]["participation"]["left_at"].GetType().ToString()=="MongoDB.Bson.BsonNull")){
                      var date_left=(string)_match["data"][0]["match"]["players"][i]["participation"]["left_at"];
                      DateTime datetimeLeft = DateTime.Parse( date_left.Substring(1, date_left.Length-2));
                      diff = datetimeLeft.ToUniversalTime() - origin;
                      endTime = Math.Floor(diff.TotalSeconds);
                  }
                  else
                      endTime = joinTime + secondsPlayed;
                  if ((int)_match["data"][0]["match"]["players"][i]["team"]["id"] == 0)
                      team1players.Add(new TeamPlayer(tag, endTime - joinTime, joinTime, endTime, kcount, dcount, (bool)_match["data"][0]["match"]["players"][i]["participation"]["presence"]["completion"], isBot));
                  else
                      team2players.Add(new TeamPlayer(tag, endTime - joinTime, joinTime, endTime, kcount, dcount, (bool)_match["data"][0]["match"]["players"][i]["participation"]["presence"]["completion"], isBot));


              }

              var team1 = new Team(team1players);
              var team2 = new Team(team2players);
              string outcome = (string)(_match["data"][0]["match"]["teams"]["details"][0]["outcome"]);
              var winner = Match.Winner.DRAW;
              if (outcome == "loss"){
                  winner = Match.Winner.TEAM2;
              }
              else if (outcome == "win"){
                  winner = Match.Winner.TEAM1;
              }
              string mode=(string)(_match["data"][0]["match"]["details"]["gamevariant"]["name"]);
              mode=mode[6..];




              string id = (string)_match["data"][0]["match"]["id"];
              var played_at=(string)_match["data"][0]["match"]["played_at"];
              var startTime = Math.Floor(((DateTime.Parse( played_at.Substring(1, played_at.Length-2))).ToUniversalTime() - origin).TotalSeconds);
              var endTimeMatch = startTime + secondsPlayed;

              var match = new Match(team1, team2, winner, id, mode, startTime, endTimeMatch, secondsPlayed);
              matches.Add(match);
            } 
          
          }

          Console.WriteLine("DEBUG match ottenuti");
          return matches;
        }



      static void updateMongo(List<Match> sottolista, Gaussian[] skills, string[] nomi){



        var dbClient = new MongoClient(MONGO_STRING);
        IMongoDatabase db = dbClient.GetDatabase(MONGO_DB);
        var Matches = db.GetCollection<BsonDocument>(MONGO_COLLECTION);

        foreach (Match item in sottolista){

          var filter = Builders<BsonDocument>.Filter.Eq("data.0.match.id", item.id);

          for(int i=0;i<item.team1.nPlayers();i=i+1){
            var skill_player=skills[IndexOf(nomi,item.team1.teammates[i].tag)].GetMean();
            var update = Builders<BsonDocument>.Update.Set("data.0.match.players."+i+".rank", skill_player);
            Matches.UpdateOne(filter, update);

          }
          for(int i=0;i<item.team2.nPlayers();i=i+1){
            var skill_player=skills[IndexOf(nomi,item.team2.teammates[i].tag)].GetMean();
            var tot_i=i+item.team1.nPlayers();
            var update = Builders<BsonDocument>.Update.Set("data.0.match.players."+tot_i+".rank", skill_player);
            Matches.UpdateOne(filter, update);
          }


        }
      }

      static void calcoloSkill(string moda_scelta){

        Console.WriteLine("Hai scelto la modalità: "+moda_scelta);
        var skip = 0;
        var matches = getMatches(moda_scelta, skip);
        var parcalc = new ParamsCalculator();
        Console.WriteLine("Inizio calcolo dei parametri...");
        
        //TODO setta parametri a seconda della partita
        // var Players_arr = parcalc.players(matches);
        // var Players_arr = parcalc.players(matches).ToArray();
        // Gaussian[] Skillls = new Gaussian[Players_arr.Length];
        List<Gaussian> Skillls = new List<Gaussian>();
        List<double> last_game_all_players = new List<double>();
        List<int> experience_all_players = new List<int>();
        List<string> players_fino_ad_ora=  new List<string>();


        // var sublists = SplitToSublists(matches,1000);
        // // var sublists = SplitToSublists(matches,50);
        // Console.WriteLine("Ci sono "+sublists.Count+" sublist");
        // var totale=0;
        // for(var i=0;i<sublists.Count;i=i+1){
        //   Console.WriteLine("La sublist numero "+ i + " ha "+ sublists[i].Count+" match");
        //   totale=totale+ sublists[i].Count;
        // }
        // Console.WriteLine("In totale ci sono "+totale+" match per la modalità "+moda_scelta );


        // for (int i=0;i<sublists.Count;i=i+1){
        while (matches.Count() > 0) 
        {

          // Console.WriteLine("Sto analizzando la sublist numero "+i);
          // parcalc.SetMatches(sublists[i]);
          parcalc.SetMatches(matches);

          // var appoggio_player = parcalc.players(sublists[i]).ToArray();
          var appoggio_player = parcalc.players(matches).ToArray();
          double[][][] timepassed = new double[matches.Count()][][];
          int[][] experience = new int[appoggio_player.Count()][];

          var arr_players_common = players_fino_ad_ora.Intersect(appoggio_player);
          if(skip>0){
          // if(i>0){
            Gaussian[] baseskills = new Gaussian[appoggio_player.Length];
            Console.WriteLine("I giocatori già incontrati nei precedenti gruppi sono: "+arr_players_common.Count());
            foreach(string item in arr_players_common){
              // baseskills[IndexOf(appoggio_player,item)]=Skillls[Players_arr.IndexOf(item)];
              var tmpIndex = players_fino_ad_ora.IndexOf(item);

              baseskills[IndexOf(appoggio_player,item)]
                = Skillls[tmpIndex];
            } 

            parcalc.SetBaseSkills(baseskills);

          }

          var nomi_nuovi=appoggio_player.Except(arr_players_common).ToList();
          foreach(string item in nomi_nuovi){
            players_fino_ad_ora.Add(item);
            experience_all_players.Add(0);
            last_game_all_players.Add(0);
          }

          //init experience and time_passed
          for (int i = 0; i < appoggio_player.Count(); i++)
          {
                experience[i] = new int[matches.Count()];
                //FIXME: è un po' uno spreco di memoria perché mi salvo l'esperienza per ogni partita
                //quindi se tra le partite 200 e 300 il giocatore non gioca, mi salvo comunque 100 volte 
                //lo stesso livello di esperienza (perché non cambia - perché il giocatore non ha giocato)
                for (int j = 0; j < matches.Count(); j++)
                {
                    experience[i][j] = -1;
                }
          }
          
          for(int i = 0; i < matches.Count(); i++){
            timepassed[i] = new double [2][];
            timepassed[i][0] = new double[matches[i].team1.nPlayers()];
            timepassed[i][1] = new double[matches[i].team2.nPlayers()];

            for (int k = 0; k < matches[i].team1.nPlayers(); k++){
              var player = IndexOf(appoggio_player, matches[i].team1.teammates[k].tag);
              var playerGlobalIndex = players_fino_ad_ora.FindIndex(t => t == matches[i].team1.teammates[k].tag);

                
              timepassed[i][0][k] = 0.0; //TODO se si analizzano le partite a batch di 1000, tra un batch e l'altro ci si perde l'ultima partita del giocatore
              if (i > 0)
              {   
                  //search the last game of the player
                  for (int s = 0; s < i ; s++)
                  {
                      var pl = matches[s].team1.teammates.FindIndex(t => t.tag == matches[i].team1.teammates[k].tag);
                      if (pl < 0)
                          pl = matches[s].team2.teammates.FindIndex(t => t.tag == matches[i].team1.teammates[k].tag);
                      if (pl > 0)
                      {
                          if (arr_players_common.Contains(appoggio_player[i]))
                            timepassed[i][0][k] = matches[i].startTime - last_game_all_players[playerGlobalIndex]; 
                          else
                            timepassed[i][0][k] = matches[i].startTime - matches[s].startTime;
                          last_game_all_players[playerGlobalIndex] = matches[i].startTime;
                          break;
                      }

                  }
                  
              }
              

              if (experience[player][0] < 0){ //TODO se si analizzano le partite a batch di 1000, tra un batch e l'altro ci si perde l'esperienza del giocatore
                  
                  if (arr_players_common.Contains(appoggio_player[i]))
                  {
                    
                    for (int x = 0; x < i; x++)
                        experience[player][x] = experience_all_players[playerGlobalIndex];
                    experience[player][i] = experience_all_players[playerGlobalIndex] + 1;

                    experience_all_players[playerGlobalIndex] += 1;
                  } 
                  else 
                  {
                    for (int x = 0; x < i; x++)
                        experience[player][x] = 0;
                    experience[player][i] = 1;

                    experience_all_players[playerGlobalIndex] = 1;
                  }
                  
              }
              else
              {
                  int searchMatch = 1; 
                  while(searchMatch < i)
                  {
                      if (experience[player][searchMatch] < 0)    
                          experience[player][searchMatch] = experience[player][searchMatch-1];
                      searchMatch += 1;
                  }
                  experience[player][i] = experience[player][i-1] + 1;

                  experience_all_players[playerGlobalIndex] += 1;
              }
            }

            for (int k = 0; k < matches[i].team2.nPlayers(); k++){
              var player = IndexOf(appoggio_player, matches[i].team2.teammates[k].tag);
              var playerGlobalIndex = players_fino_ad_ora.FindIndex(t => t == matches[i].team2.teammates[k].tag);
              
              timepassed[i][1][k] = 0.0; //TODO se si analizzano le partite a batch di 1000, tra un batch e l'altro ci si perde l'ultima partita del giocatore
              if (i > 0)
              {   
                  //search the last game of the player
                  for (int s = 0; s < i ; s++)
                  {
                      var pl = matches[s].team1.teammates.FindIndex(t => t.tag == matches[i].team2.teammates[k].tag);
                      if (pl < 0)
                          pl = matches[s].team2.teammates.FindIndex(t => t.tag == matches[i].team2.teammates[k].tag);
                      if (pl > 0)
                      {
                          if (arr_players_common.Contains(appoggio_player[i]))
                            timepassed[i][1][k] = matches[i].startTime - last_game_all_players[playerGlobalIndex]; 
                          else
                            timepassed[i][1][k] = matches[i].startTime - matches[s].startTime;
                          last_game_all_players[playerGlobalIndex] = matches[i].startTime;

                          // timepassed[i][0][k] = matches[i].startTime - matches[s].startTime;
                          break;
                      }

                  }
                  
              }
              

              if (experience[player][0] < 0){ //TODO se si analizzano le partite a batch di 1000, tra un batch e l'altro ci si perde l'esperienza del giocatore
                  if (arr_players_common.Contains(appoggio_player[i]))
                  {
                    
                    for (int x = 0; x < i; x++)
                        experience[player][x] = experience_all_players[playerGlobalIndex];
                    experience[player][i] = experience_all_players[playerGlobalIndex] + 1;

                    experience_all_players[playerGlobalIndex] += 1;
                  } 
                  else 
                  {
                    for (int x = 0; x < i; x++)
                        experience[player][x] = 0;
                    experience[player][i] = 1;

                    experience_all_players[playerGlobalIndex] = 1;
                  }
                  
              }
              else
              {
                  int searchMatch = 1; 
                  while(searchMatch < i)
                  {
                      if (experience[player][searchMatch] < 0)    
                          experience[player][searchMatch] = experience[player][searchMatch-1];
                      searchMatch += 1;
                  }
                  experience[player][i] = experience[player][i-1] + 1;

                  experience_all_players[playerGlobalIndex] += 1;
              }
            }
          }

          if (skip > 36000) //TODO bisogna calcolare il totale delle partite e la percentuale delle partite per le quali vogliamo calcolare l'accuracy
          {
            // parcalc.PredictAccuracy();
            parcalc.predictAccuracy(matches);
          }
          var appoggio_skill = parcalc.ComputeSkills(timepassed, experience);
          foreach(string item in appoggio_player){
            // Skillls[Players_arr.IndexOf(item)] = appoggio_skill[IndexOf(appoggio_player,item)]; //Mi salvo le skill
            var tmpIndex = players_fino_ad_ora.IndexOf(item);
            var tmpSkill = appoggio_skill[IndexOf(appoggio_player,item)]; 
            //Mi salvo le skill
            if (Skillls.Count() > tmpIndex) //vero solo dopo le prime 1000 partite
              Skillls[tmpIndex] = tmpSkill;
            else
              Skillls.Add(tmpSkill);
          }
          // updateMongo(sublists[i],appoggio_skill,appoggio_player); TODO

          // Console.WriteLine("Nel sottogruppo "+ i+ " ho "+ appoggio_player.Length+" giocatori");
          // Console.WriteLine("Nel sottogruppo "+ i+ " ho "+  appoggio_skill.Length+" skills");
          Console.WriteLine("Ora il numero di giocatori incontrati è "+players_fino_ad_ora.Count);

          Console.WriteLine("\n=========================================================");
          Console.WriteLine("Analizzate partite dalla " + skip + " alla " + (skip + 1000));
          Console.WriteLine("=========================================================\n");

          skip += 1000;
          
          matches = getMatches(moda_scelta, skip);
        }
      

      }




        static void Main(string[] args)
        {

          string[] mode_array = new string[] { "CTF","Slayer","Strongholds","Oddball","One Flag CTF"};
          Console.WriteLine("Inserisci il nome delle modalità di gioco che vuoi analizzare, separate dalla virgola senza spazi");
          // string moda_utente = Console.ReadLine();
          string moda_utente = "CTF";
          List<string> result = moda_utente.Split(',').ToList();

          for(int i=0;i<result.Count;i=i+1) calcoloSkill(result[i]);

          //List<List<Match>> match_per_mode=split_match_mode(matches,mode_array);



          //Console.WriteLine("Precisione previsioni partite: " + parcalc.predictAccuracy(matches2predict));



        }


    }

}
