namespace ts2
{
    using System;
    using System.Linq;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic.Distributions;
    using MongoDB.Bson;
    using MongoDB.Driver;
    



    class Program
    {

        static string MONGO_STRING = ""; 
        static string MONGO_DB = ""; 
        static string MONGO_COLLECTION = ""; 
        static IMongoCollection<BsonDocument> Matches;

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

          Console.WriteLine("Chiedo i match al dataset...");

          
          FindOptions options = new FindOptions{
            AllowDiskUse = true
          };
          var o = Matches.Find("{'name_arena': 'Arena:" + play_mode + "'}", options).Limit(1000).Skip(skip).Sort("{'played_at': 1}").ToList();
          var matches = new List<Match>();


          DateTime origin = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);

          int max = 10000;

          foreach (var _match in o)
          {
              
              if (max-- < 0)
                  break;
              string play_mode_mongo=(string)(_match["name_arena"]); 
              play_mode_mongo= play_mode_mongo[6..];
              if(play_mode==play_mode_mongo)
              {
                var team1players = new List<TeamPlayer>();
                var team2players = new List<TeamPlayer>();
                var secondsPlayed = (int)_match["duration"]; 
                var players_string = _match["players"].ToJson(); 
                int num_players= _match["players"].AsBsonArray.Count; 
                for (int i=0;i<num_players;i=i+1)
              {
                  var tag = (string)_match["players"][i]["name"]; 
                  var isBot = Equals(((string)_match["players"][i]["type"]), "bot"); 
                  var date_join=(string)_match["players"][i]["duration"]["joined_at"]; 

                  DateTime datetimeJoin = DateTime.Parse( date_join.Substring(1, date_join.Length-2));
                  TimeSpan diff = datetimeJoin.ToUniversalTime() - origin;

                  var joinTime = Math.Floor(diff.TotalSeconds);

                  var endTime = 0.0;



                  int kcount = (int)_match["players"][i]["performances"]["kills"]; 
                  int dcount = (int)_match["players"][i]["performances"]["death"]; 

                  if (!(_match["players"][i]["duration"]["left_at"].GetType().ToString()=="MongoDB.Bson.BsonNull")){
                      var date_left=(string)_match["players"][i]["duration"]["left_at"]; 
                      DateTime datetimeLeft = DateTime.Parse( date_left.Substring(1, date_left.Length-2));
                      diff = datetimeLeft.ToUniversalTime() - origin;
                      endTime = Math.Floor(diff.TotalSeconds);
                  }
                  else
                      endTime = joinTime + secondsPlayed;
                  if ((int)_match["players"][i]["team"] == 0)
                      team1players.Add(new TeamPlayer(tag, endTime - joinTime, joinTime, endTime, kcount, dcount, _match["players"][i]["outcome"] == "left", isBot));
                  else
                      team2players.Add(new TeamPlayer(tag, endTime - joinTime, joinTime, endTime, kcount, dcount, _match["players"][i]["outcome"] == "left", isBot));


              }

              var team1 = new Team(team1players);
              var team2 = new Team(team2players);
              var winner = Match.Winner.DRAW;
              if ((int)_match["team_winner"] == 1)
              {
                winner = Match.Winner.TEAM1;
              } else if ((int)_match["team_winner"] == 2)
              {
                winner = Match.Winner.TEAM2;
              }
              


              string id = (string)_match["id"]; 
              var played_at=(string)_match["played_at"]; 
              var startTime = Math.Floor((DateTime.Parse( played_at).ToUniversalTime() - origin).TotalSeconds);
              
              var endTimeMatch = startTime + secondsPlayed;

              var match = new Match(team1, team2, winner, id, play_mode_mongo, startTime, endTimeMatch, secondsPlayed);
              matches.Add(match);
            } 
          
          }

          Console.WriteLine("Match ottenuti");
          return matches;
        }



      static void updateMongo(List<Match> sottolista, Gaussian[] skills, string[] nomi){ 



        

        foreach (Match item in sottolista){

          var filter = Builders<BsonDocument>.Filter.Eq("id", item.id);
          

          for(int i=0;i<item.team1.nPlayers();i=i+1){
            var skill_player=skills[IndexOf(nomi,item.team1.teammates[i].tag)].GetMean();
            var update = Builders<BsonDocument>.Update.Set("players."+i+".skill", skill_player);
            
            Matches.UpdateOne(filter, update);

          }
          for(int i=0;i<item.team2.nPlayers();i=i+1){
            var skill_player=skills[IndexOf(nomi,item.team2.teammates[i].tag)].GetMean();
            var tot_i=i+item.team1.nPlayers();
            var update = Builders<BsonDocument>.Update.Set("players."+tot_i+".skill", skill_player);
            Matches.UpdateOneAsync(filter, update);
          }


        }
      }

      static int getTotGames(string play_mode)
      {
        
        int tot = (int)Matches.CountDocuments("{'name_arena': 'Arena:" + play_mode + "'}");
        return tot;
      }

      static void calcoloSkill(string moda_scelta){

        var skip = 0;
        var percentGamesToPredict = 10; //10%
        var matches = getMatches(moda_scelta, skip);
        var parcalc = new ParamsCalculator();
        Console.WriteLine("Inizio calcolo dei parametri...");
        
        List<Gaussian> Skillls = new List<Gaussian>();
        List<double> last_game_all_players = new List<double>();
        List<int> experience_all_players = new List<int>();
        List<string> players_fino_ad_ora=  new List<string>();




        int correctPredictions = 0;
        double totalPredicted = 0;
        var totGames = getTotGames(moda_scelta);
        int gamesToPredict = totGames - (int)((totGames/100)*percentGamesToPredict); 
        
        

        while (matches.Count() > 0) 
        {

          parcalc.SetMatches(matches);

          var appoggio_player = parcalc.players(matches).ToArray();
          double[][][] timepassed = new double[matches.Count()][][];
          int[][] experience = new int[appoggio_player.Count()][];

          var arr_players_common = players_fino_ad_ora.Intersect(appoggio_player);
          if(skip>0){
          
            Gaussian[] baseskills = new Gaussian[appoggio_player.Length];
            foreach(string item in arr_players_common){
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

                
              timepassed[i][0][k] = 0.0; 
              if (i == 0){
                last_game_all_players[playerGlobalIndex] = matches[i].startTime;      
                if (experience_all_players[playerGlobalIndex] == 0){
                  experience[player][i] = 1;
                  experience_all_players[playerGlobalIndex] = 1;
                } else {
                  experience[player][i] = experience_all_players[playerGlobalIndex];
                }
              }
              if (i > 0)
              {   
                if (last_game_all_players[playerGlobalIndex] != 0)
                  timepassed[i][0][k] = matches[i].startTime - last_game_all_players[playerGlobalIndex];      
                
                //search the last game of the player
                last_game_all_players[playerGlobalIndex] = matches[i].startTime;  
              
                if (experience_all_players[playerGlobalIndex] == 0)
                {
                  for (int x = 0; x < i; x++)
                    experience[player][x] = 0;
                  experience[player][i] = 1;
                }
                else 
                {
                  if (experience[player][0] < 0) 
                    experience[player][0] = experience_all_players[playerGlobalIndex];
                  int x = 0;
                  while(experience[player][x] >= 0)
                    x++;
                  for (int y = x; y < i; y++)
                    experience[player][y] = experience[player][y-1];
                  experience[player][i] = experience[player][i-1] + 1;
                }

                experience_all_players[playerGlobalIndex] += 1;
              }
            }

            for (int k = 0; k < matches[i].team2.nPlayers(); k++){
              var player = IndexOf(appoggio_player, matches[i].team2.teammates[k].tag);
              var playerGlobalIndex = players_fino_ad_ora.FindIndex(t => t == matches[i].team2.teammates[k].tag);
              
              timepassed[i][1][k] = 0.0; 
              if (i == 0){ //primo game del batch
                last_game_all_players[playerGlobalIndex] = matches[i].startTime;      
                if (experience_all_players[playerGlobalIndex] == 0){
                  experience[player][i] = 1;
                  experience_all_players[playerGlobalIndex] = 1;
                } else {
                  experience[player][i] = experience_all_players[playerGlobalIndex];
                }
              }
              if (i > 0)
              {   
                if (last_game_all_players[playerGlobalIndex] != 0)
                  timepassed[i][1][k] = matches[i].startTime - last_game_all_players[playerGlobalIndex];      
                  
                last_game_all_players[playerGlobalIndex] = matches[i].startTime;
                      
                if (experience_all_players[playerGlobalIndex] == 0)
                {
                  for (int x = 0; x < i; x++)
                    experience[player][x] = 0;
                  experience[player][i] = 1;
                }
                else 
                {
                  //abbiamo un giocatore con esperienza, ma per questo batch di partite non abbiamo ancora inizializzato
                  // il suo array dell'esperienza
                  if (experience[player][0] < 0) 
                    experience[player][0] = experience_all_players[playerGlobalIndex];
                  //altrimenti abbiamo un giocatore con esperienza che abbiamo già incontrato in questo batch
                  int x = 0;
                  while(experience[player][x] >= 0)
                    x++;
                  for (int y = x; y < i; y++)
                  {
                    experience[player][y] = experience[player][y-1];
                  }
                  experience[player][i] = experience[player][i-1] + 1;
                  
                  
                }
                
                experience_all_players[playerGlobalIndex] += 1;
              }
            }
          }

          int _correct = 0, offset = matches.Count() - 1;
          if (skip + 1000 > gamesToPredict && totalPredicted == 0)
          {
            offset = (gamesToPredict%1000);
            totalPredicted += (matches.Count() - offset);
          }
          if (skip > gamesToPredict)
          {
            offset = 0;
            totalPredicted += matches.Count();
          }
            
          var appoggio_skill = parcalc.ComputeSkills(timepassed, experience, offset, out _correct);
          correctPredictions += _correct;
         
          foreach(string item in appoggio_player){
            var tmpIndex = players_fino_ad_ora.IndexOf(item);
            var tmpSkill = appoggio_skill[IndexOf(appoggio_player,item)]; 
            //Mi salvo le skill
            if (Skillls.Count() > tmpIndex) //vero solo dopo le prime 1000 partite
              Skillls[tmpIndex] = tmpSkill;
            else
              Skillls.Add(tmpSkill);
          }
          updateMongo(matches,appoggio_skill,appoggio_player);

          Console.WriteLine("Il numero di giocatori incontrati è "+players_fino_ad_ora.Count);

          Console.WriteLine("\n=========================================================");
          Console.WriteLine("Analizzate partite dalla " + skip + " alla " + (skip + 1000));
          Console.WriteLine("=========================================================\n");

          skip += 1000;
          
          matches = getMatches(moda_scelta, skip);

          checkPlayers(players_fino_ad_ora, matches);
        }
      
        Console.WriteLine($"Predict accuracy: {correctPredictions/totalPredicted}");
      }

      static private void checkPlayers(List<string> old_players, List<Match> matches)
      {
        
        int[] known_player_per_match = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        foreach (var match in matches)
        {
          int known_players = 0;
          foreach (var player in match.team1.teammates)
          {
            if (old_players.FindIndex( p => p.SequenceEqual(player.tag)) >= 0)
              known_players += 1;
            
          }
          foreach (var player in match.team2.teammates)
          {
            if (old_players.FindIndex( p => p.SequenceEqual(player.tag)) >= 0)
              known_players += 1;
          }
          known_player_per_match[known_players]++;
        }
        
      }


      static void Main(string[] args)
      {

        var dbClient = new MongoClient(MONGO_STRING);
        IMongoDatabase db = dbClient.GetDatabase(MONGO_DB);
        Matches = db.GetCollection<BsonDocument>(MONGO_COLLECTION);

        string[] mode_array = new string[] { "CTF","Slayer","Strongholds","Oddball","One Flag CTF"};
        Console.WriteLine("Inserisci il nome delle modalità di gioco che vuoi analizzare, separate dalla virgola senza spazi");
        string moda_utente = Console.ReadLine();
        
        // 'Arena:Attrition', 
        // 'Arena:CTF', 
        // 'Arena:King of the Hill', 
        // 'Arena:Oddball', 
        // 'Arena:One Flag CTF', 
        // 'Arena:Slayer', 
        // 'Arena:Strongholds'  

        List<string> result = moda_utente.Split(',').ToList();

        for(int i=0;i<result.Count;i=i+1) calcoloSkill(result[i]);


      }


    }

}
