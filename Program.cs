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

        static double m0 = 3;           //definito nel paper di trueskill2
        static double v0 = 1.6;         //definito nel paper di trueskill2
        static double gamma = 10e-3;    //definito nel paper di trueskill2
        static double gamma_sqr = 10e-5;
        static double tau = 10e-8;      //definito nel paper di trueskill2
        static double beta = 1;         //definito nel paper di trueskill2
        static double beta_sqr = 1;
        static double epsilon = 10e-3;  //definito nel paper di trueskill2


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


        static List<Match> getMatches()
        {
          var dbClient = new MongoClient("mongodb://127.0.0.1:27017");
          IMongoDatabase db = dbClient.GetDatabase("dbTestHalo");
          var Matches = db.GetCollection<BsonDocument>("Collection");
          var o = Matches.Find(new BsonDocument()).ToList();
          var matches = new List<Match>();


          DateTime origin = new DateTime(1970, 1, 1, 0, 0, 0, 0, DateTimeKind.Utc);

          int max = 10000;

            foreach (var _match in o)
            {

                if (max-- < 0)
                    break;

                var team1players = new List<TeamPlayer>();
                var team2players = new List<TeamPlayer>();

                var secondsPlayed = (int)_match["data"][0]["match"]["duration"]["seconds"];
                var players_string = _match["data"][0]["match"]["players"].ToJson();
                int num_players=Regex.Matches(players_string, "details").Count;



                for (int i=0;i<num_players;i=i+1)
                {
                    var tag = (string)_match["data"][0]["match"]["players"][i]["details"]["name"];
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
                        team1players.Add(new TeamPlayer(tag, endTime - joinTime, joinTime, endTime, kcount, dcount, (bool)_match["data"][0]["match"]["players"][i]["participation"]["presence"]["completion"]));
                    else
                        team2players.Add(new TeamPlayer(tag, endTime - joinTime, joinTime, endTime, kcount, dcount, (bool)_match["data"][0]["match"]["players"][i]["participation"]["presence"]["completion"]));


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

            return matches;
        }

        static void Main(string[] args)
        {
          var matches = getMatches();
          string[] mode_array = new string[] { "CTF","Slayer","Strongholds","Oddball","One Flag CTF"};
          List<List<Match>> match_per_mode=split_match_mode(matches,mode_array);
          int totale=0;
          for(var i=0;i<mode_array.Length;i=i+1){
            Console.WriteLine("Per la modalità "+mode_array[i]+" ci sono "+ match_per_mode[i].Count+" match");
            totale=totale+match_per_mode[i].Count;
         }
          Console.WriteLine("In totale ci sono "+totale+" match");
          Console.WriteLine("Inizio calcolo dei parametri...");
            //Console.WriteLine(matches[0].team1.teammates[0].tag);
            //Console.WriteLine(matches[0].mode);

          var calc = new ParamsCalculator(matches);
          var parcalc = new ParamsCalculator();
            //TODO setta parametri a seconda della partita

            //skills è un array; per sapere a quale giocatore corrisponde la skill i-esima ci serve la lista dei giocatori
          var Players_arr = parcalc.players(match_per_mode[0]).ToArray();
          Console.WriteLine("Nella modalità "+mode_array[0]+" ci sono "+Players_arr.Length+" players" );
          Gaussian[] Skillls = new Gaussian[Players_arr.Length];
          List<string> players_fino_ad_ora=  new List<string>();





          var sublists = SplitToSublists(match_per_mode[0],50);
          Console.WriteLine("Ci sono "+sublists.Count+" sublist");
          totale=0;
          for(var i=0;i<sublists.Count;i=i+1){
            Console.WriteLine("La sublist numero "+ i + " ha "+ sublists[i].Count+" match");
            totale=totale+ sublists[i].Count;
         }
         Console.WriteLine("In totale ci sono "+totale+" match per la modalità "+mode_array[0] );



         for (int i=0;i<sublists.Count;i=i+1){
           parcalc.SetMatches(sublists[i]);
           var appoggio_player = parcalc.players(sublists[i]).ToArray();
           var arr_players_common = players_fino_ad_ora.Intersect(appoggio_player);
           if(i>0){
           Gaussian[] baseskills = new Gaussian[appoggio_player.Length];
           foreach(string item in arr_players_common){
             Console.WriteLine(item);
             baseskills[IndexOf(appoggio_player,item)]=Skillls[IndexOf(Players_arr,item)];}

            parcalc.SetBaseSkills(baseskills);

           }

           var nomi_nuovi=appoggio_player.Except(arr_players_common).ToList();
           foreach(string item in nomi_nuovi){
             players_fino_ad_ora.Add(item);
           }
           var appoggio_skill = parcalc.ComputeSkills();
           foreach(string item in appoggio_player){
               Skillls[IndexOf(Players_arr,item)]= appoggio_skill[IndexOf(appoggio_player,item)]; //Mi salvo le skill
             }

           Console.WriteLine("Nella sottogruppo"+ i+ "ho "+ appoggio_player.Length+" giocatori");
           Console.WriteLine("Nella sottogruppo"+ i+ "ho "+  appoggio_skill.Length+" skills");
           Console.WriteLine("Ora il numero di giocatori incontrati è"+players_fino_ad_ora.Count);
         }



            //Mi calcolo le skill per la prima modalità,primo blocco

/*

            //Mi calcolo le skill per la prima modalità,secondo blocco
            parcalc.SetMatches(sublists[1]);
            var appoggio_player1 = parcalc.players(sublists[1]).ToArray();
            Gaussian[] baseskills = new Gaussian[appoggio_player1.Length];
            var arr_players_common = appoggio_player1.Intersect(appoggio_player);
            Console.WriteLine("La lista dei giocatori in comune è:");

            foreach(string item in arr_players_common){
              Console.WriteLine(item);
              baseskills[IndexOf(appoggio_player1,item)]=appoggio_skill[IndexOf(appoggio_player,item)];}


            Console.WriteLine(baseskills[3].Equals(baseskills[3]));

            Console.WriteLine(baseskills[3]);

            parcalc.SetBaseSkills(baseskills);
            var appoggio_skill1= parcalc.ComputeSkills();



*/





/*
            foreach (Gaussian item in baseskills){
              Console.WriteLine(item);

            }

            foreach (string item in arr_players_common){
              Console.WriteLine(item);
                }




            foreach (Gaussian item in skills){
              Console.WriteLine(item);
                }


*/






            //Console.WriteLine("Precisione previsioni partite: " + parcalc.predictAccuracy(matches2predict));



        }


    }

}
