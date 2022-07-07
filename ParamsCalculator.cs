namespace ts2
{
    using System;
    using System.Linq;
    using System.Collections;
    using Microsoft.ML.Probabilistic;
    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Models;
    using Microsoft.ML.Probabilistic.Algorithms;
    using Microsoft.ML.Probabilistic.Models.Attributes;
    using Range = Microsoft.ML.Probabilistic.Models.Range;
    using Microsoft.ML.Probabilistic.Factors.Attributes;
    using Microsoft.ML.Probabilistic.Factors;
    using Microsoft.ML.Probabilistic.Compiler;

    class ParamsCalculator
    {
         /*
            Parameters to tune:
            m0,
            v0,
            γ,
            eps


            τ   = γ/10^5
            β   = 1 (by design)

         */
        bool ParameterComputed;
        private bool testAccuracy;
        double m0 = 3;  
        double v0 = 1.6;
        double gamma = 10e-3;
        double tau = 10e-8;
        double beta = 1;
        double epsilon = 10e-4;
	
	//CTF 49.5% accuracy
        //double w_k_p = 1.1e-5, w_k_o = -9.1e-5, w_d_p = -1.1e-5, w_d_o = 7.5e-5, v_c = 1e-5; //54% accuracy without m_q, v_q
	// test CTF
        // double w_k_p = 1.1e-5, w_k_o = -9.1e-5, w_d_p = -2.1e-5, w_d_o = 9.5e-5, v_c = 1e-5; //54% accuracy without m_q, v_q
	// test 2 CTF
        double w_k_p = 1.9*10e-6;
        double w_k_o = -9.5*10e-6;
        double w_d_p = -8.5*10e-6;
        double w_d_o = 1.1*10e-6;
        double v_c = 1*10e-6; //54% accuracy without m_q, v_q
        double m_q = 1.5*10e-6;
        double v_q = 1*10e-6;
        double unrelated = 0.5;
        double related = 0.5;
        Range nMatches;
        Range nPlayers;
        Range nTeamsPerMatch;
        VariableArray<VariableArray<int>, int[][]> playersInTeam;
        Range nPlayersPerTeam;
        Variable<bool> unrelatedVariable;
        Variable<bool> relatedVariable; 


        VariableArray<VariableArray<VariableArray<int>, int[][]>, int[][][]> matchesVariable;
        VariableArray<double> skillsVariable;
        VariableArray<double> skillsVarianceVariable;
        
        VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> killcount; 
        VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> deathcount; 
        VariableArray<int> outcomes; 
        
        VariableArray<double> matchTime; 
        VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> playerTime;

        // VariableArray<VariableArray<VariableArray<bool>, bool[][]>, bool[][][]> quit; //TODO

        // VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> timePassedVariable; TODO
        // VariableArray<double> experienceOffsetVariable; TODO
        // VariableArray<VariableArray<int>, int[][]> experienceVariable;  TODO
        // VariableArray<bool> humanPlayersVariable; TODO
        Gaussian[] BaseSkills;

        const int N_INTERATIONS = 500;


        // Gaussian[] skills;

        private List<Match> matches ;

        public ParamsCalculator()
        {
            this.ParameterComputed = false;
            this.testAccuracy = false;
        }


        public List<string> players(List<Match> matches)
        {
            if (matches == null)
                matches = this.matches;
            //TODO usare hash?
            var players = new List<string>();
            foreach (var m in matches)
            {
                foreach (var p in m.team1.teammates){
                    if (players.Find(tag => tag == p.tag) == null){
                        // se non trovi il giocatore, aumenta il conteggio
                        players.Add(p.tag);
                    }
                }
                foreach (var p in m.team2.teammates){
                    if (players.Find(tag => tag == p.tag) == null){
                        // se non trovi il giocatore, aumenta il conteggio
                        players.Add(p.tag);
                    }
                }
            }

            return players;
        }

        public void SetBaseSkills(Gaussian[] baseSkills)
        {
            this.BaseSkills = baseSkills;
        }

        

    
        public void SetMatches(List<Match> matches)
        {
            this.matches = matches;
        }

        private void SetVariables(int[][][] matchData, int nMatchesObserved, int nPlayersObserved, int[][] nPlayersPerTeamObserved, string[] playersName, int[] outcomesObserved,
            double[] matchTimeObserved, double[][][] playersTimeObserved, double[][][] killCountObserved, double[][][] deathCountObserved, bool[][][] quitObserved, 
            double[][][] timePassedObserved, int[][] experienceObserved, bool[] humanPlayersObserved)
        {
            
            nMatches = new Range(nMatchesObserved).Named("nMatches"); // a sample of n matches
            nPlayers = new Range(nPlayersObserved).Named("nPlayers"); // p unique players in the sample
            nTeamsPerMatch = new Range(2).Named("nTeamsPerMatch"); // 2 teams in each m
            playersInTeam = Variable.Array<int>(Variable.Array<int>(nTeamsPerMatch), nMatches).Named("PlayerInTeam");
            nPlayersPerTeam = new Range(playersInTeam[nMatches][nTeamsPerMatch]).Named("nPlayers-PerTeam"); // 4 players per team

            matchesVariable    = Variable.Array(Variable.Array(Variable.Array<int>   (nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("matches"); 
            skillsVariable     = Variable.Array<double>(nPlayers).Named("skills"); 
            killcount  = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("killcount"); 
            deathcount = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("deathcount"); 
            outcomes   = Variable.Array<int>(nMatches).Named("outcomes");
            matchTime  = Variable.Array<double>(nMatches).Named("matchTime"); 
            playerTime = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("playerTime");
            
            // quit       = Variable.Array(Variable.Array(Variable.Array<bool>   (nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("quit"); //TODO
            
            // timePassedVariable = Variable.Array(Variable.Array(Variable.Array<double>   (nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("time passed"); TODO
            // experienceVariable = Variable.Array(Variable.Array<int>(nMatches), nPlayers); TODO
            // experienceOffsetVariable = Variable.Array<double>(new Range(200)); TODO
            // humanPlayersVariable = Variable.Array<bool>(nPlayers); TODO

            relatedVariable = Variable.Bernoulli(related);
            unrelatedVariable = Variable.Bernoulli(unrelated);

            matchesVariable.ObservedValue = matchData;
            outcomes.ObservedValue = outcomesObserved;
            playersInTeam.ObservedValue = nPlayersPerTeamObserved;
            matchTime.ObservedValue = matchTimeObserved; 
            playerTime.ObservedValue = playersTimeObserved;
            killcount.ObservedValue = killCountObserved;
            deathcount.ObservedValue = deathCountObserved; 
            
            // quit.ObservedValue = quitObserved;  //TODO
            
            // timePassedVariable.ObservedValue = timePassedObserved; TODO e anche sotto
            // experienceVariable.ObservedValue = experienceObserved;
            // humanPlayersVariable.ObservedValue = humanPlayersObserved;

            Console.WriteLine("\n");
            Console.WriteLine("@@@@@@@@@ Observed Variables @@@@@@@@@");
            Console.WriteLine($"matchData size: {matchData.Count()} players: {nPlayersObserved} matchTime: {matchTimeObserved.Count()} experienceVariable: {experienceObserved.Count()}" );
            Console.WriteLine("\n");

            var experienceOffset = new double[200];
            var exp = 0.01; 
            for (int i = 0; i < 200; i += 1 )
            {
                experienceOffset[i] = exp; 
                exp += 0.01;
            }

            // experienceOffsetVariable.ObservedValue = experienceOffset; TODO

            if (BaseSkills != null)
            {
                setSkills();
            }

        }

        private void setSkills()
        {
            Console.WriteLine("Setto skills...");
            skillsVariable = Variable.Array<double>(nPlayers);
            skillsVarianceVariable = Variable.Array<double>(nPlayers);
            Console.WriteLine("m0 v0: " + m0 + " " + v0);
            int i = 0, m0v0 = 0;
            int _nPlayers = players(null).Count(); //TODO inefficiente
            var skillsArray = new double[_nPlayers]; 
            var varianceArray = new double[_nPlayers];
            // skillsVariable[nPlayers] = Variable.GaussianFromMeanAndVariance(m0, v0).ForEach(nPlayers); //TODO
            
            // Gaussian conf = new Gaussian();

            /*using (var player = Variable.ForEach(nPlayers))
            {
                // if (!(BaseSkills[i].Equals(conf)))
                if (!(BaseSkills[i].IsUniform()))
                    skillsVariable[player.Index] = Variable.GaussianFromMeanAndVariance(BaseSkills[i].GetMean(), BaseSkills[i].GetVariance());
                else
                    skillsVariable[player.Index] = Variable.GaussianFromMeanAndVariance(m0, v0);
                i += 1;
            }*/

            foreach (Gaussian skill in BaseSkills)
            {

                if (!skill.IsUniform())
                {
                    try
                    {
                        //Console.WriteLine("Skill esistente: " + skill.GetMean() + " " + skill.GetVariance());
                        // skillsVariable[i].SetTo(Variable.GaussianFromMeanAndVariance(skill.GetMean(), skill.GetVariance()));
                        skillsArray[i] = skill.GetMean();
                        varianceArray[i] = skill.GetVariance();
                    } catch (Microsoft.ML.Probabilistic.Distributions.ImproperDistributionException e)
                    {
                        Console.WriteLine("eccezione! \n" + e.StackTrace);
                        //skillsVariable[i] = Variable.GaussianFromMeanAndVariance(m0, v0);
                    }

                }
                else
                {
                    //Console.WriteLine("Nuovo giocatore: " + m0 + " " + v0);
                    skillsArray[i] = m0;
                    varianceArray[i] = v0; 
                    m0v0 += 1;


                }

                i += 1;
            }
            skillsVariable.ObservedValue = skillsArray;
            skillsVarianceVariable.ObservedValue = varianceArray; 
            Console.WriteLine("Ho settato " + i + " skill (m0v0: " + m0v0 + ")");
        }

        public Gaussian[] ComputeSkills(double[][][] timepassed, int[][] playersExperience)
        {
            //Init the Observed Values
            var _players = players(null);
            var nPlayers = _players.Count();
            int[] outcomes = new int[matches.Count()]; //array di vittorie/pareggi
            // le squadre sono 0 e 1
            int[][][] playersInGame = new int[matches.Count()][][];
            int[][] nPlayersPerTeam = new int[matches.Count()][];
            double[] matchTime = new double[matches.Count()];
            double[][][] playersTime = new double[matches.Count()][][];
            double[][][] deathcount = new double[matches.Count()][][];
            double[][][] killcount = new double[matches.Count()][][];
            bool[][][] quit = new bool[matches.Count()][][];
            // double[][][] timepassed = new double[matches.Count()][][];
            // int[][] playersExperience = new int[nPlayers][];
            bool[] humanPlayers = new bool[nPlayers];

            var gameInPlayersList = new List<List<int>>();
            for (int i = 0; i < nPlayers; i++)
            {
                gameInPlayersList.Add(new List<int>());
                // playersExperience[i] = new int[matches.Count()];
                //FIXME: è un po' uno spreco di memoria perché mi salvo l'esperienza per ogni partita
                //quindi se tra le partite 200 e 300 il giocatore non gioca, mi salvo comunque 100 volte 
                //lo stesso livello di esperienza (perché non cambia - perché il giocatore non ha giocato)
                // for (int j = 0; j < matches.Count(); j++)
                // {
                //     playersExperience[i][j] = -1;
                // }
            }



            var matchData = new int[matches.Count()][][];

            for(int i = 0; i < matches.Count(); i++){

                matchTime[i] = matches[i].secondsPlayed;

                nPlayersPerTeam[i] = new int[2];
                nPlayersPerTeam[i][0] = matches[i].team1.nPlayers();
                nPlayersPerTeam[i][1] = matches[i].team2.nPlayers();

                matchData[i] = new int[2][];
                matchData[i][0] = new int[matches[i].team1.nPlayers()];
                matchData[i][1] = new int[matches[i].team2.nPlayers()];

                playersTime[i] = new double[2][];
                playersTime[i][0] = new double[matches[i].team1.nPlayers()];
                playersTime[i][1] = new double[matches[i].team2.nPlayers()];

                killcount[i] = new double[2][];
                killcount[i][0] = new double[matches[i].team1.nPlayers()];
                killcount[i][1] = new double[matches[i].team2.nPlayers()];

                deathcount[i] = new double[2][];
                deathcount[i][0] = new double[matches[i].team1.nPlayers()];
                deathcount[i][1] = new double[matches[i].team2.nPlayers()];

                quit[i] = new bool[2][];
                quit[i][0] = new bool[matches[i].team1.nPlayers()];
                quit[i][1] = new bool[matches[i].team2.nPlayers()];


                playersInGame[i] = new int [2][];
                playersInGame[i][0] = new int[matches[i].team1.nPlayers()];
                playersInGame[i][1] = new int[matches[i].team2.nPlayers()];
                
                // timepassed[i] = new double [2][];
                // timepassed[i][0] = new double[matches[i].team1.nPlayers()];
                // timepassed[i][1] = new double[matches[i].team2.nPlayers()];

                for (int k = 0; k < matches[i].team1.nPlayers(); k++){
                    var player = _players.FindIndex(t => t == matches[i].team1.teammates[k].tag);
                    playersInGame[i][0][k] = player;
                    humanPlayers[player] = !matches[i].team1.teammates[k].bot; //player is not a bot => is human
                    gameInPlayersList[player].Add(i);

                    matchData[i][0][k] = player;

                    playersTime[i][0][k] = matches[i].team1.teammates[k].secondsPlayed;

                    killcount[i][0][k] = matches[i].team1.teammates[k].killcount;
                    deathcount[i][0][k] = matches[i].team1.teammates[k].deathcount;

                    quit[i][0][k] = !matches[i].team1.teammates[k].quit;

                    
                    /*timepassed[i][0][k] = 0.0; //TODO se si analizzano le partite a batch di 1000, tra un batch e l'altro ci si perde l'ultima partita del giocatore
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
                                timepassed[i][0][k] = matches[i].startTime - matches[s].startTime;
                                break;
                            }

                        }
                        
                    }*/
                    
                    /*if (playersExperience[player][0] < 0){ //TODO se si analizzano le partite a batch di 1000, tra un batch e l'altro ci si perde l'esperienza del giocatore
                        for (int x = 0; x < i; x++)
                            playersExperience[player][x] = 0;
                        playersExperience[player][i] = 1;
                    }
                    else
                    {
                        int searchMatch = 1; 
                        while(searchMatch < i)
                        {
                            if (playersExperience[player][searchMatch] < 0)    
                                playersExperience[player][searchMatch] = playersExperience[player][searchMatch-1];
                            searchMatch += 1;
                        }
                        playersExperience[player][i] = playersExperience[player][i-1] + 1;
                    }*/
                        
                    

                }
                for (int k = 0; k < matches[i].team2.nPlayers(); k++){
                    var player = _players.FindIndex(t => t == matches[i].team2.teammates[k].tag);
                    playersInGame[i][1][k] = player;
                    humanPlayers[player] = !matches[i].team2.teammates[k].bot; //player is not a bot => is human
                    gameInPlayersList[player].Add(i);


                    matchData[i][1][k] = player;

                    playersTime[i][1][k] = matches[i].team2.teammates[k].secondsPlayed;

                    killcount[i][1][k] = matches[i].team2.teammates[k].killcount;
                    deathcount[i][1][k] = matches[i].team2.teammates[k].deathcount;

                    quit[i][1][k] = !matches[i].team2.teammates[k].quit;

                    /*timepassed[i][1][k] = 0.0; //TODO se si analizzano le partite a batch di 1000, tra un batch e l'altro ci si perde l'ultima partita del giocatore
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
                                timepassed[i][1][k] = matches[i].startTime - matches[s].startTime;
                                break;
                            }

                        }
                        
                    }

                    if (playersExperience[player][0] < 0){ //TODO se si analizzano le partite a batch di 1000, tra un batch e l'altro ci si perde l'esperienza del giocatore
                        for (int x = 0; x < i; x++)
                            playersExperience[player][x] = 0;
                        playersExperience[player][i] = 1;
                    }
                    else
                    {
                        int searchMatch = 1; 
                        while(searchMatch < i)
                        {
                            if (playersExperience[player][searchMatch] < 0)    
                                playersExperience[player][searchMatch] = playersExperience[player][searchMatch-1];
                            searchMatch += 1;
                        }
                        playersExperience[player][i] = playersExperience[player][i-1] + 1;
                    }*/
                }


                if (matches[i].isDraw()){
                    outcomes[i] = 2;
                }
                else if (matches[i].isTeam1Winner()){
                    outcomes[i] = 0;
                }
                else
                    outcomes[i] = 1;


            }


            int[][] gameInPlayers = new int[nPlayers][];

            for (int i = 0; i < nPlayers; i++)
            {
                gameInPlayers[i] = new int[gameInPlayersList[i].Count()];
                gameInPlayers[i] = gameInPlayersList[i].ToArray();
            }

            SetVariables(matchData, matches.Count, nPlayers,  nPlayersPerTeam, _players.ToArray(), 
                outcomes, matchTime, playersTime, killcount, deathcount, quit, timepassed, playersExperience, humanPlayers);

            Gaussian[] skills;

            
            if (this.ParameterComputed)
            {
                Console.WriteLine("Calcolo skills (parametri già calcolati)");
                // skills = augmentVarianceTau(nPlayers); //TODO
                // SetBaseSkills(skills); setSkills(); //TODO
                Console.WriteLine("🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬🐬");
                Console.WriteLine("Accuracy: " + predictAccuracy(matches));
                skills = InferSkills();
                // Console.WriteLine("Inizio setbaseskillz");
                // SetBaseSkills(skills); setSkills(); //TODO
                // Console.WriteLine("setbaseskillz fatto");

                // Console.WriteLine("Inizio applicazione gamma");
                // skills = augmentVarianceGamma(nPlayers); //TODO
                // Console.WriteLine("Calcolo skillz terminato"); //TODO
            }
            else{
                
                ParameterComputed = true;
                // skills = inferTau(nPlayers);
                // SetBaseSkills(skills); setSkills();
                skills = InferSkillsAndParameters(matches.Count(), nPlayers);
                // SetBaseSkills(skills); setSkills(); //TODO
                // skills = inferGamma(nPlayers); //TODO
            }
            

            //DEBUG:
            int skillNotSet = 0;
            foreach (var skill in skills)
            {
                if (skill.GetMean() == m0)
                    skillNotSet += 1;
            }

            Console.WriteLine($"DEBUG ci sono {skillNotSet} skill non settate");

            return skills;
        }
        
        
        Gaussian[] InferSkillsAndParameters(int nMatchesObserved, int nPlayersObserved)
        {

            var _m0 = 3; //Variable.GaussianFromMeanAndVariance(3, 10).Named("m0");
            var _p0 = 1.6 ;//Variable.GammaFromShapeAndScale(1,1).Named("p0");
            var _drawMargin = 0.06276; //Variable.GaussianFromMeanAndVariance(1151, 927*927).Named("Epsilon");
            var _w_k_p = Variable.GaussianFromMeanAndVariance(3, 10).Named("_w_k_p");
            var _w_k_o = -Variable.GaussianFromMeanAndVariance(3, 10).Named("_w_k_o");
            var _w_d_p = -Variable.GaussianFromMeanAndVariance(3, 10).Named("_w_d_p");
            var _w_d_o = Variable.GaussianFromMeanAndVariance(3, 10).Named("_w_d_o");
            var _v_c = Variable.GammaFromShapeAndScale(1,1).Named("_v_c");
            // var _m_q = Variable.GaussianFromMeanAndVariance(1151, 927*927).Named("_m_q");
            // var _v_q = Variable.GammaFromShapeAndScale(0.5,0.02).Named("_v_q");
            
            // double alpha = 2.0, beta = 5.0;
            // double alpha = 1.0, beta = 3.0;
            double alpha = 2, beta = 2;
            var mean = alpha/(alpha+beta);
            var variance = (alpha*beta)/(Math.Pow(alpha + beta, 2)*(alpha+beta+1));   
            // var _unrl = Variable.BetaFromMeanAndVariance(mean,variance);
            // var _rel = Variable.BetaFromMeanAndVariance(mean,variance);
            // var _unrVar = Variable.Bernoulli(_unrl);
            // var _relVar = Variable.Bernoulli(_rel);

            // _m0.AddAttribute(new PointEstimate());
            // _m0.AddAttribute(new ListenToMessages());
            // _p0.AddAttribute(new PointEstimate());
            // _p0.AddAttribute(new ListenToMessages());
            // _drawMargin.AddAttribute(new PointEstimate());
            // _drawMargin.AddAttribute(new ListenToMessages());
            _w_k_p.AddAttribute(new PointEstimate());            
            _w_k_p.AddAttribute(new ListenToMessages());
            _w_k_o.AddAttribute(new PointEstimate());            
            _w_k_o.AddAttribute(new ListenToMessages());
            _w_d_p.AddAttribute(new PointEstimate());            
            _w_d_p.AddAttribute(new ListenToMessages());
            _w_d_o.AddAttribute(new PointEstimate());            
            _w_d_o.AddAttribute(new ListenToMessages());
            _v_c.AddAttribute(new PointEstimate());            
            _v_c.AddAttribute(new ListenToMessages());
            // _m_q.AddAttribute(new PointEstimate());            
            // _m_q.AddAttribute(new ListenToMessages());
            // _v_q.AddAttribute(new PointEstimate());
            // _v_q.AddAttribute(new ListenToMessages());
            
            // _unrl.AddAttribute(new PointEstimate());
            // _unrl.AddAttribute(new ListenToMessages());
            // _rel.AddAttribute(new PointEstimate());
            // _rel.AddAttribute(new ListenToMessages());
            
            // skillsVariable[nPlayers] = Variable.GaussianFromMeanAndPrecision(m0, 1/v0).ForEach(nPlayers); // same prior for each player
            skillsVariable[nPlayers] = Variable.GaussianFromMeanAndPrecision(_m0, _p0).ForEach(nPlayers); // same prior for each player

            // Variable.ConstrainPositive(_drawMargin);
            // Variable.ConstrainPositive(_m0);

            using (var match = Variable.ForEach(nMatches))
            {
                var n = match.Index;
                var playerPerformance = Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch).Named("playerPerformance");

                using (var team = Variable.ForEach(nTeamsPerMatch))
                {
                    using (var player = Variable.ForEach(nPlayersPerTeam))
                    {
                        var playerIndex = matchesVariable[nMatches][nTeamsPerMatch][nPlayersPerTeam];
                        playerIndex.Named("PlayerIndex");

                        //player perfomance in current game
                        var dampedSkill = Variable<double>.Factor(Damp.Backward, skillsVariable[playerIndex], 0.5).Named("Damped skill");
                        playerPerformance[nTeamsPerMatch][nPlayersPerTeam] = (Variable.GaussianFromMeanAndPrecision(dampedSkill, 1).Named("perf from ds")*playerTime[nMatches][nTeamsPerMatch][nPlayersPerTeam].Named("player n Time")).Named("perf n,m")/matchTime[nMatches].Named("match m Time");
                    }
                }



                using (var team = Variable.ForEach(nTeamsPerMatch)) 
                {
                    using (var player = Variable.ForEach(nPlayersPerTeam))
                    {

                        var mean_play_time_team = Variable.Sum(playerTime[nMatches][nTeamsPerMatch].Named("player time m, t"))/matchTime[nMatches].Named("Match time m");
                        mean_play_time_team.Named("Avg play time team");
                        // var b = v_c*playerTime[nMatches][nTeamsPerMatch][nPlayersPerTeam];
                        var b = _v_c*playerTime[nMatches][nTeamsPerMatch][nPlayersPerTeam]; 
                        b.Named("vc * p time");
                        var ptime = playerTime[nMatches][nTeamsPerMatch][player.Index];
                        ptime.Named("p time m, t, n");
                        var pPerf = playerPerformance[nTeamsPerMatch][player.Index];
                        pPerf.Named("player perf t, n");


                        var perf_opposing = Variable.New<double>();
                        perf_opposing.Named("Perfomance opposing team");


                        using(Variable.Case(team.Index, 0))
                        {
                            perf_opposing.SetTo((Variable.Sum(playerTime[nMatches][team.Index+1])/matchTime[nMatches]) * (Variable.Sum(playerPerformance[nTeamsPerMatch])/mean_play_time_team));


                        }

                        using(Variable.Case(team.Index, 1))
                        {
                            perf_opposing.SetTo((Variable.Sum(playerTime[nMatches][team.Index-1])/matchTime[nMatches]) * (Variable.Sum(playerPerformance[nTeamsPerMatch])/mean_play_time_team));
                        }

                        //kill count
                        var karg = _w_k_p*pPerf + _w_k_o*perf_opposing; 
                        // var karg = w_k_p*pPerf + w_k_o*perf_opposing;
                        var kill = (karg)*ptime;
                        kill.Named("kill count m, t, n");
                        killcount[nMatches][team.Index][nPlayersPerTeam] = Variable.Max(0, Variable.GaussianFromMeanAndPrecision(kill, b));

                        //death count
                        // var darg = w_d_p*pPerf + w_d_o*perf_opposing;
                        var darg = _w_d_p*pPerf + _w_d_o*perf_opposing; 
                        var death = (darg)*ptime;
                        death.Named("death count m, t, n");
                        deathcount[nMatches][team.Index][nPlayersPerTeam] = Variable.Max(0, Variable.GaussianFromMeanAndPrecision(death, b));

                        //is the player underperforming? (for quit penalty)
                        // var under = (Variable.GaussianFromMeanAndPrecision(pPerf - perf_opposing - m_q, v_q) < 0);
                        // var under = (Variable.GaussianFromMeanAndPrecision(pPerf - perf_opposing - _m_q, _v_q) < 0); //TODO
                        // under.Named("Under performing");  //TODO

                        //quit penalty
                        // quit[nMatches][nTeamsPerMatch][nPlayersPerTeam] = unrelatedVariable | (relatedVariable & under);
                        // quit[nMatches][nTeamsPerMatch][nPlayersPerTeam] = _unrVar | (_relVar & under); //TODO
                    }
                }


                var diff = (Variable.Sum(playerPerformance[0]).Named("SumTeam1") - Variable.Sum(playerPerformance[1]).Named("SumTeam2"));
                diff.Named("Diff");

                using(Variable.Case(outcomes[n].Named("team1wins"), 0))  
                {
                    // Variable.ConstrainTrue(diff > epsilon);
                    Variable.ConstrainTrue(diff > _drawMargin);
                }

                using(Variable.Case(outcomes[n].Named("team2wins"), 1)) 
                {
                    // Variable.ConstrainTrue(diff < -epsilon); 
                    Variable.ConstrainTrue(diff < -_drawMargin); 
                }

                using (Variable.Case(outcomes[n].Named("draw"), 2)) 
                {
                    // using (Variable.If( diff >= 0))
                    // Variable.ConstrainBetween(diff, -epsilon, epsilon);

                    Variable.ConstrainTrue(diff <= _drawMargin); 
                    Variable.ConstrainTrue(diff >= -_drawMargin); 
                    // Variable.ConstrainBetween(diff, -_drawMargin, _drawMargin);  
                    
                }

                

            }

            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false,
                NumberOfIterations = N_INTERATIONS,
                ShowProgress = true,
                ShowWarnings = true
            };


            // var inferredm0 = inferenceEngine.Infer<Gaussian>(_m0);
            // var inferredp0 = inferenceEngine.Infer<Gamma>(_p0);
            // var inferredDrawMargin = inferenceEngine.Infer<Gaussian>(_drawMargin);
            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(skillsVariable);
            var inferred_w_k_p = inferenceEngine.Infer<Gaussian>(_w_k_p);
            var inferred_w_k_o = inferenceEngine.Infer<Gaussian>(_w_k_o);
            var inferred_w_d_p = inferenceEngine.Infer<Gaussian>(_w_d_p);
            var inferred_w_d_o = inferenceEngine.Infer<Gaussian>(_w_d_o);
            var inferred_v_c = inferenceEngine.Infer<Gamma>(_v_c);
            // var inferred_m_q = inferenceEngine.Infer<Gaussian>(_m_q);
            // var inferred_v_q = inferenceEngine.Infer<Gamma>(_v_q);

            // var inferred_unr = inferenceEngine.Infer<Beta>(_unrl);
            // var inferred_rel = inferenceEngine.Infer<Beta>(_rel);

            // Console.WriteLine("inferredm0: " + inferredm0);
            // Console.WriteLine("inferredp0: " + inferredp0);
            // Console.WriteLine("inferredDrawMargin: " + inferredDrawMargin);

            // this.epsilon = Math.Max(10e-5, inferredDrawMargin.Point);
            // this.m0 = inferredm0.Point;
            // this.v0 = 1/inferredp0.Point;
            this.w_k_p = inferred_w_k_p.Point;
            this.w_k_o = inferred_w_k_o.Point;
            this.w_d_p = inferred_w_d_p.Point;
            this.w_d_o = inferred_w_d_o.Point;
            this.v_c = inferred_v_c.Point;
            // this.m_q = inferred_m_q.Point;
            // this.v_q = inferred_v_q.Point;

            // this.related = inferred_rel.Point;
            // this.unrelated = inferred_unr.Point;

            // Console.WriteLine("Bernoulli variables: (related/unrelated) " + related + " " + unrelated);            

            Console.WriteLine("w_k_p: " + w_k_p);
            Console.WriteLine("w_k_o: " + w_k_o);
            Console.WriteLine("w_d_p: " + w_d_p);
            Console.WriteLine("w_d_o: " + w_d_o);
            Console.WriteLine("v_c: " + v_c);
            // Console.WriteLine("m_q: " + m_q);
            // Console.WriteLine("v_q: " + v_q);

            return inferredSkills;
        }

        Gaussian[] InferSkills()
        {

            var _skillsVariable     = Variable.Array<double>(nPlayers).Named("skills prior"); 
            using(var p = Variable.ForEach(nPlayers))
            {
                _skillsVariable[p.Index] = Variable.GaussianFromMeanAndPrecision(skillsVariable[p.Index], skillsVarianceVariable[p.Index]);
            }
            // _skillsVariable[nPlayers] = Variable.GaussianFromMeanAndPrecision(skillsVariable[nPlayers], 1/v0).ForEach(nPlayers);

            using (var match = Variable.ForEach(nMatches))
            {
                var n = match.Index;
                var playerPerformance = Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch).Named("playerPerformance");

                using (var team = Variable.ForEach(nTeamsPerMatch))
                {
                    using (var player = Variable.ForEach(nPlayersPerTeam))
                    {
                        var playerIndex = matchesVariable[nMatches][nTeamsPerMatch][nPlayersPerTeam];
                        playerIndex.Named("PlayerIndex");

                        // var dampedSkill = Variable<double>.Factor(Damp.Backward, _skillsVariable[playerIndex], 0.5).Named("Damped skill");
                        playerPerformance[nTeamsPerMatch][nPlayersPerTeam] = (Variable.GaussianFromMeanAndPrecision(_skillsVariable[playerIndex], 1).Named("perf from ds")*playerTime[nMatches][nTeamsPerMatch][nPlayersPerTeam].Named("player n Time")).Named("perf n,m")/matchTime[nMatches].Named("match m Time");
                    }
                }



                using (var team = Variable.ForEach(nTeamsPerMatch)) 
                {
                    using (var player = Variable.ForEach(nPlayersPerTeam))
                    {

                        var mean_play_time_team = Variable.Sum(playerTime[nMatches][nTeamsPerMatch].Named("player time m, t"))/matchTime[nMatches].Named("Match time m");
                        mean_play_time_team.Named("Avg play time team");
                        var b = v_c*playerTime[nMatches][nTeamsPerMatch][nPlayersPerTeam];
                        b.Named("vc * p time");
                        var ptime = playerTime[nMatches][nTeamsPerMatch][player.Index];
                        ptime.Named("p time m, t, n");
                        var pPerf = playerPerformance[nTeamsPerMatch][player.Index];
                        pPerf.Named("player perf t, n");


                        var perf_opposing = Variable.New<double>();
                        perf_opposing.Named("Perfomance opposing team");


                        using(Variable.Case(team.Index, 0))
                        {
                            perf_opposing.SetTo((Variable.Sum(playerTime[nMatches][team.Index+1])/matchTime[nMatches]) * (Variable.Sum(playerPerformance[nTeamsPerMatch])/mean_play_time_team));
                        }

                        using(Variable.Case(team.Index, 1))
                        {
                            perf_opposing.SetTo((Variable.Sum(playerTime[nMatches][team.Index-1])/matchTime[nMatches]) * (Variable.Sum(playerPerformance[nTeamsPerMatch])/mean_play_time_team));
                        }

                        //kill count
                        var karg = w_k_p*pPerf + w_k_o*perf_opposing;
                        var kill = (karg)*ptime;
                        kill.Named("kill count m, t, n");
                        killcount[nMatches][team.Index][nPlayersPerTeam] = Variable.Max(0, Variable.GaussianFromMeanAndPrecision(kill, b));

                        //death count
                        var darg = w_d_p*pPerf + w_d_o*perf_opposing;
                        var death = (darg)*ptime;
                        death.Named("death count m, t, n");
                        deathcount[nMatches][team.Index][nPlayersPerTeam] = Variable.Max(0, Variable.GaussianFromMeanAndPrecision(death, b));

                        //is the player underperforming? (for quit penalty) //TODO
                        // var under = (Variable.GaussianFromMeanAndPrecision(pPerf - perf_opposing - m_q, v_q) < 0); //TODO
                        // under.Named("Under performing"); //TODO

                        // //quit penalty //TODO
                        // quit[nMatches][nTeamsPerMatch][nPlayersPerTeam] = unrelatedVariable | (relatedVariable & under); //TODO
                    }
                }


                var diff = (Variable.Sum(playerPerformance[0]).Named("SumTeam1") - Variable.Sum(playerPerformance[1]).Named("SumTeam2"));
                diff.Named("Diff");

                using (Variable.Case(outcomes[n].Named("draw"), 2))
                {
                    // Variable.ConstrainBetween(diff, -epsilon, epsilon);

                    Variable.ConstrainTrue(diff <= epsilon); 
                    Variable.ConstrainTrue(diff >= -epsilon);
                }

                using(Variable.Case(outcomes[n].Named("team1wins"), 0))
                {
                    Variable.ConstrainTrue(diff > epsilon);
                }

                using(Variable.Case(outcomes[n].Named("team2wins"), 1))
                {
                    Variable.ConstrainTrue(diff < -epsilon);
                }

            }

            

            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false,
                NumberOfIterations = N_INTERATIONS,
                ShowProgress = false
            };


            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(_skillsVariable);

            Console.WriteLine("");
            Console.WriteLine("DEBUG prime 3 skill: ");
            Console.WriteLine(inferredSkills[0]);
            Console.WriteLine(inferredSkills[1]);
            Console.WriteLine(inferredSkills[2]);
            Console.WriteLine("");


            return inferredSkills;
        }
        
        /*
        private Gaussian[] inferTau(int nPlayersObserved)
        {
            // skillti ∼ N (skillti , τ 2 (t0 − t))
            var skillsArray = new double[nPlayersObserved];
            for (int i = 0; i < nPlayersObserved; i++ )
            {
                skillsArray[i] = this.BaseSkills[i].GetMean();
            }

            var _tau = Variable.GammaFromShapeAndScale(1,1).Named("_tau");
            _tau.AddAttribute(new PointEstimate());
            _tau.AddAttribute(new ListenToMessages());

            //"reset" of the variable
            var skillsVariable     = Variable.Array<double>(nPlayers).Named("infer tau (skills)"); //TODO rinomina variabili


            // var matches = Variable.Array(Variable.Array(Variable.Array<int>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("matches"); 
            // var skills = Variable.Array<double>(nPlayers); 

            using (Variable.ForEach(nMatches))
            {
                using (Variable.ForEach(nTeamsPerMatch))
                {
                    using (Variable.ForEach(nPlayersPerTeam))
                    {
                        
                        var timepassed = timePassedVariable[nMatches][nTeamsPerMatch][nPlayersPerTeam];

                        var playerIndex = matchesVariable[nMatches][nTeamsPerMatch][nPlayersPerTeam];

                        using(Variable.If(humanPlayersVariable[playerIndex]))
                        {
                            var dampedSkill = Variable<double>.Factor(Damp.Backward, skillsVariable[playerIndex], 0.5);

                            skillsVariable[playerIndex] = Variable.GaussianFromMeanAndPrecision(dampedSkill, _tau*_tau*timepassed);
                        }
                    }


                }
            }

            skillsVariable.ObservedValue = skillsArray;

            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false,
                NumberOfIterations = N_INTERATIONS,
                ShowProgress = false,
                ShowWarnings = false
            };

            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(skillsVariable);
            var inferredtau = inferenceEngine.Infer<Gamma>(_tau);

            Console.WriteLine(" \nInferenza di Tau:");
            Console.WriteLine(inferredtau + $"({1/inferredtau.GetMean()})");
            this.tau = 1/inferredtau.GetMean(); //TODO dovrebbe essere un numero fisso

            return inferredSkills;
        }
       
        private Gaussian[] augmentVarianceTau(int nPlayersObserved)
        {
            // skillti ∼ N (skillti , τ 2 (t0 − t))
            var skillsArray = new double[nPlayersObserved];
            for (int i = 0; i < nPlayersObserved; i++ )
            {
                if (BaseSkills[i].IsUniform())
                {
                    skillsArray[i] = m0;
                }
                else 
                {
                    skillsArray[i] = this.BaseSkills[i].GetMean();
                }
            }

            //"reset" of the variable
            var skillsVariable     = Variable.Array<double>(nPlayers).Named("skills (tau)"); 


            // Inferenza di Gamma
            using (Variable.ForEach(nMatches))
            {
                using (Variable.ForEach(nTeamsPerMatch))
                {
                    using (Variable.ForEach(nPlayersPerTeam))
                    {
                        
                        var timepassed = timePassedVariable[nMatches][nTeamsPerMatch][nPlayersPerTeam];

                        var playerIndex = matchesVariable[nMatches][nTeamsPerMatch][nPlayersPerTeam];
                        using(Variable.If(humanPlayersVariable[playerIndex]))
                        {
                            var dampedSkill = Variable<double>.Factor(Damp.Backward, skillsVariable[playerIndex], 0.5);

                            skillsVariable[playerIndex] = Variable.GaussianFromMeanAndPrecision(dampedSkill, tau*tau*timepassed);
                        }
                    }


                }
            }

            skillsVariable.ObservedValue = skillsArray;

            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false,
                NumberOfIterations = N_INTERATIONS,
                ShowProgress = false,
                ShowWarnings = false
            };

            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(skillsVariable);

            return inferredSkills;
        }
       
        Gaussian[] inferGamma(int nPlayersObserved)
        {

            var skillsArray = new double[nPlayersObserved];
            for (int i = 0; i < nPlayersObserved; i++ )
            {
                skillsArray[i] = this.BaseSkills[i].GetMean();
            }

            

            var _gamma = Variable.GammaFromShapeAndScale(1,1).Named("_gamma");
            // var _gamma = Variable.GaussianFromMeanAndPrecision(0, 100);
            _gamma.AddAttribute(new PointEstimate());
            _gamma.AddAttribute(new ListenToMessages());

            //"reset" of the variable
            var skillsVariable     = Variable.Array<double>(nPlayers).Named("infer gamma (skills)"); 

            // var matches = Variable.Array(Variable.Array(Variable.Array<int>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("matches"); 
            // var skills = Variable.Array<double>(nPlayers); 

            // Inferenza di Gamma
            using (Variable.ForEach(nMatches))
            {
                using (Variable.ForEach(nTeamsPerMatch))
                {
                    using (Variable.ForEach(nPlayersPerTeam))
                    {
                        
                        var playerIndex = matchesVariable[nMatches][nTeamsPerMatch][nPlayersPerTeam];
                        using(Variable.If(humanPlayersVariable[playerIndex]))
                        {
                            var dampedSkill = Variable<double>.Factor(Damp.Backward, skillsVariable[playerIndex], 0.5);

                            skillsVariable[playerIndex] = Variable.GaussianFromMeanAndPrecision(dampedSkill 
                                + experienceOffsetVariable[Variable<double>.Min(199, experienceVariable[playerIndex][nMatches])], _gamma);
                                // + experienceOffsetVariable[Variable<double>.Min(199, experienceVariable[playerIndex][nMatches])], _gamma*_gamma);
                        }
                    }


                }
            }

            // Variable.ConstrainTrue( _gamma/tau == 10e5 );

            skillsVariable.ObservedValue = skillsArray;

            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false,
                NumberOfIterations = N_INTERATIONS,
                ShowProgress = false,
                ShowWarnings = false
            };

            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(skillsVariable);
            var inferredgamma = inferenceEngine.Infer<Gamma>(_gamma);

            Console.WriteLine(" \nInferenza di gamma:");
            Console.WriteLine(inferredgamma + $"({1/inferredgamma.GetMean()})");
            this.gamma = 1/inferredgamma.GetMean();
            this.tau = gamma/10e5;

            return inferredSkills;
        }
        
        Gaussian[] augmentVarianceGamma(int nPlayersObserved)
        {

            var skillsArray = new double[nPlayersObserved];
            for (int i = 0; i < nPlayersObserved; i++ )
            {
                skillsArray[i] = this.BaseSkills[i].GetMean();
            }

            //"reset" of the variable
            var skillsVariable     = Variable.Array<double>(nPlayers).Named("skills (gamma)"); 

            using (Variable.ForEach(nMatches))
            {
                using (Variable.ForEach(nTeamsPerMatch))
                {
                    using (Variable.ForEach(nPlayersPerTeam))
                    {
                        var playerIndex = matchesVariable[nMatches][nTeamsPerMatch][nPlayersPerTeam];
                        using(Variable.If(humanPlayersVariable[playerIndex]))
                        {
                            var dampedSkill = Variable<double>.Factor(Damp.Backward, skillsVariable[playerIndex], 0.5);
                        

                            skillsVariable[playerIndex] = Variable.GaussianFromMeanAndPrecision(dampedSkill 
                                + experienceOffsetVariable[Variable<double>.Min(199, experienceVariable[playerIndex][nMatches])], gamma*gamma);
                        }
                    }


                }
            }

            skillsVariable.ObservedValue = skillsArray;

            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false,
                NumberOfIterations = N_INTERATIONS,
                ShowProgress = false,
                ShowWarnings = false
            };

            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(skillsVariable);

            return inferredSkills;
        
        }

        */

        public Gaussian[] predictOutcomes(double[] observedSkills)
        {
            skillsVariable.ObservedValue = observedSkills; 

            var _outcomes = Variable.Array<double>(nMatches);
            _outcomes.AddAttribute(new PointEstimate());
            _outcomes.AddAttribute(new ListenToMessages());

            using (var match = Variable.ForEach(nMatches))
            {
                var n = match.Index;
                var playerPerformance = Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch);

                using (var team = Variable.ForEach(nTeamsPerMatch))
                {
                    using (var player = Variable.ForEach(nPlayersPerTeam))
                    {
                        var playerIndex = matchesVariable[nMatches][nTeamsPerMatch][nPlayersPerTeam];
                        playerIndex.Named("PlayerIndex");

                        //player perfomance in current game
                        // var dampedSkill = Variable<double>.Factor(Damp.Backward, skillsVariable[playerIndex], 0.5);
                        playerPerformance[nTeamsPerMatch][nPlayersPerTeam] = (Variable.GaussianFromMeanAndPrecision(skillsVariable[playerIndex], 1)*playerTime[nMatches][nTeamsPerMatch][nPlayersPerTeam])/matchTime[nMatches];
                        // playerPerformance[nTeamsPerMatch][nPlayersPerTeam] = (Variable.GaussianFromMeanAndPrecision(dampedSkill, 1)*playerTime[nMatches][nTeamsPerMatch][nPlayersPerTeam])/matchTime[nMatches];
                    }
                }



             
                var diff = (Variable.Sum(playerPerformance[0]) - Variable.Sum(playerPerformance[1]));
                _outcomes[n] = diff; 
                // using(Variable.If(diff > epsilon))  //team 1 wins
                // {
                //     _outcomes[n] = 0.0;
                // }

                // using(Variable.If(diff < -epsilon)) //team 2 wins
                // {
                //     _outcomes[n] = 1.0;
                // }

                // using (Variable.If((diff <= epsilon) & (diff >= -epsilon))) //draw
                // {
                //     _outcomes[n] = 2.0;
                // }

                

            }

            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false,
                NumberOfIterations = N_INTERATIONS,
                ShowProgress = false,
                ShowWarnings = false
            };


            var inferredOutcomes = inferenceEngine.Infer<Gaussian[]>(_outcomes);

            return inferredOutcomes;
        
        }

        public double predictAccuracy(List<Match> match2predictList)
        {

            
            

            List<string> skill_players = players(match2predictList);
            var skills = new double[skill_players.Count()];
            int knownSkill = 0;
            for (int i = 0; i < skills.Count(); i++)
            {
                if (!BaseSkills[i].IsUniform()){
                    knownSkill += 1;
                    skills[i] = BaseSkills[i].GetMean();
                }
                else
                    skills[i] = m0;
            }

            Console.WriteLine("===> DEBUG le skill note date dalle precedenti partite sono: " + knownSkill + " su " + skills.Count() + " settate");

            var _players = players(match2predictList);

            int[] outcomes = new int[match2predictList.Count()];
            var match2predict = new int[match2predictList.Count()][][];

            for(int i = 0; i < match2predictList.Count(); i++){

                match2predict[i] = new int[2][];
                match2predict[i][0] = new int[match2predictList[i].team1.nPlayers()];
                match2predict[i][1] = new int[match2predictList[i].team2.nPlayers()];

                for (int k = 0; k < match2predictList[i].team1.nPlayers(); k++){
                    var player = _players.FindIndex(t => t == match2predictList[i].team1.teammates[k].tag);

                    match2predict[i][0][k] = player;

                    

                }
                for (int k = 0; k < match2predictList[i].team2.nPlayers(); k++){
                    var player = _players.FindIndex(t => t == match2predictList[i].team2.teammates[k].tag);

                    match2predict[i][1][k] = player;
                }


                if (match2predictList[i].isDraw()){
                    outcomes[i] = 2;
                }
                else if (match2predictList[i].isTeam1Winner()){
                    outcomes[i] = 0; 
                }
                else
                    outcomes[i] = 1;
            }



            var accuracy = 0.0;

        /*
            var correctPredictions = 0.0;

            int predT1win = 0;
            int predT2win = 0;
            int predDraw = 0;
            int predDrawTotal = 0;
            int predT1winTotal = 0;
            int predT2winTotal = 0;
            int unknownteams1 = 0;
            int unknownteams2 = 0;
            int unknownmatch = 0;

            int winT1 = 0; int winT2 = 0; int draw = 0;

            for (int m = 0; m < match2predict.Count(); m++)
            {
                double expectedPerfT1 = 0.0;
                double expectedPerfT2 = 0.0;

                bool unknownteam1 = true;
                bool unknownteam2 = true;

                for(int p = 0; p < match2predict[m][0].Count(); p++)
                {
                    var pl = _players[match2predict[m][0][p]];
                    var index = skill_players.FindIndex(0, skill_players.Count(), tag => tag == pl);
                    // Console.WriteLine("DEBUG " + skills[index]);
                    
                    if (index > 0){
                        var _sk = new Gaussian(skills[index].GetMean(), 1).GetMean();
                        expectedPerfT1 += _sk;
                        unknownteam1 = _sk == 3;
                    }
                    else {
                        expectedPerfT1 += new Gaussian(m0, 1).GetMean();
                    }
                }

                for(int p = 0; p < match2predict[m][1].Count(); p++)
                {
                    var pl = _players[match2predict[m][1][p]];
                    var index = skill_players.FindIndex(0, skill_players.Count(), tag => tag == pl);
                    // Console.WriteLine("DEBUG " + skills[index]);
                    
                    
                    if (index > 0){
                        var _sk = new Gaussian(skills[index].GetMean(), 1).GetMean();
                        expectedPerfT2 += _sk;
                        unknownteam2 = _sk == 3;
                    }
                    else {
                        expectedPerfT2 += new Gaussian(m0, 1).GetMean();
                    }
                }


                if (unknownteam1){
                    unknownteams1 += 1;
                }
                if (unknownteam2){
                    unknownteams2 += 1;
                }

                if (unknownteam1 && unknownteam2){
                    unknownmatch += 1;
                    continue;
                }

                //DEBUG 
                if (m%100 == 0){
                    Console.WriteLine("Perfomance t1: " + expectedPerfT1 + " Performance t2: " + expectedPerfT2);
                }

                if (outcomes[m] == 0 )
                    winT1 += 1;

                if (outcomes[m] == 1 )
                    winT2 += 1;

                if (outcomes[m] == 2 )
                    draw += 1;

                var diff = expectedPerfT1 - expectedPerfT2; 

                

                if ( diff > this.epsilon && outcomes[m] == 0)
                {
                    correctPredictions += 1;
                    predT1win += 1;
                }
                if ( diff < -this.epsilon && outcomes[m] == 1)
                {
                    correctPredictions += 1;
                    predT2win += 1;
                }
                if (diff <= this.epsilon && diff >= -this.epsilon && outcomes[m] == 2)
                {
                    correctPredictions += 1;
                    predDraw += 1;
                }

                if ( diff > this.epsilon)
                {
                    predT1winTotal += 1;
                }
                if ( diff < -this.epsilon)
                {
                    predT2winTotal += 1;
                }
                if (diff <= this.epsilon && diff >= -this.epsilon)
                {
                    predDrawTotal += 1;
                }

                

            }
            
            Console.WriteLine($"DEBUG {predT1win}/{winT1} {predT2win}/{winT2} {predDraw}/{draw}");
            Console.WriteLine($"DEBUG total team1 win: {predT1winTotal} team2 win: {predT2winTotal} draw: {predDrawTotal}");
            Console.WriteLine($"DEBUG unknown teams: {unknownteams1 + unknownteams2} in the same game: { unknownmatch }");
            accuracy = correctPredictions/(winT1 + winT2 + draw);
            // accuracy = correctPredictions/outcomes.Count();
        */
            
            var predictedOutcomes = predictOutcomes(skills);

            for (int i = 0; i < outcomes.Length; i++){
                if (outcomes[i] == 0 && (predictedOutcomes[i].Point > epsilon)){
                    accuracy += 1;
                }

                if (outcomes[i] == 1 && (predictedOutcomes[i].Point < -epsilon)){
                    accuracy += 1;
                }

                if (outcomes[i] == 2 && (predictedOutcomes[i].Point >= -epsilon && predictedOutcomes[i].Point <= epsilon)){
                    accuracy += 1;
                }
            }   

            accuracy = accuracy/(outcomes.Length);

            return accuracy*100;
        } 


    }
}
