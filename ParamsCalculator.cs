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
         
        bool ParameterComputed;
        private bool testAccuracy;
        double m0;
        double v0;
        double gamma;
        double tau;
        double beta;
        double epsilon;

	    double w_k_p;
        double w_k_o;
        double w_d_p;
        double w_d_o;
        double v_c;
        double m_q;
        double v_q;
        double unrelated;
        double related;
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

        VariableArray<VariableArray<VariableArray<bool>, bool[][]>, bool[][][]> quit; 

        VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> timePassedVariable;
        VariableArray<double> experienceOffsetVariable;
        VariableArray<VariableArray<int>, int[][]> experienceVariable;
        VariableArray<int> humanPlayersVariable;
        Gaussian[] BaseSkills;

        const int N_INTERATIONS = 1000;


        

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
            double[][][] timePassedObserved, int[][] experienceObserved, int[] humanPlayersObserved)
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

            quit       = Variable.Array(Variable.Array(Variable.Array<bool>   (nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("quit");

            timePassedVariable = Variable.Array(Variable.Array(Variable.Array<double>   (nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("time passed");
            experienceVariable = Variable.Array(Variable.Array<int>(nMatches), nPlayers);
            experienceOffsetVariable = Variable.Array<double>(new Range(200));
            humanPlayersVariable = Variable.Array<int>(nPlayers);

            relatedVariable = Variable.Bernoulli(related);
            unrelatedVariable = Variable.Bernoulli(unrelated);

            matchesVariable.ObservedValue = matchData;
            outcomes.ObservedValue = outcomesObserved;
            playersInTeam.ObservedValue = nPlayersPerTeamObserved;
            matchTime.ObservedValue = matchTimeObserved;
            playerTime.ObservedValue = playersTimeObserved;
            killcount.ObservedValue = killCountObserved;
            deathcount.ObservedValue = deathCountObserved;

            quit.ObservedValue = quitObserved; 

            timePassedVariable.ObservedValue = timePassedObserved;
            experienceVariable.ObservedValue = experienceObserved;
            humanPlayersVariable.ObservedValue = humanPlayersObserved;

            
            var experienceOffset = new double[200];
            var exp = 0.01;
            for (int i = 0; i < 200; i += 1 )
            {
                experienceOffset[i] = exp;
                exp += 0.01;
            }

            experienceOffsetVariable.ObservedValue = experienceOffset;

            if (BaseSkills != null)
            {

                setSkills();
            }

        }

        private void setSkills()
        {
            
            skillsVariable = Variable.Array<double>(nPlayers);
            skillsVarianceVariable = Variable.Array<double>(nPlayers);
            
            int i = 0, m0v0 = 0;
            int _nPlayers = players(null).Count(); 
            var skillsArray = new double[_nPlayers];
            var varianceArray = new double[_nPlayers];
            
            foreach (Gaussian skill in BaseSkills)
            {

                if (!skill.IsUniform())
                {
                    try
                    {
                        skillsArray[i] = skill.GetMean();
                        varianceArray[i] = skill.GetVariance();
                    } catch (Microsoft.ML.Probabilistic.Distributions.ImproperDistributionException e)
                    {
                        Console.WriteLine("eccezione! \n" + e.StackTrace);
                    }

                }
                else
                {
                    skillsArray[i] = m0;
                    varianceArray[i] = v0;
                    m0v0 += 1;


                }

                i += 1;
            }
            skillsVariable.ObservedValue = skillsArray;
            skillsVarianceVariable.ObservedValue = varianceArray;
        }

        public Gaussian[] ComputeSkills(double[][][] timepassed, int[][] playersExperience, int predictOffset, out int correct)
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
            int[] humanPlayers = new int[nPlayers]; //1: umano; 0: bot

            var gameInPlayersList = new List<List<int>>();
            for (int i = 0; i < nPlayers; i++)
            {
                gameInPlayersList.Add(new List<int>());
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


                for (int k = 0; k < matches[i].team1.nPlayers(); k++){
                    var player = _players.FindIndex(t => t == matches[i].team1.teammates[k].tag);
                    playersInGame[i][0][k] = player;
                    humanPlayers[player] = matches[i].team1.teammates[k].bot ? 0 : 1; 
                    gameInPlayersList[player].Add(i);

                    matchData[i][0][k] = player;

                    playersTime[i][0][k] = matches[i].team1.teammates[k].secondsPlayed;

                    killcount[i][0][k] = matches[i].team1.teammates[k].killcount;
                    deathcount[i][0][k] = matches[i].team1.teammates[k].deathcount;

                    quit[i][0][k] = !matches[i].team1.teammates[k].quit;

                }
                for (int k = 0; k < matches[i].team2.nPlayers(); k++){
                    var player = _players.FindIndex(t => t == matches[i].team2.teammates[k].tag);
                    playersInGame[i][1][k] = player;
                    humanPlayers[player] = matches[i].team2.teammates[k].bot ? 0 : 1;
                    gameInPlayersList[player].Add(i);


                    matchData[i][1][k] = player;

                    playersTime[i][1][k] = matches[i].team2.teammates[k].secondsPlayed;

                    killcount[i][1][k] = matches[i].team2.teammates[k].killcount;
                    deathcount[i][1][k] = matches[i].team2.teammates[k].deathcount;

                    quit[i][1][k] = !matches[i].team2.teammates[k].quit;

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
            correct = 0;
            if (this.ParameterComputed)
            {
                // Console.WriteLine("Calcolo skills (parametri già calcolati)");
                correct = predictAccuracy(matches, predictOffset);
                // Console.WriteLine("Accuracy: " + ((double)correct/(matches.Count() - predictOffset))); 

                skills = augmentVarianceTau(nPlayers); 
                SetBaseSkills(skills); setSkills();

                skills = InferSkills();
                SetBaseSkills(skills); setSkills();

                skills = augmentVarianceGamma(nPlayers);
            }
            else{

                ParameterComputed = true;
                skills = InferSkillsAndParameters(matches.Count(), nPlayers);
                SetBaseSkills(skills); setSkills(); 
                skills = inferGamma(nPlayers);
            }

            return skills;
        }


        Gaussian[] InferSkillsAndParameters(int nMatchesObserved, int nPlayersObserved)
        {

        var _m0 = Variable.GaussianFromMeanAndVariance(1151, 927*927).Named("m0"); //generico
        // var _m0 = Variable.GaussianFromMeanAndVariance(1493, 1228*1228).Named("m0"); //oddball
        // var _m0 = Variable.GaussianFromMeanAndVariance(670, 688*688).Named("m0"); //one flag ctf
        // var _m0 = Variable.GaussianFromMeanAndVariance(1800, 3500*3500).Named("m0"); //CTF
        // var _m0 = Variable.GaussianFromMeanAndVariance(1800, 2700*2700).Named("m0"); //slayer
        // var _m0 = Variable.GaussianFromMeanAndVariance(1267, 822*822).Named("m0"); //koth
        // var _m0 = Variable.GaussianFromMeanAndVariance(1400, 1500*1500).Named("m0"); //strongholds

	    var _p0 = Variable.GammaFromShapeAndScale(0.5,0.02).Named("p0");
            var _drawMargin = Variable.GaussianFromMeanAndVariance(1258, 900*900).Named("Epsilon");
            var _w_k_p = Variable.GaussianFromMeanAndVariance(1258, 900*900).Named("_w_k_p");
            var _w_k_o = -Variable.GaussianFromMeanAndVariance(1258, 900*900).Named("_w_k_o");
            var _w_d_p = -Variable.GaussianFromMeanAndVariance(1258, 900*900).Named("_w_d_p");
            var _w_d_o = Variable.GaussianFromMeanAndVariance(1258, 900*900).Named("_w_d_o");
            var _v_c = Variable.GammaFromShapeAndScale(0.5,0.02).Named("_v_c");
            var _m_q = Variable.GaussianFromMeanAndVariance(1258, 900*900).Named("_m_q");
            var _v_q = Variable.GammaFromShapeAndScale(0.5,0.02).Named("_v_q");

            double alpha = 2, beta = 2;
            var mean = alpha/(alpha+beta);
            var variance = (alpha*beta)/(Math.Pow(alpha + beta, 2)*(alpha+beta+1));
            var _unrl = Variable.BetaFromMeanAndVariance(mean,variance);
            var _rel = Variable.BetaFromMeanAndVariance(mean,variance);
            var _unrVar = Variable.Bernoulli(_unrl);
            var _relVar = Variable.Bernoulli(_rel);

            _m0.AddAttribute(new PointEstimate());
            _m0.AddAttribute(new ListenToMessages());
            _p0.AddAttribute(new PointEstimate());
            _p0.AddAttribute(new ListenToMessages());
            _drawMargin.AddAttribute(new PointEstimate());
            _drawMargin.AddAttribute(new ListenToMessages());
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
            _m_q.AddAttribute(new PointEstimate());
            _m_q.AddAttribute(new ListenToMessages());
            _v_q.AddAttribute(new PointEstimate());
            _v_q.AddAttribute(new ListenToMessages());

            _unrl.AddAttribute(new PointEstimate());
            _unrl.AddAttribute(new ListenToMessages());
            _rel.AddAttribute(new PointEstimate());
            _rel.AddAttribute(new ListenToMessages());

            skillsVariable[nPlayers] = Variable.GaussianFromMeanAndPrecision(_m0, _p0).ForEach(nPlayers); // same prior for each player


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
                        var dampedSkill = Variable<double>.Factor(Damp.Backward, skillsVariable[playerIndex], 0.50).Named("Damped skill");
                        playerPerformance[nTeamsPerMatch][nPlayersPerTeam] = (Variable.GaussianFromMeanAndPrecision(dampedSkill, 1).Named("perf from ds")*playerTime[nMatches][nTeamsPerMatch][nPlayersPerTeam].Named("player n Time")).Named("perf n,m")/matchTime[nMatches].Named("match m Time");
                    }
                }



                using (var team = Variable.ForEach(nTeamsPerMatch))
                {
                    using (var player = Variable.ForEach(nPlayersPerTeam))
                    {

                        var mean_play_time_team = Variable.Sum(playerTime[nMatches][nTeamsPerMatch].Named("player time m, t"))/matchTime[nMatches].Named("Match time m");
                        mean_play_time_team.Named("Avg play time team");
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
                        var kill = (karg)*ptime;
                        kill.Named("kill count m, t, n");
                        killcount[nMatches][team.Index][nPlayersPerTeam] = Variable.Max(0, Variable.GaussianFromMeanAndPrecision(kill, b));

                        //death count
                        
                        var darg = _w_d_p*pPerf + _w_d_o*perf_opposing;
                        var death = (darg)*ptime;
                        death.Named("death count m, t, n");
                        deathcount[nMatches][team.Index][nPlayersPerTeam] = Variable.Max(0, Variable.GaussianFromMeanAndPrecision(death, b));

                        //is the player underperforming? (for quit penalty)
                        var under = (Variable.GaussianFromMeanAndPrecision(pPerf - perf_opposing - _m_q, _v_q) < 0);
                        under.Named("Under performing"); 

                        //quit penalty
                        quit[nMatches][nTeamsPerMatch][nPlayersPerTeam] = _unrVar | (_relVar & under);
                    }
                }


                var diff = (Variable.Sum(playerPerformance[0]).Named("SumTeam1") - Variable.Sum(playerPerformance[1]).Named("SumTeam2"));
                diff.Named("Diff");

                using(Variable.Case(outcomes[n].Named("team1wins"), 0))
                {
                    Variable.ConstrainTrue(diff > _drawMargin);
                }

                using(Variable.Case(outcomes[n].Named("team2wins"), 1))
                {
                    Variable.ConstrainTrue(diff < -_drawMargin);
                }

                using (Variable.Case(outcomes[n].Named("draw"), 2))
                {

                    Variable.ConstrainTrue(diff <= _drawMargin);
                    Variable.ConstrainTrue(diff >= -_drawMargin);

                }



            }

            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false,
                NumberOfIterations = N_INTERATIONS,
                ShowProgress = true,
                ShowWarnings = true
            };


            var inferredm0 = inferenceEngine.Infer<Gaussian>(_m0);
            var inferredp0 = inferenceEngine.Infer<Gamma>(_p0);
            var inferredDrawMargin = inferenceEngine.Infer<Gaussian>(_drawMargin);
            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(skillsVariable);
            var inferred_w_k_p = inferenceEngine.Infer<Gaussian>(_w_k_p);
            var inferred_w_k_o = inferenceEngine.Infer<Gaussian>(_w_k_o);
            var inferred_w_d_p = inferenceEngine.Infer<Gaussian>(_w_d_p);
            var inferred_w_d_o = inferenceEngine.Infer<Gaussian>(_w_d_o);
            var inferred_v_c = inferenceEngine.Infer<Gamma>(_v_c);
            var inferred_m_q = inferenceEngine.Infer<Gaussian>(_m_q);
            var inferred_v_q = inferenceEngine.Infer<Gamma>(_v_q);

            var inferred_unr = inferenceEngine.Infer<Beta>(_unrl);
            var inferred_rel = inferenceEngine.Infer<Beta>(_rel);

            Console.WriteLine("inferredm0: " + inferredm0);
            Console.WriteLine("inferredp0: " + inferredp0);
            Console.WriteLine("inferredDrawMargin: " + inferredDrawMargin);

            this.epsilon = Math.Max(10e-5, inferredDrawMargin.Point);
            this.m0 = inferredm0.Point;
            this.v0 = 1/inferredp0.Point;
            this.w_k_p = inferred_w_k_p.Point;
            this.w_k_o = inferred_w_k_o.Point;
            this.w_d_p = inferred_w_d_p.Point;
            this.w_d_o = inferred_w_d_o.Point;
            this.v_c = inferred_v_c.Point;
            this.m_q = inferred_m_q.Point;
            this.v_q = inferred_v_q.Point;

            this.related = inferred_rel.Point;
            this.unrelated = inferred_unr.Point;

            Console.WriteLine("Bernoulli variables: (related/unrelated) " + related + " " + unrelated);

            Console.WriteLine("w_k_p: " + w_k_p);
            Console.WriteLine("w_k_o: " + w_k_o);
            Console.WriteLine("w_d_p: " + w_d_p);
            Console.WriteLine("w_d_o: " + w_d_o);
            Console.WriteLine("v_c: " + v_c);
            Console.WriteLine("m_q: " + m_q);
            Console.WriteLine("v_q: " + v_q);

            return inferredSkills;
        }

        Gaussian[] InferSkills()
        {

            var _skillsVariable     = Variable.Array<double>(nPlayers).Named("skills prior");
            using(var p = Variable.ForEach(nPlayers))
            {
                _skillsVariable[p.Index] = Variable.GaussianFromMeanAndPrecision(skillsVariable[p.Index], skillsVarianceVariable[p.Index]);
            }

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
                        var under = (Variable.GaussianFromMeanAndPrecision(pPerf - perf_opposing - m_q, v_q) < 0);
                        under.Named("Under performing");

                        //quit penalty
                        quit[nMatches][nTeamsPerMatch][nPlayersPerTeam] = unrelatedVariable | (relatedVariable & under);
                    }
                }


                var diff = (Variable.Sum(playerPerformance[0]).Named("SumTeam1") - Variable.Sum(playerPerformance[1]).Named("SumTeam2"));
                diff.Named("Diff");

                using (Variable.Case(outcomes[n].Named("draw"), 2))
                {
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

        private Gaussian[] augmentVarianceTau(int nPlayersObserved)
        {
            

            var _skillsVariable     = Variable.Array<double>(nPlayers);
            using(var p = Variable.ForEach(nPlayers))
            {
                _skillsVariable[p.Index] = Variable.GaussianFromMeanAndPrecision(skillsVariable[p.Index], skillsVarianceVariable[p.Index]);
            }

            // Inferenza di Gamma
            using (Variable.ForEach(nMatches))
            {
                using (Variable.ForEach(nTeamsPerMatch))
                {
                    using (Variable.ForEach(nPlayersPerTeam))
                    {

                        var timepassed = timePassedVariable[nMatches][nTeamsPerMatch][nPlayersPerTeam];

                        var playerIndex = matchesVariable[nMatches][nTeamsPerMatch][nPlayersPerTeam];
                        using(Variable.Case(humanPlayersVariable[playerIndex], 1))
                        {
                            var dampedSkill = Variable<double>.Factor(Damp.Backward, _skillsVariable[playerIndex], 0.50);

                            skillsVariable[playerIndex] = Variable.GaussianFromMeanAndPrecision(dampedSkill, tau*tau*timepassed);
                        }
                    }


                }
            }

            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false,
                NumberOfIterations = N_INTERATIONS,
                ShowProgress = false,
                ShowWarnings = false
            };

            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(_skillsVariable);

            return inferredSkills;
        }

        Gaussian[] inferGamma(int nPlayersObserved)
        {

            
            var _skillsVariable     = Variable.Array<double>(nPlayers);
            _skillsVariable[nPlayers] = skillsVariable[nPlayers];
            
            var _gamma = Variable.GammaFromShapeAndScale(1,1).Named("_gamma");
            _gamma.AddAttribute(new PointEstimate());
            _gamma.AddAttribute(new ListenToMessages());

            
            // Inferenza di Gamma
            using (Variable.ForEach(nMatches))
            {
                using (Variable.ForEach(nTeamsPerMatch))
                {
                    using (Variable.ForEach(nPlayersPerTeam))
                    {

                        var playerIndex = matchesVariable[nMatches][nTeamsPerMatch][nPlayersPerTeam];
                        var dampedSkill = Variable<double>.Factor(Damp.Backward, skillsVariable[playerIndex], 0.50);
                        
                        skillsVariable[playerIndex] = (Variable.GaussianFromMeanAndPrecision(dampedSkill
                                + experienceOffsetVariable[Variable<double>.Min(199, experienceVariable[playerIndex][nMatches])], _gamma*_gamma));
                        
                    }


                }
            }

            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false,
                NumberOfIterations = N_INTERATIONS,
                ShowProgress = true,
                ShowWarnings = false
            };

            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(_skillsVariable);
            var inferredgamma = inferenceEngine.Infer<Gamma>(_gamma);

            Console.WriteLine(" \nInferenza di gamma:");
            Console.WriteLine(inferredgamma + $"({1/inferredgamma.Point})");
            this.gamma = 1/inferredgamma.Point;
            this.tau = gamma/10e5;

            return inferredSkills;
        }

        Gaussian[] augmentVarianceGamma(int nPlayersObserved)
        {

            var _skillsVariable     = Variable.Array<double>(nPlayers);
            _skillsVariable[nPlayers] = skillsVariable[nPlayers];

            using (Variable.ForEach(nMatches))
            {
                using (Variable.ForEach(nTeamsPerMatch))
                {
                    using (Variable.ForEach(nPlayersPerTeam))
                    {
                        var playerIndex = matchesVariable[nMatches][nTeamsPerMatch][nPlayersPerTeam];
                        
                        var dampedSkill = Variable<double>.Factor(Damp.Backward, _skillsVariable[playerIndex], 0.50);

                        using(Variable.Case(humanPlayersVariable[playerIndex], 1))
                        {
                            


                            skillsVariable[playerIndex] = Variable.GaussianFromMeanAndPrecision(dampedSkill
                                + experienceOffsetVariable[Variable<double>.Min(199, experienceVariable[playerIndex][nMatches])], gamma);
                        }


                    
                    }


                }
            }


            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false,
                NumberOfIterations = N_INTERATIONS,
                ShowProgress = false,
                ShowWarnings = false
            };

            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(_skillsVariable);

            return inferredSkills;

        }

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
                        playerPerformance[nTeamsPerMatch][nPlayersPerTeam] = (Variable.GaussianFromMeanAndPrecision(skillsVariable[playerIndex], 1)*playerTime[nMatches][nTeamsPerMatch][nPlayersPerTeam])/matchTime[nMatches];
                    }
                }




                var diff = (Variable.Sum(playerPerformance[0]) - Variable.Sum(playerPerformance[1]));
                _outcomes[n] = diff;


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

        public int predictAccuracy(List<Match> match2predictList, int offset)
        {




            List<string> skill_players = players(match2predictList);
            var skills = new double[skill_players.Count()];
            for (int i = 0; i < skills.Count(); i++)
            {
                if (!BaseSkills[i].IsUniform()){
                    skills[i] = BaseSkills[i].GetMean();
                }
                else
                    skills[i] = m0;
            }

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

            double predictedDraws = 0, predictedWinT1 = 0, predictedWinT2 = 0, winT1 = 0, winT2 = 0, draw = 0,
                correctPredictedWinT1 = 0, correctPredictedWinT2 = 0;
            
            var known_player_per_match = new int[match2predictList.Count()];
            var countPlayersPerMatch = new double[] {0.0, 0.0, 0.0, 0.0};

            int matchIndex = 0;
            foreach (var match in match2predictList)
            {
                if (matchIndex < offset)
                {
                    matchIndex += 1;
                    continue;
                }

                int known_players = 0;
                foreach (var player in match.team1.teammates)
                {
                    var pIndex = skill_players.FindIndex( p => p.SequenceEqual(player.tag));
                    if (pIndex >= 0) 
                    {
                        if (!BaseSkills[pIndex].IsUniform())
                            known_players += 1;
                    }
                    
                }
                foreach (var player in match.team2.teammates)
                {
                    var pIndex = skill_players.FindIndex( p => p.SequenceEqual(player.tag));
                    if (pIndex >= 0)
                    {
                        if (!BaseSkills[pIndex].IsUniform())
                            known_players += 1;
                    }
                }
                known_player_per_match[matchIndex] = known_players; 
                switch(known_players)
                {
                    case 0:
                        break;
                    case 1:
                        countPlayersPerMatch[0] += 1;
                        
                        break;
                    case 2:
                        countPlayersPerMatch[1] += 1;
                        countPlayersPerMatch[0] += 1;
                        
                        break;
                    case 3:
                        countPlayersPerMatch[2] += 1;
                        countPlayersPerMatch[0] += 1;
                        countPlayersPerMatch[1] += 1;
                        break;
                    case 4:
                        countPlayersPerMatch[3] += 1;
                        countPlayersPerMatch[0] += 1;
                        countPlayersPerMatch[1] += 1;
                        countPlayersPerMatch[2] += 1;
                        break;
                    default:
                        countPlayersPerMatch[3] += 1;
                        countPlayersPerMatch[0] += 1;
                        countPlayersPerMatch[1] += 1;
                        countPlayersPerMatch[2] += 1;
                        break;
                }
                matchIndex += 1; 
            }
          
            var predictedOutcomes = predictOutcomes(skills);

            double pred1Player = 0;
            double pred2Player = 0;
            double pred3Player = 0;
            double pred4OrMorePlayer = 0;

            for (int i = offset; i < outcomes.Length; i++){

                if (predictedOutcomes[i].Point > epsilon)
                {
                    predictedWinT1 += 1;
                    
                }

                if (predictedOutcomes[i].Point < -epsilon)
                {
                    predictedWinT2 += 1;
                    
                }

                if ((predictedOutcomes[i].Point >= -epsilon && predictedOutcomes[i].Point <= epsilon))
                {
                    predictedDraws += 1;
                    
                }

                if (outcomes[i] == 0)
                    winT1 += 1;
                if (outcomes[i] == 1)
                    winT2 += 1;
                if (outcomes[i] == 2)
                    draw += 1;

                if (outcomes[i] == 0 && (predictedOutcomes[i].Point > epsilon)){
                    correctPredictedWinT1 += 1;
                    accuracy += 1;
                    switch(known_player_per_match[i])
                    {
                        case 0:
                            break;
                        case 1:
                            pred1Player += 1;
                            
                            break;
                        case 2:
                            pred2Player += 1;
                            pred1Player += 1;
                            
                            break;
                        case 3:
                            pred3Player += 1;
                            pred1Player += 1;
                            pred2Player += 1;
                            
                            break;
                        case 4:
                            pred4OrMorePlayer += 1;
                            pred1Player += 1;
                            pred2Player += 1;
                            pred3Player += 1;
                            
                            break;
                        default:
                            pred4OrMorePlayer += 1;
                            pred1Player += 1;
                            pred2Player += 1;
                            pred3Player += 1;
                            
                            break;
                    }
                } 

                if (outcomes[i] == 1 && (predictedOutcomes[i].Point < -epsilon)){
                    correctPredictedWinT2 += 1;
                    accuracy += 1;
                    switch(known_player_per_match[i])
                    {
                        case 0:
                            break;
                        case 1:
                            pred1Player += 1;
                            
                            break;
                        case 2:
                            pred2Player += 1;
                            pred1Player += 1;
                            
                            break;
                        case 3:
                            pred3Player += 1;
                            pred1Player += 1;
                            pred2Player += 1;
                            
                            break;
                        case 4:
                            pred4OrMorePlayer += 1;
                            pred1Player += 1;
                            pred2Player += 1;
                            pred3Player += 1;
                            
                            break;
                        default:
                            pred4OrMorePlayer += 1;
                            pred1Player += 1;
                            pred2Player += 1;
                            pred3Player += 1;
                            
                            break;
                    }
                } 

                if (outcomes[i] == 2 && (predictedOutcomes[i].Point >= -epsilon && predictedOutcomes[i].Point <= epsilon)){
                    accuracy += 1;
                    switch(known_player_per_match[i])
                    {
                        case 0:
                            break;
                        case 1:
                            pred1Player += 1;
                            
                            break;
                        case 2:
                            pred2Player += 1;
                            pred1Player += 1;
                            
                            break;
                        case 3:
                            pred3Player += 1;
                            pred1Player += 1;
                            pred2Player += 1;
                            
                            break;
                        case 4:
                            pred4OrMorePlayer += 1;
                            pred1Player += 1;
                            pred2Player += 1;
                            pred3Player += 1;
                            
                            break;
                        default:
                            pred4OrMorePlayer += 1;
                            pred1Player += 1;
                            pred2Player += 1;
                            pred3Player += 1;
                            
                            break;
                    }
                }
            }

            Console.WriteLine("ACCURACY");
            Console.WriteLine($" predicted draws: {predictedDraws}/{draw} predicted win t1: {predictedWinT1}/{winT1} predicted win T2: {predictedWinT2}/{winT2}");
            Console.WriteLine("winT1/winT2 accuracy: " + ((correctPredictedWinT1 + correctPredictedWinT2) / (winT1 + winT2)) );
            var pred1accuracy = countPlayersPerMatch[0] > 0 ? pred1Player/countPlayersPerMatch[0] : -1;
            var pred2accuracy = countPlayersPerMatch[1] > 0 ? pred2Player/countPlayersPerMatch[1] : -1;
            var pred3accuracy = countPlayersPerMatch[2] > 0 ? pred3Player/countPlayersPerMatch[2] : -1;
            var pred4accuracy = countPlayersPerMatch[3] > 0 ? pred4OrMorePlayer/countPlayersPerMatch[3] : -1; 
            Console.WriteLine("============= accuracy with reference to knwon players ==============");
            Console.WriteLine("# 1 known player ## 2 known players ## 3 kn pls ## 4 or more kn pls #");
            Console.WriteLine($"# {pred1accuracy} ## {pred2accuracy} ## {pred3accuracy} ## {pred4accuracy} #");
            Console.WriteLine($"# {pred1Player}/{countPlayersPerMatch[0]} ## {pred2Player}/{countPlayersPerMatch[1]} ## {pred3Player}/{countPlayersPerMatch[2]} ## {pred4OrMorePlayer}/{countPlayersPerMatch[3]} #");
            Console.WriteLine("======================================================================");
            
            int correct = (int)accuracy;

            accuracy = accuracy/(outcomes.Length - offset);
            Console.WriteLine("Overall Accuracy: " + accuracy);

            return correct;
        }


    }
}
