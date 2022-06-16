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



    public static class MyFactor
    {
        [ParameterNames("Sum", "a")]
        public static double Sum(double[] array)
        {
            double sum = 0;
            for (int i = 0; i < array.Length; i++)
            {
                sum = sum + array[i];
            }
            return sum;
        }
    }



    [FactorMethod(typeof(MyFactor), "Sum")]
    public static class SumOp
        {
            public static Gaussian SumAverageConditional(IList<Gaussian> array)
            {
                double mean=0;
                double variance=0;
                double mean1;
                double variance1;
                for (int i = 0; i < array.Count; i++)
                {
                    array[i].GetMeanAndVariance(out mean1, out variance1);
                    mean = mean + mean1;
                    variance = variance + variance1;
                }
                return new Gaussian(mean, variance);
            }

            public static GaussianArray ArrayAverageConditional<GaussianArray>([SkipIfUniform] GaussianArray array, [SkipIfUniform] Gaussian sum, [Fresh] Gaussian to_sum, GaussianArray result)
                where GaussianArray : IList<Gaussian>
            {
                double mean, mean1; double variance, variance1; // get the mean and variance of sum of all the Gaussians
                to_sum.GetMeanAndVariance(out mean, out variance);     // subtract it off from the mean and variance of incoming Gaussian from Sum
                sum.GetMeanAndVariance(out mean1, out variance1);
                mean = mean1 - mean;
                variance = variance1 + variance;
                for (int i = 0; i < array.Count; i++)
                {
                    array[i].GetMeanAndVariance(out mean1, out variance1);
                    result[i] = new Gaussian(mean + mean1, variance - variance1);
                }
                return result;

            }

            public static GaussianArray ArrayAverageConditional<GaussianArray>([SkipIfUniform] GaussianArray array, double sum, GaussianArray result)
            where GaussianArray : IList<Gaussian>
            {
                Gaussian to_sum = SumAverageConditional(array);
                return ArrayAverageConditional(array, Gaussian.PointMass(sum), to_sum, result);
            }

            public static double LogAverageFactor(double sum, [SkipIfAnyUniform] IList<Gaussian> array)
            {
                Gaussian to_sum = SumAverageConditional(array);
                return to_sum.GetLogProb(sum);
            }

            public static double LogAverageFactor([SkipIfUniform] Gaussian sum, [SkipIfAnyUniform] IList<Gaussian> array)
            {
                Gaussian to_sum = SumAverageConditional(array);
                return to_sum.GetLogAverageOf(sum);
            }

            public static double LogEvidenceRatio(double sum, [SkipIfAnyUniform] IList<Gaussian> array) { return LogAverageFactor(sum, array); }

            [Skip]
            public static double LogEvidenceRatio(Gaussian sum) { return 0.0; }


    }




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
        double m0;
        double v0;
        double gamma;
        double tau;
        double beta = 1;
        double epsilon = 10e-3;

        double w_k_p = 1.1e-5, w_k_o = -9.1e-5, w_d_p = -1.1e-5, w_d_o = 7.5e-5, v_c = 1e-5; //54% accuracy without m_q, v_q
        double m_q = 15, v_q = 1*10e-3;
        Range nMatches;
        Range nPlayers;
        Range nTeamsPerMatch;
        VariableArray<VariableArray<int>, int[][]> playersInTeam;
        Range nPlayersPerTeam;
        Variable<double> drawMargin;
        Variable<bool> unrelated;
        Variable<bool> related;

        VariableArray<VariableArray<VariableArray<int>, int[][]>, int[][][]> matchesVariable;
        VariableArray<double> skillsVariable;
        VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> killcount;
        VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> deathcount;
        VariableArray<int> outcomes;
        VariableArray<double> matchTime;
        VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> playerTime;
        VariableArray<VariableArray<VariableArray<bool>, bool[][]>, bool[][][]> quit;
        Gaussian[] BaseSkills;

        const int N_INTERATIONS = 500;


        Gaussian[] skills;

        private List<Match> matches ;

         public ParamsCalculator(List<Match> matches)
         {
             this.matches = matches;
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

        public Gaussian[] ComputeSkills()
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
                    gameInPlayersList[player].Add(i);

                    matchData[i][0][k] = player;

                    playersTime[i][0][k] = matches[i].team1.teammates[k].secondsPlayed;

                    killcount[i][0][k] = matches[i].team1.teammates[k].killcount;
                    deathcount[i][0][k] = matches[i].team1.teammates[k].deathcount;

                    quit[i][0][k] = matches[i].team1.teammates[k].quit;

                }
                for (int k = 0; k < matches[i].team2.nPlayers(); k++){
                    var player = _players.FindIndex(t => t == matches[i].team2.teammates[k].tag);
                    playersInGame[i][1][k] = player;
                    gameInPlayersList[player].Add(i);


                    matchData[i][1][k] = player;

                    playersTime[i][1][k] = matches[i].team2.teammates[k].secondsPlayed;

                    killcount[i][1][k] = matches[i].team2.teammates[k].killcount;
                    deathcount[i][1][k] = matches[i].team2.teammates[k].deathcount;

                    quit[i][1][k] = matches[i].team2.teammates[k].quit;
                }


                if (matches[i].isDraw())
                    outcomes[i] = 2;
                else if (matches[i].isTeam1Winner())
                    outcomes[i] = 0;
                else
                    outcomes[i] = 1;


            }


            int[][] gameInPlayers = new int[nPlayers][];

            for (int i = 0; i < nPlayers; i++)
            {
                gameInPlayers[i] = new int[gameInPlayersList[i].Count()];
                gameInPlayers[i] = gameInPlayersList[i].ToArray();
            }

            SetVariables(matchData, matches.Count, nPlayers,  nPlayersPerTeam, _players.ToArray(), outcomes, matchTime, playersTime, killcount, deathcount, quit);

            if (this.ParameterComputed)
            {
                this.skills = InferSkills();
            }
            else{
                this.ParameterComputed = true;
                this.skills = InferSkillsAndParameters(matches.Count(), nPlayers);
                inferGamma(nPlayers); //TODO scrivi funzione che applica aumento skill secondo gamma senza fare l'inferenza
            }

            return this.skills;
        }

        public ParamsCalculator()
        {
            this.ParameterComputed = false;
        }

        public void SetMatches(List<Match> matches)
        {
            this.matches = matches;
        }

        private void SetVariables(int[][][] matchData, int nMatchesObserved, int nPlayersObserved, int[][] nPlayersPerTeamObserved, string[] playersName, int[] outcomesObserved,
            double[] matchTimeObserved, double[][][] playersTimeObserved, double[][][] killCountObserved, double[][][] deathCountObserved, bool[][][] quitObserved)
        {
            nMatches = new Range(nMatchesObserved).Named("nMatches"); // a sample of n matches
            nPlayers = new Range(nPlayersObserved).Named("nPlayers"); // p unique players in the sample
            nTeamsPerMatch = new Range(2).Named("nTeamsPerMatch"); // 2 teams in each m
            playersInTeam = Variable.Array<int>(Variable.Array<int>(nTeamsPerMatch), nMatches).Named("PlayerInTeam");
            nPlayersPerTeam = new Range(playersInTeam[nMatches][nTeamsPerMatch]).Named("nPlayers-PerTeam"); // 4 players per team

            matchesVariable    = Variable.Array(Variable.Array(Variable.Array<int>   (nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("matches"); //.Attrib(new DoNotInfer());
            skillsVariable     = Variable.Array<double>(nPlayers).Named("skills"); //.Attrib(new DoNotInfer());
            killcount  = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("killcount");
            deathcount = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("deathcount");
            outcomes   = Variable.Array<int>(nMatches).Named("outcomes");
            matchTime  = Variable.Array<double>(nMatches).Named("matchTime");
            playerTime = Variable.Array(Variable.Array(Variable.Array<double>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("playerTime");
            quit       = Variable.Array(Variable.Array(Variable.Array<bool>   (nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("quit"); //.Attrib(new DoNotInfer());

            related = Variable.Bernoulli(0.5);
            unrelated = Variable.Bernoulli(0.5);

            matchesVariable.ObservedValue = matchData;
            outcomes.ObservedValue = outcomesObserved;
            playersInTeam.ObservedValue = nPlayersPerTeamObserved;
            matchTime.ObservedValue = matchTimeObserved;
            playerTime.ObservedValue = playersTimeObserved;
            killcount.ObservedValue = killCountObserved;
            deathcount.ObservedValue = deathCountObserved;
            quit.ObservedValue = quitObserved;

            if (BaseSkills != null)
            {
                Console.WriteLine("m0 v0: " + m0 + " " + v0);
                int i = 0;
                foreach (Gaussian skill in BaseSkills)
                {
                  Gaussian conf= new Gaussian();

                    if (!(skill.Equals(conf)))
                    {
                        try
                        {
                            //Console.WriteLine("Skill esistente: " + skill.GetMean() + " " + skill.GetVariance());
                            skillsVariable[i] = Variable.GaussianFromMeanAndVariance(skill.GetMean(), skill.GetVariance());
                        } catch (Microsoft.ML.Probabilistic.Distributions.ImproperDistributionException e)
                        {
                            Console.WriteLine("Errore per il giocatore " + playersName[i]);
                            //skillsVariable[i] = Variable.GaussianFromMeanAndVariance(m0, v0);
                        }

                    }
                    else
                    {
                        //Console.WriteLine("Nuovo giocatore: " + m0 + " " + v0);
                        skillsVariable[i] = Variable.GaussianFromMeanAndVariance(m0, v0);


                    }

                    i += 1;
                }
            }

        }

        Gaussian[] InferSkillsAndParameters(int nMatchesObserved, int nPlayersObserved)
        {
            var _m0 = Variable.GaussianFromMeanAndVariance(10, nMatchesObserved*nMatchesObserved).Named("m0");
            var _p0 = Variable.GammaFromShapeAndScale(1,1).Named("p0");
            var _drawMargin = Variable.GaussianFromMeanAndVariance(10e-2, nMatchesObserved).Named("Epsilon");

            _m0.AddAttribute(new PointEstimate());
            _m0.AddAttribute(new ListenToMessages());
            _p0.AddAttribute(new PointEstimate());
            _p0.AddAttribute(new ListenToMessages());
            _drawMargin.AddAttribute(new PointEstimate());
            _drawMargin.AddAttribute(new ListenToMessages());

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
                        kill.Named("death count m, t, n");
                        deathcount[nMatches][team.Index][nPlayersPerTeam] = Variable.Max(0, Variable.GaussianFromMeanAndPrecision(death, b));

                        //is the player underperforming? (for quit penalty)
                        var under = (Variable.GaussianFromMeanAndPrecision(pPerf - perf_opposing - m_q, v_q) < 0);
                        under.Named("Under performing");

                        //quit penalty
                        quit[nMatches][nTeamsPerMatch][nPlayersPerTeam] = unrelated | (related & under);
                    }
                }


                var diff = (Variable.Sum(playerPerformance[0]).Named("SumTeam1") - Variable.Sum(playerPerformance[1]).Named("SumTeam2"));
                diff.Named("Diff");

                using (Variable.Case(outcomes[n].Named("draw"), 2))
                {
                    Variable.ConstrainBetween(diff, -_drawMargin, _drawMargin);
                }

                using(Variable.Case(outcomes[n].Named("team1wins"), 0))
                {
                    Variable.ConstrainTrue(diff > _drawMargin);
                }

                using(Variable.Case(outcomes[n].Named("team2wins"), 1))
                {
                    Variable.ConstrainTrue(diff < -_drawMargin);
                }

            }

            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false,
                NumberOfIterations = N_INTERATIONS
            };


            var inferredm0 = inferenceEngine.Infer<Gaussian>(_m0);
            var inferredp0 = inferenceEngine.Infer<Gamma>(_p0);
            var inferredDrawMargin = inferenceEngine.Infer<Gaussian>(_drawMargin);
            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(skillsVariable);

            Console.WriteLine(inferredm0);
            Console.WriteLine(inferredp0);
            Console.WriteLine(inferredDrawMargin);

            this.epsilon = Math.Max(10e-3, inferredDrawMargin.Point);
            this.m0 = inferredm0.Point;
            this.v0 = 1/inferredp0.Point;

            var skillsArray = new double[nPlayersObserved];

            /*Console.WriteLine("Best player: " + inferredSkills.OrderByDescending(s => s.GetMean()).First());
            Console.WriteLine("===============================");

            for (int i = 0; i < nPlayersObserved; i++ )
            {
                if (i < 10)
                {
                    Console.WriteLine("Player: " + playersName[i]);
                    Console.WriteLine(inferredSkills[i]);
                }
                skillsArray[i] = inferredSkills[i].GetMean();
            }

            Console.WriteLine("===============================");
            Console.WriteLine("Worst player: " + inferredSkills.OrderByDescending(s => s.GetMean()).Last());*/

            return inferredSkills;
        }

        Gaussian[] InferSkills()
        {
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
                        kill.Named("death count m, t, n");
                        deathcount[nMatches][team.Index][nPlayersPerTeam] = Variable.Max(0, Variable.GaussianFromMeanAndPrecision(death, b));

                        //is the player underperforming? (for quit penalty)
                        var under = (Variable.GaussianFromMeanAndPrecision(pPerf - perf_opposing - m_q, v_q) < 0);
                        under.Named("Under performing");

                        //quit penalty
                        quit[nMatches][nTeamsPerMatch][nPlayersPerTeam] = unrelated | (related & under);
                    }
                }


                var diff = (Variable.Sum(playerPerformance[0]).Named("SumTeam1") - Variable.Sum(playerPerformance[1]).Named("SumTeam2"));
                diff.Named("Diff");

                using (Variable.Case(outcomes[n].Named("draw"), 2))
                {
                    Variable.ConstrainBetween(diff, -epsilon, epsilon);
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
                NumberOfIterations = N_INTERATIONS
            };


            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(skillsVariable);

            return inferredSkills;
        }

        void inferGamma(int nPlayersObserved)
        {

            /*var nMatches = new Range(nMatchesObserved).Named("nMatches"); // a sample of 1000 matches
            // nMatches.AddAttribute(new Sequential());
            var nPlayers = new Range(nPlayersObserved).Named("nPlayers"); // 363 unique players in the sample
            var nTeamsPerMatch = new Range(2).Named("nTeamsPerMatch"); // 2 teams in each match

            var playersInTeam = Variable.Array<int>(Variable.Array<int>(nTeamsPerMatch), nMatches);
            var nPlayersPerTeam = new Range(playersInTeam[nMatches][nTeamsPerMatch]).Named("nPlayers-PerTeam"); // 4 players per team

            // var nPlayersPerTeam = new Range(4).Named("nPlayersPerTeam"); // 4 players per team*/
            var skillsArray = new double[nPlayersObserved];
            for (int i = 0; i < nPlayersObserved; i++ )
            {
                skillsArray[i] = this.skills[i].GetMean();
            }



            var _gamma = Variable.GammaFromShapeAndScale(1,1);
            // var _gamma = Variable.GaussianFromMeanAndPrecision(0, 100);
            _gamma.AddAttribute(new PointEstimate());
            _gamma.AddAttribute(new ListenToMessages());

            //"reset" of the variable
            var skillsVariable     = Variable.Array<double>(nPlayers).Named("skills (gamma)"); //.Attrib(new DoNotInfer());


            // var matches = Variable.Array(Variable.Array(Variable.Array<int>(nPlayersPerTeam), nTeamsPerMatch), nMatches).Named("matches"); //.Attrib(new DoNotInfer());
            // var skills = Variable.Array<double>(nPlayers); //.Attrib(new DoNotInfer());

            /* Inferenza di Gamma */
            using (Variable.ForEach(nMatches))
            {
                using (Variable.ForEach(nTeamsPerMatch))
                {
                    using (Variable.ForEach(nPlayersPerTeam))
                    {
                        var playerIndex = matchesVariable[nMatches][nTeamsPerMatch][nPlayersPerTeam];
                        var dampedSkill = Variable<double>.Factor(Damp.Backward, skillsVariable[playerIndex], 0.5);

                        skillsVariable[playerIndex] = Variable.GaussianFromMeanAndPrecision(dampedSkill, _gamma*_gamma);
                    }


                }
            }

            skillsVariable.ObservedValue = skillsArray;

            var inferenceEngine = new InferenceEngine
            {
                ShowFactorGraph = false,
                NumberOfIterations = N_INTERATIONS
            };

            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(skillsVariable);
            var inferredgamma = inferenceEngine.Infer<Gamma>(_gamma);

            Console.WriteLine(" \nInferenza di gamma:");
            Console.WriteLine(inferredgamma + $"({1/inferredgamma.GetMean()})");
            this.gamma = 1/inferredgamma.GetMean();
        }

        public double predictAccuracy(List<Match> match2predictList)
        {

            List<string> skill_players = players(match2predictList);

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


                if (match2predictList[i].isDraw())
                    outcomes[i] = 2;
                else if (match2predictList[i].isTeam1Winner())
                    outcomes[i] = 0; //TODO
                else
                    outcomes[i] = 1;
            }



            var accuracy = 0.0;

            var correctPredictions = 0.0;

            int predT1win = 0;
            int predT2win = 0;
            int predDraw = 0;

            int winT1 = 0; int winT2 = 0; int draw = 0;

            for (int m = 0; m < match2predict.Count(); m++)
            {
                var expectedPerfT1 = 0.0;
                var expectedPerfT2 = 0.0;

                for(int p = 0; p < match2predict[m][0].Count(); p++)
                {
                    var pl = _players[match2predict[m][0][p]];
                    var index = skill_players.FindIndex(0, skill_players.Count(), tag => tag == pl);
                    if (index > 0)
                        expectedPerfT1 += new Gaussian(this.skills[index].GetMean(), 1).GetMean();
                    else {
                        expectedPerfT1 += new Gaussian(m0, 1).GetMean();
                    }
                }

                for(int p = 0; p < match2predict[m][1].Count(); p++)
                {
                    var pl = _players[match2predict[m][1][p]];
                    var index = skill_players.FindIndex(0, skill_players.Count(), tag => tag == pl);
                    if (index > 0)
                        expectedPerfT2 += new Gaussian(this.skills[index].GetMean(), 1).GetMean();
                    else {
                        expectedPerfT2 += new Gaussian(m0, 1).GetMean();
                    }
                }

                if (outcomes[m] == 0 )
                    winT1 += 1;

                if (outcomes[m] == 1 )
                    winT2 += 1;

                if (outcomes[m] == 2 )
                    draw += 1;

                if (expectedPerfT1 - expectedPerfT2 > this.epsilon && outcomes[m] == 0)
                {
                    correctPredictions += 1;
                    predT1win += 1;
                }
                else if (expectedPerfT1 - expectedPerfT2 < -this.epsilon && outcomes[m] == 1)
                {
                    correctPredictions += 1;
                    predT2win += 1;
                }
                else if (Math.Abs(expectedPerfT1 - expectedPerfT2) <= this.epsilon && outcomes[m] == 2)
                {
                    correctPredictions += 1;
                    predDraw += 0;
                }

            }

            Console.WriteLine($"DEBUG {predT1win}/{winT1} {predT2win}/{winT2} {predDraw}/{draw}");
            accuracy = correctPredictions/outcomes.Count();

            return accuracy*100;
        }

    }
}
