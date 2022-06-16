namespace ts2
{
    class Match
    {
    public enum Winner
    {
        TEAM1, TEAM2, DRAW
    }
    public Team team1 {get; set;}
    public Team team2 {get; set;}
    private Winner winner;
    public string id { get; set; }
    public string mode { get; set; }
    public double secondsPlayed { get; set; }
    public double startTime { get; set; }
    public double endTime { get; set; }

    public Match(Team team1, Team team2, Winner winner, string id, string mode, double startTime, double endTime, double secondsPlayed)
    {
        this.team1 = team1;
        this.team2 = team2;
        this.winner = winner;
        this.id = id;
        this.mode=mode;
        this.startTime = startTime;
        this.endTime = endTime;
        this.secondsPlayed = secondsPlayed;
    }

    public Team getWinner()
    {
        return winner == Winner.TEAM1 ? team1 : team2;
    }

    public Team getLoser()
    {
        return winner == Winner.TEAM2 ? team1 : team2;
    }

    public bool isDraw()
    {
        return winner == Winner.DRAW;
    }

    public bool isTeam1Winner()
    {
        return winner == Winner.TEAM1;
    }

    public bool isTeam2Winner()
    {
        return winner == Winner.TEAM2;
    }

    public int totPlayers()
    {
        return this.team1.nPlayers() + this.team2.nPlayers();
    }

    }

    class Team
    {
        public List<TeamPlayer> teammates { get; set; }

        public Team(List<TeamPlayer> teammates)
        {
            this.teammates = teammates;
        }

        public int nPlayers()
        {
            return teammates.Count();
        }
    }


}
