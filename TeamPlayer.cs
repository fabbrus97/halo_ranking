namespace ts2
{
    using Microsoft.ML.Probabilistic.Models;

    class TeamPlayer
    {
        public string tag { get; set; } 
        public double secondsPlayed { get; set; }
        public double joinTime { get; set; }
        public double endTime { get; set; }
        public int index { get; set; }
        public double killcount { get; set; }
        public double deathcount { get; set; }
        public bool quit { get; set; }
        public TeamPlayer(string tag, double secondsPlayed, double joinTime, double endTime, double killcount, double deathcount, bool quit)
        {
            this.tag = tag;
            this.secondsPlayed = secondsPlayed;
            this.joinTime = joinTime;
            this.endTime = endTime;
            this.killcount = killcount;
            this.deathcount = deathcount;
            this.quit = quit;
        }
    }
}