// <auto-generated />
#pragma warning disable 1570, 1591

using System;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;

namespace Models
{
	/// <summary>
	/// Generated algorithm for performing inference.
	/// </summary>
	/// <remarks>
	/// If you wish to use this class directly, you must perform the following steps:
	/// 1) Create an instance of the class.
	/// 2) Set the value of any externally-set fields e.g. data, priors.
	/// 3) Call the Execute(numberOfIterations) method.
	/// 4) Use the XXXMarginal() methods to retrieve posterior marginals for different variables.
	/// 
	/// Generated by Infer.NET 0.4.2203.202 at 14:06 on sabato 25 giugno 2022.
	/// </remarks>
	public partial class Model82_EP : IGeneratedAlgorithm
	{
		#region Fields
		/// <summary>True if Changed_matches has executed. Set this to false to force re-execution of Changed_matches</summary>
		public bool Changed_matches_isDone;
		/// <summary>True if Changed_PlayerInTeam has executed. Set this to false to force re-execution of Changed_PlayerInTeam</summary>
		public bool Changed_PlayerInTeam_isDone;
		/// <summary>True if Changed_PlayerInTeam_time_passed has executed. Set this to false to force re-execution of Changed_PlayerInTeam_time_passed</summary>
		public bool Changed_PlayerInTeam_time_passed_isDone;
		/// <summary>True if Changed_skills__tau_ has executed. Set this to false to force re-execution of Changed_skills__tau_</summary>
		public bool Changed_skills__tau__isDone;
		/// <summary>True if Changed_vbool__113 has executed. Set this to false to force re-execution of Changed_vbool__113</summary>
		public bool Changed_vbool__113_isDone;
		/// <summary>Field backing the matches property</summary>
		private int[][][] Matches;
		/// <summary>Message to marginal of 'matches'</summary>
		public PointMass<int[][][]> matches_marginal_F;
		/// <summary>Field backing the NumberOfIterationsDone property</summary>
		private int numberOfIterationsDone;
		/// <summary>Field backing the PlayerInTeam property</summary>
		private int[][] playerInTeam;
		/// <summary>Message to marginal of 'PlayerInTeam'</summary>
		public PointMass<int[][]> PlayerInTeam_marginal_F;
		/// <summary>Field backing the skills__tau_ property</summary>
		private double[] Skills__tau_;
		/// <summary>Message to marginal of 'skills__tau_'</summary>
		public DistributionStructArray<Gaussian,double> skills__tau__marginal_F;
		/// <summary>Field backing the time_passed property</summary>
		private double[][][] Time_passed;
		/// <summary>Message to marginal of 'time_passed'</summary>
		public DistributionRefArray<DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>,double[][]> time_passed_marginal_F;
		/// <summary>Field backing the vbool__113 property</summary>
		private bool[] Vbool__113;
		/// <summary>Message to marginal of 'vbool__113'</summary>
		public DistributionStructArray<Bernoulli,bool> vbool__113_marginal_F;
		#endregion

		#region Properties
		/// <summary>The externally-specified value of 'matches'</summary>
		public int[][][] matches
		{
			get {
				return this.Matches;
			}
			set {
				if ((value!=null)&&(value.Length!=1000)) {
					throw new ArgumentException(((("Provided array of length "+value.Length)+" when length ")+1000)+" was expected for variable \'matches\'");
				}
				this.Matches = value;
				this.numberOfIterationsDone = 0;
				this.Changed_matches_isDone = false;
			}
		}

		/// <summary>The number of iterations done from the initial state</summary>
		public int NumberOfIterationsDone
		{
			get {
				return this.numberOfIterationsDone;
			}
		}

		/// <summary>The externally-specified value of 'PlayerInTeam'</summary>
		public int[][] PlayerInTeam
		{
			get {
				return this.playerInTeam;
			}
			set {
				if ((value!=null)&&(value.Length!=1000)) {
					throw new ArgumentException(((("Provided array of length "+value.Length)+" when length ")+1000)+" was expected for variable \'PlayerInTeam\'");
				}
				this.playerInTeam = value;
				this.numberOfIterationsDone = 0;
				this.Changed_PlayerInTeam_time_passed_isDone = false;
				this.Changed_PlayerInTeam_isDone = false;
			}
		}

		/// <summary>The externally-specified value of 'skills__tau_'</summary>
		public double[] skills__tau_
		{
			get {
				return this.Skills__tau_;
			}
			set {
				if ((value!=null)&&(value.Length!=7022)) {
					throw new ArgumentException(((("Provided array of length "+value.Length)+" when length ")+7022)+" was expected for variable \'skills__tau_\'");
				}
				this.Skills__tau_ = value;
				this.numberOfIterationsDone = 0;
				this.Changed_skills__tau__isDone = false;
			}
		}

		/// <summary>The externally-specified value of 'time_passed'</summary>
		public double[][][] time_passed
		{
			get {
				return this.Time_passed;
			}
			set {
				if ((value!=null)&&(value.Length!=1000)) {
					throw new ArgumentException(((("Provided array of length "+value.Length)+" when length ")+1000)+" was expected for variable \'time_passed\'");
				}
				this.Time_passed = value;
				this.numberOfIterationsDone = 0;
				this.Changed_PlayerInTeam_time_passed_isDone = false;
			}
		}

		/// <summary>The externally-specified value of 'vbool__113'</summary>
		public bool[] vbool__113
		{
			get {
				return this.Vbool__113;
			}
			set {
				if ((value!=null)&&(value.Length!=7022)) {
					throw new ArgumentException(((("Provided array of length "+value.Length)+" when length ")+7022)+" was expected for variable \'vbool__113\'");
				}
				this.Vbool__113 = value;
				this.numberOfIterationsDone = 0;
				this.Changed_vbool__113_isDone = false;
			}
		}

		#endregion

		#region Methods
		/// <summary>Computations that depend on the observed value of matches</summary>
		private void Changed_matches()
		{
			if (this.Changed_matches_isDone) {
				return ;
			}
			// Create array for 'matches_marginal' Forwards messages.
			this.matches_marginal_F = new PointMass<int[][][]>(this.Matches);
			// Message to 'matches_marginal' from DerivedVariable factor
			this.matches_marginal_F = DerivedVariableOp.MarginalAverageConditional<PointMass<int[][][]>,int[][][]>(this.Matches, this.matches_marginal_F);
			this.Changed_matches_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of PlayerInTeam</summary>
		private void Changed_PlayerInTeam()
		{
			if (this.Changed_PlayerInTeam_isDone) {
				return ;
			}
			// Create array for 'PlayerInTeam_marginal' Forwards messages.
			this.PlayerInTeam_marginal_F = new PointMass<int[][]>(this.playerInTeam);
			// Message to 'PlayerInTeam_marginal' from DerivedVariable factor
			this.PlayerInTeam_marginal_F = DerivedVariableOp.MarginalAverageConditional<PointMass<int[][]>,int[][]>(this.playerInTeam, this.PlayerInTeam_marginal_F);
			this.Changed_PlayerInTeam_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of PlayerInTeam and time_passed</summary>
		private void Changed_PlayerInTeam_time_passed()
		{
			if (this.Changed_PlayerInTeam_time_passed_isDone) {
				return ;
			}
			// Create array for 'time_passed_marginal' Forwards messages.
			this.time_passed_marginal_F = new DistributionRefArray<DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>,double[][]>(1000);
			for(int nMatches = 0; nMatches<1000; nMatches++) {
				// Create array for 'time_passed_marginal' Forwards messages.
				this.time_passed_marginal_F[nMatches] = new DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>(2);
				for(int nTeamsPerMatch = 0; nTeamsPerMatch<2; nTeamsPerMatch++) {
					// Create array for 'time_passed_marginal' Forwards messages.
					this.time_passed_marginal_F[nMatches][nTeamsPerMatch] = new DistributionStructArray<Gaussian,double>(this.playerInTeam[nMatches][nTeamsPerMatch]);
					for(int nPlayersMinusPerTeam = 0; nPlayersMinusPerTeam<this.playerInTeam[nMatches][nTeamsPerMatch]; nPlayersMinusPerTeam++) {
						this.time_passed_marginal_F[nMatches][nTeamsPerMatch][nPlayersMinusPerTeam] = Gaussian.Uniform();
					}
				}
			}
			// Message to 'time_passed_marginal' from DerivedVariable factor
			this.time_passed_marginal_F = DerivedVariableOp.MarginalAverageConditional<DistributionRefArray<DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>,double[][]>,double[][][]>(this.Time_passed, this.time_passed_marginal_F);
			this.Changed_PlayerInTeam_time_passed_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of skills__tau_</summary>
		private void Changed_skills__tau_()
		{
			if (this.Changed_skills__tau__isDone) {
				return ;
			}
			// Create array for 'skills__tau__marginal' Forwards messages.
			this.skills__tau__marginal_F = new DistributionStructArray<Gaussian,double>(7022);
			for(int nPlayers = 0; nPlayers<7022; nPlayers++) {
				this.skills__tau__marginal_F[nPlayers] = Gaussian.Uniform();
			}
			// Message to 'skills__tau__marginal' from DerivedVariable factor
			this.skills__tau__marginal_F = DerivedVariableOp.MarginalAverageConditional<DistributionStructArray<Gaussian,double>,double[]>(this.Skills__tau_, this.skills__tau__marginal_F);
			this.Changed_skills__tau__isDone = true;
		}

		/// <summary>Computations that depend on the observed value of vbool__113</summary>
		private void Changed_vbool__113()
		{
			if (this.Changed_vbool__113_isDone) {
				return ;
			}
			// Create array for 'vbool__113_marginal' Forwards messages.
			this.vbool__113_marginal_F = new DistributionStructArray<Bernoulli,bool>(7022);
			for(int nPlayers = 0; nPlayers<7022; nPlayers++) {
				this.vbool__113_marginal_F[nPlayers] = Bernoulli.Uniform();
			}
			// Message to 'vbool__113_marginal' from DerivedVariable factor
			this.vbool__113_marginal_F = DerivedVariableOp.MarginalAverageConditional<DistributionStructArray<Bernoulli,bool>,bool[]>(this.Vbool__113, this.vbool__113_marginal_F);
			this.Changed_vbool__113_isDone = true;
		}

		/// <summary>Update all marginals, by iterating message passing the given number of times</summary>
		/// <param name="numberOfIterations">The number of times to iterate each loop</param>
		/// <param name="initialise">If true, messages that initialise loops are reset when observed values change</param>
		private void Execute(int numberOfIterations, bool initialise)
		{
			this.Changed_skills__tau_();
			this.Changed_vbool__113();
			this.Changed_PlayerInTeam_time_passed();
			this.Changed_matches();
			this.Changed_PlayerInTeam();
			this.numberOfIterationsDone = numberOfIterations;
		}

		/// <summary>Update all marginals, by iterating message-passing the given number of times</summary>
		/// <param name="numberOfIterations">The total number of iterations that should be executed for the current set of observed values.  If this is more than the number already done, only the extra iterations are done.  If this is less than the number already done, message-passing is restarted from the beginning.  Changing the observed values resets the iteration count to 0.</param>
		public void Execute(int numberOfIterations)
		{
			this.Execute(numberOfIterations, true);
		}

		/// <summary>Get the observed value of the specified variable.</summary>
		/// <param name="variableName">Variable name</param>
		public object GetObservedValue(string variableName)
		{
			if (variableName=="PlayerInTeam") {
				return this.PlayerInTeam;
			}
			if (variableName=="matches") {
				return this.matches;
			}
			if (variableName=="time_passed") {
				return this.time_passed;
			}
			if (variableName=="vbool__113") {
				return this.vbool__113;
			}
			if (variableName=="skills__tau_") {
				return this.skills__tau_;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName)
		{
			if (variableName=="skills__tau_") {
				return this.Skills__tau_Marginal();
			}
			if (variableName=="vbool__113") {
				return this.Vbool__113Marginal();
			}
			if (variableName=="time_passed") {
				return this.Time_passedMarginal();
			}
			if (variableName=="matches") {
				return this.MatchesMarginal();
			}
			if (variableName=="PlayerInTeam") {
				return this.PlayerInTeamMarginal();
			}
			throw new ArgumentException("This class was not built to infer "+variableName);
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable, converted to type T</summary>
		/// <typeparam name="T">The distribution type.</typeparam>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public T Marginal<T>(string variableName)
		{
			return Distribution.ChangeType<T>(this.Marginal(variableName));
		}

		/// <summary>Get the query-specific marginal distribution of a variable.</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <param name="query">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName, string query)
		{
			if (query=="Marginal") {
				return this.Marginal(variableName);
			}
			throw new ArgumentException(((("This class was not built to infer \'"+variableName)+"\' with query \'")+query)+"\'");
		}

		/// <summary>Get the query-specific marginal distribution of a variable, converted to type T</summary>
		/// <typeparam name="T">The distribution type.</typeparam>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <param name="query">QueryType name. For example, GibbsSampling answers 'Marginal', 'Samples', and 'Conditionals' queries</param>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public T Marginal<T>(string variableName, string query)
		{
			return Distribution.ChangeType<T>(this.Marginal(variableName, query));
		}

		/// <summary>
		/// Returns the marginal distribution for 'matches' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public PointMass<int[][][]> MatchesMarginal()
		{
			return this.matches_marginal_F;
		}

		private void OnProgressChanged(ProgressChangedEventArgs e)
		{
			// Make a temporary copy of the event to avoid a race condition
			// if the last subscriber unsubscribes immediately after the null check and before the event is raised.
			EventHandler<ProgressChangedEventArgs> handler = this.ProgressChanged;
			if (handler!=null) {
				handler(this, e);
			}
		}

		/// <summary>
		/// Returns the marginal distribution for 'PlayerInTeam' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public PointMass<int[][]> PlayerInTeamMarginal()
		{
			return this.PlayerInTeam_marginal_F;
		}

		/// <summary>Reset all messages to their initial values.  Sets NumberOfIterationsDone to 0.</summary>
		public void Reset()
		{
			this.Execute(0);
		}

		/// <summary>Set the observed value of the specified variable.</summary>
		/// <param name="variableName">Variable name</param>
		/// <param name="value">Observed value</param>
		public void SetObservedValue(string variableName, object value)
		{
			if (variableName=="PlayerInTeam") {
				this.PlayerInTeam = (int[][])value;
				return ;
			}
			if (variableName=="matches") {
				this.matches = (int[][][])value;
				return ;
			}
			if (variableName=="time_passed") {
				this.time_passed = (double[][][])value;
				return ;
			}
			if (variableName=="vbool__113") {
				this.vbool__113 = (bool[])value;
				return ;
			}
			if (variableName=="skills__tau_") {
				this.skills__tau_ = (double[])value;
				return ;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>
		/// Returns the marginal distribution for 'skills__tau_' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public DistributionStructArray<Gaussian,double> Skills__tau_Marginal()
		{
			return this.skills__tau__marginal_F;
		}

		/// <summary>
		/// Returns the marginal distribution for 'time_passed' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public DistributionRefArray<DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>,double[][]> Time_passedMarginal()
		{
			return this.time_passed_marginal_F;
		}

		/// <summary>Update all marginals, by iterating message-passing an additional number of times</summary>
		/// <param name="additionalIterations">The number of iterations that should be executed, starting from the current message state.  Messages are not reset, even if observed values have changed.</param>
		public void Update(int additionalIterations)
		{
			this.Execute(checked(this.numberOfIterationsDone+additionalIterations), false);
		}

		/// <summary>
		/// Returns the marginal distribution for 'vbool__113' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public DistributionStructArray<Bernoulli,bool> Vbool__113Marginal()
		{
			return this.vbool__113_marginal_F;
		}

		#endregion

		#region Events
		/// <summary>Event that is fired when the progress of inference changes, typically at the end of one iteration of the inference algorithm.</summary>
		public event EventHandler<ProgressChangedEventArgs> ProgressChanged;
		#endregion

	}

}
