// <auto-generated />
#pragma warning disable 1570, 1591

using System;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Collections;

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
	/// Generated by Infer.NET 0.4.2203.202 at 15:04 on giovedì 14 luglio 2022.
	/// </remarks>
	public partial class Model37_EP : IGeneratedAlgorithm
	{
		#region Fields
		/// <summary>True if Changed_matchTime has executed. Set this to false to force re-execution of Changed_matchTime</summary>
		public bool Changed_matchTime_isDone;
		/// <summary>True if Changed_matchTime_numberOfIterations_PlayerIndex_PlayerInTeam_playerTime_vdouble__345 has executed. Set this to false to force re-execution of Changed_matchTime_numberOfIterations_PlayerIndex_PlayerInTeam_playerTime_vdouble__345</summary>
		public bool Changed_matchTime_numberOfIterations_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isDone;
		/// <summary>True if Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345 has executed. Set this to false to force re-execution of Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345</summary>
		public bool Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isDone;
		/// <summary>True if Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345 has performed initialisation. Set this to false to force re-execution of Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345</summary>
		public bool Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isInitialised;
		/// <summary>True if Changed_PlayerIndex has executed. Set this to false to force re-execution of Changed_PlayerIndex</summary>
		public bool Changed_PlayerIndex_isDone;
		/// <summary>True if Changed_PlayerInTeam has executed. Set this to false to force re-execution of Changed_PlayerInTeam</summary>
		public bool Changed_PlayerInTeam_isDone;
		/// <summary>True if Changed_PlayerInTeam_playerTime has executed. Set this to false to force re-execution of Changed_PlayerInTeam_playerTime</summary>
		public bool Changed_PlayerInTeam_playerTime_isDone;
		/// <summary>True if Changed_vdouble__345 has executed. Set this to false to force re-execution of Changed_vdouble__345</summary>
		public bool Changed_vdouble__345_isDone;
		/// <summary>True if Constant has executed. Set this to false to force re-execution of Constant</summary>
		public bool Constant_isDone;
		/// <summary>Field backing the matchTime property</summary>
		private double[] MatchTime;
		/// <summary>Message to marginal of 'matchTime'</summary>
		public DistributionStructArray<Gaussian,double> matchTime_marginal_F;
		/// <summary>Field backing the NumberOfIterationsDone property</summary>
		private int numberOfIterationsDone;
		/// <summary>Field backing the PlayerIndex property</summary>
		private int[][][] playerIndex;
		/// <summary>Message to marginal of 'PlayerIndex'</summary>
		public PointMass<int[][][]> PlayerIndex_marginal_F;
		/// <summary>Field backing the PlayerInTeam property</summary>
		private int[][] playerInTeam;
		/// <summary>Message to marginal of 'PlayerInTeam'</summary>
		public PointMass<int[][]> PlayerInTeam_marginal_F;
		/// <summary>Field backing the playerTime property</summary>
		private double[][][] PlayerTime;
		/// <summary>Message to marginal of 'playerTime'</summary>
		public DistributionRefArray<DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>,double[][]> playerTime_marginal_F;
		/// <summary>Field backing the vdouble__345 property</summary>
		private double[] Vdouble__345;
		/// <summary>Message to marginal of 'vdouble__345'</summary>
		public DistributionStructArray<Gaussian,double> vdouble__345_marginal_F;
		/// <summary>Message to marginal of 'vdouble__347'</summary>
		public DistributionStructArray<Gaussian,double> vdouble__347_marginal_F;
		/// <summary>Buffer for VariablePointOp_Rprop.MarginalAverageConditional</summary>
		public RpropBufferData[] vdouble__347_use_B_nMatches__buffer;
		#endregion

		#region Properties
		/// <summary>The externally-specified value of 'matchTime'</summary>
		public double[] matchTime
		{
			get {
				return this.MatchTime;
			}
			set {
				if ((value!=null)&&(value.Length!=1000)) {
					throw new ArgumentException(((("Provided array of length "+value.Length)+" when length ")+1000)+" was expected for variable \'matchTime\'");
				}
				this.MatchTime = value;
				this.numberOfIterationsDone = 0;
				this.Changed_matchTime_isDone = false;
				this.Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isInitialised = false;
				this.Changed_matchTime_numberOfIterations_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isDone = false;
			}
		}

		/// <summary>The number of iterations done from the initial state</summary>
		public int NumberOfIterationsDone
		{
			get {
				return this.numberOfIterationsDone;
			}
		}

		/// <summary>The externally-specified value of 'PlayerIndex'</summary>
		public int[][][] PlayerIndex
		{
			get {
				return this.playerIndex;
			}
			set {
				if ((value!=null)&&(value.Length!=1000)) {
					throw new ArgumentException(((("Provided array of length "+value.Length)+" when length ")+1000)+" was expected for variable \'PlayerIndex\'");
				}
				this.playerIndex = value;
				this.numberOfIterationsDone = 0;
				this.Changed_PlayerIndex_isDone = false;
				this.Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isInitialised = false;
				this.Changed_matchTime_numberOfIterations_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isDone = false;
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
				this.Changed_PlayerInTeam_playerTime_isDone = false;
				this.Changed_PlayerInTeam_isDone = false;
				this.Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isInitialised = false;
				this.Changed_matchTime_numberOfIterations_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isDone = false;
			}
		}

		/// <summary>The externally-specified value of 'playerTime'</summary>
		public double[][][] playerTime
		{
			get {
				return this.PlayerTime;
			}
			set {
				if ((value!=null)&&(value.Length!=1000)) {
					throw new ArgumentException(((("Provided array of length "+value.Length)+" when length ")+1000)+" was expected for variable \'playerTime\'");
				}
				this.PlayerTime = value;
				this.numberOfIterationsDone = 0;
				this.Changed_PlayerInTeam_playerTime_isDone = false;
				this.Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isInitialised = false;
				this.Changed_matchTime_numberOfIterations_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isDone = false;
			}
		}

		/// <summary>The externally-specified value of 'vdouble__345'</summary>
		public double[] vdouble__345
		{
			get {
				return this.Vdouble__345;
			}
			set {
				if ((value!=null)&&(value.Length!=6523)) {
					throw new ArgumentException(((("Provided array of length "+value.Length)+" when length ")+6523)+" was expected for variable \'vdouble__345\'");
				}
				this.Vdouble__345 = value;
				this.numberOfIterationsDone = 0;
				this.Changed_vdouble__345_isDone = false;
				this.Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isInitialised = false;
				this.Changed_matchTime_numberOfIterations_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isDone = false;
			}
		}

		#endregion

		#region Methods
		/// <summary>Computations that depend on the observed value of matchTime</summary>
		private void Changed_matchTime()
		{
			if (this.Changed_matchTime_isDone) {
				return ;
			}
			// Create array for 'matchTime_marginal' Forwards messages.
			this.matchTime_marginal_F = new DistributionStructArray<Gaussian,double>(1000);
			for(int nMatches = 0; nMatches<1000; nMatches++) {
				this.matchTime_marginal_F[nMatches] = Gaussian.Uniform();
			}
			// Message to 'matchTime_marginal' from DerivedVariable factor
			this.matchTime_marginal_F = DerivedVariableOp.MarginalAverageConditional<DistributionStructArray<Gaussian,double>,double[]>(this.MatchTime, this.matchTime_marginal_F);
			this.Changed_matchTime_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of matchTime and numberOfIterations and PlayerIndex and PlayerInTeam and playerTime and vdouble__345</summary>
		/// <param name="numberOfIterations">The number of times to iterate each loop</param>
		private void Changed_matchTime_numberOfIterations_PlayerIndex_PlayerInTeam_playerTime_vdouble__345(int numberOfIterations)
		{
			if (this.Changed_matchTime_numberOfIterations_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isDone) {
				return ;
			}
			DistributionStructArray<Gaussian,double> vdouble__347_F;
			// Create array for 'vdouble__347' Forwards messages.
			vdouble__347_F = new DistributionStructArray<Gaussian,double>(1000);
			for(int nMatches = 0; nMatches<1000; nMatches++) {
				vdouble__347_F[nMatches] = Gaussian.Uniform();
			}
			// Create array for replicates of 'vdouble____102_F'
			DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>[] vdouble____102_F = new DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>[1000];
			// Create array for replicates of 'vdouble1186_F'
			Gaussian[][][] vdouble1186_F = new Gaussian[1000][][];
			// Create array for replicates of 'vdouble1188_F'
			Gaussian[][][] vdouble1188_F = new Gaussian[1000][][];
			// Create array for replicates of 'vdouble1193_F'
			Gaussian[] vdouble1193_F = new Gaussian[1000];
			// Create array for replicates of 'vdouble1192_F'
			Gaussian[] vdouble1192_F = new Gaussian[1000];
			for(int nMatches = 0; nMatches<1000; nMatches++) {
				// Create array for 'vdouble____102' Forwards messages.
				vdouble____102_F[nMatches] = new DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>(2);
				// Create array for replicates of 'vdouble1186_F'
				vdouble1186_F[nMatches] = new Gaussian[2][];
				// Create array for replicates of 'vdouble1188_F'
				vdouble1188_F[nMatches] = new Gaussian[2][];
				for(int nTeamsPerMatch = 0; nTeamsPerMatch<2; nTeamsPerMatch++) {
					// Create array for 'vdouble____102' Forwards messages.
					vdouble____102_F[nMatches][nTeamsPerMatch] = new DistributionStructArray<Gaussian,double>(this.playerInTeam[nMatches][nTeamsPerMatch]);
					// Create array for replicates of 'vdouble1186_F'
					vdouble1186_F[nMatches][nTeamsPerMatch] = new Gaussian[this.playerInTeam[nMatches][nTeamsPerMatch]];
					// Create array for replicates of 'vdouble1188_F'
					vdouble1188_F[nMatches][nTeamsPerMatch] = new Gaussian[this.playerInTeam[nMatches][nTeamsPerMatch]];
					for(int nPlayersMinusPerTeam = 0; nPlayersMinusPerTeam<this.playerInTeam[nMatches][nTeamsPerMatch]; nPlayersMinusPerTeam++) {
						vdouble____102_F[nMatches][nTeamsPerMatch][nPlayersMinusPerTeam] = Gaussian.Uniform();
						vdouble1186_F[nMatches][nTeamsPerMatch][nPlayersMinusPerTeam] = Gaussian.Uniform();
						// Message to 'vdouble1186' from Gaussian factor
						vdouble1186_F[nMatches][nTeamsPerMatch][nPlayersMinusPerTeam] = GaussianOpBase.SampleAverageConditional(this.Vdouble__345[this.playerIndex[nMatches][nTeamsPerMatch][nPlayersMinusPerTeam]], 1.0);
						vdouble1188_F[nMatches][nTeamsPerMatch][nPlayersMinusPerTeam] = Gaussian.Uniform();
						// Message to 'vdouble1188' from Product factor
						vdouble1188_F[nMatches][nTeamsPerMatch][nPlayersMinusPerTeam] = GaussianProductOpBase.ProductAverageConditional(vdouble1186_F[nMatches][nTeamsPerMatch][nPlayersMinusPerTeam], this.PlayerTime[nMatches][nTeamsPerMatch][nPlayersMinusPerTeam]);
						// Message to 'vdouble____102' from Ratio factor
						vdouble____102_F[nMatches][nTeamsPerMatch][nPlayersMinusPerTeam] = RatioGaussianOp.RatioAverageConditional(vdouble1188_F[nMatches][nTeamsPerMatch][nPlayersMinusPerTeam], this.MatchTime[nMatches]);
					}
				}
				vdouble1192_F[nMatches] = Gaussian.Uniform();
				// Message to 'vdouble1192' from Sum factor
				vdouble1192_F[nMatches] = FastSumOp.SumAverageConditional(vdouble____102_F[nMatches][0]);
				vdouble1193_F[nMatches] = Gaussian.Uniform();
				// Message to 'vdouble1193' from Sum factor
				vdouble1193_F[nMatches] = FastSumOp.SumAverageConditional(vdouble____102_F[nMatches][1]);
				// Message to 'vdouble__347' from Difference factor
				vdouble__347_F[nMatches] = Tracing.FireEvent<Gaussian>(DoublePlusOp.AAverageConditional(vdouble1192_F[nMatches], vdouble1193_F[nMatches]), string.Format("vdouble__347_F[{0}]", new object[1] {nMatches}), this.OnMessageUpdated, false);
			}
			Gaussian vdouble__347_use_B_reduced;
			vdouble__347_use_B_reduced = Gaussian.Uniform();
			for(int iteration = this.numberOfIterationsDone; iteration<numberOfIterations; iteration++) {
				for(int nMatches = 0; nMatches<1000; nMatches++) {
					this.vdouble__347_use_B_nMatches__buffer[nMatches] = VariablePointOp_Rprop.Buffer(vdouble__347_use_B_reduced, vdouble__347_F[nMatches], this.vdouble__347_marginal_F[nMatches], this.vdouble__347_use_B_nMatches__buffer[nMatches]);
					// Message to 'vdouble__347_marginal' from VariablePoint factor
					this.vdouble__347_marginal_F[nMatches] = Tracing.FireEvent<Gaussian>(VariablePointOp_Rprop.MarginalAverageConditional(vdouble__347_use_B_reduced, vdouble__347_F[nMatches], this.vdouble__347_use_B_nMatches__buffer[nMatches], this.vdouble__347_marginal_F[nMatches]), string.Format("vdouble__347_marginal_F[{0}]", new object[1] {nMatches}), this.OnMessageUpdated, false);
				}
				this.OnProgressChanged(new ProgressChangedEventArgs(iteration));
			}
			this.Changed_matchTime_numberOfIterations_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of numberOfIterationsDecreased and must reset on changes to matchTime and PlayerIndex and PlayerInTeam and playerTime and vdouble__345</summary>
		/// <param name="initialise">If true, reset messages that initialise loops</param>
		private void Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345(bool initialise)
		{
			if (this.Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isDone&&((!initialise)||this.Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isInitialised)) {
				return ;
			}
			for(int nMatches = 0; nMatches<1000; nMatches++) {
				this.vdouble__347_marginal_F[nMatches] = Gaussian.Uniform();
				this.vdouble__347_use_B_nMatches__buffer[nMatches] = VariablePointOp_Rprop.BufferInit();
			}
			this.Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isDone = true;
			this.Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isInitialised = true;
		}

		/// <summary>Computations that depend on the observed value of PlayerIndex</summary>
		private void Changed_PlayerIndex()
		{
			if (this.Changed_PlayerIndex_isDone) {
				return ;
			}
			// Create array for 'PlayerIndex_marginal' Forwards messages.
			this.PlayerIndex_marginal_F = new PointMass<int[][][]>(this.playerIndex);
			// Message to 'PlayerIndex_marginal' from DerivedVariable factor
			this.PlayerIndex_marginal_F = DerivedVariableOp.MarginalAverageConditional<PointMass<int[][][]>,int[][][]>(this.playerIndex, this.PlayerIndex_marginal_F);
			this.Changed_PlayerIndex_isDone = true;
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

		/// <summary>Computations that depend on the observed value of PlayerInTeam and playerTime</summary>
		private void Changed_PlayerInTeam_playerTime()
		{
			if (this.Changed_PlayerInTeam_playerTime_isDone) {
				return ;
			}
			// Create array for 'playerTime_marginal' Forwards messages.
			this.playerTime_marginal_F = new DistributionRefArray<DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>,double[][]>(1000);
			for(int nMatches = 0; nMatches<1000; nMatches++) {
				// Create array for 'playerTime_marginal' Forwards messages.
				this.playerTime_marginal_F[nMatches] = new DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>(2);
				for(int nTeamsPerMatch = 0; nTeamsPerMatch<2; nTeamsPerMatch++) {
					// Create array for 'playerTime_marginal' Forwards messages.
					this.playerTime_marginal_F[nMatches][nTeamsPerMatch] = new DistributionStructArray<Gaussian,double>(this.playerInTeam[nMatches][nTeamsPerMatch]);
					for(int nPlayersMinusPerTeam = 0; nPlayersMinusPerTeam<this.playerInTeam[nMatches][nTeamsPerMatch]; nPlayersMinusPerTeam++) {
						this.playerTime_marginal_F[nMatches][nTeamsPerMatch][nPlayersMinusPerTeam] = Gaussian.Uniform();
					}
				}
			}
			// Message to 'playerTime_marginal' from DerivedVariable factor
			this.playerTime_marginal_F = DerivedVariableOp.MarginalAverageConditional<DistributionRefArray<DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>,double[][]>,double[][][]>(this.PlayerTime, this.playerTime_marginal_F);
			this.Changed_PlayerInTeam_playerTime_isDone = true;
		}

		/// <summary>Computations that depend on the observed value of vdouble__345</summary>
		private void Changed_vdouble__345()
		{
			if (this.Changed_vdouble__345_isDone) {
				return ;
			}
			// Create array for 'vdouble__345_marginal' Forwards messages.
			this.vdouble__345_marginal_F = new DistributionStructArray<Gaussian,double>(6523);
			for(int nPlayers = 0; nPlayers<6523; nPlayers++) {
				this.vdouble__345_marginal_F[nPlayers] = Gaussian.Uniform();
			}
			// Message to 'vdouble__345_marginal' from DerivedVariable factor
			this.vdouble__345_marginal_F = DerivedVariableOp.MarginalAverageConditional<DistributionStructArray<Gaussian,double>,double[]>(this.Vdouble__345, this.vdouble__345_marginal_F);
			this.Changed_vdouble__345_isDone = true;
		}

		/// <summary>Computations that do not depend on observed values</summary>
		private void Constant()
		{
			if (this.Constant_isDone) {
				return ;
			}
			// Create array for 'vdouble__347_marginal' Forwards messages.
			this.vdouble__347_marginal_F = new DistributionStructArray<Gaussian,double>(1000);
			// Create array for replicates of 'vdouble__347_use_B_nMatches__buffer'
			this.vdouble__347_use_B_nMatches__buffer = new RpropBufferData[1000];
			this.Constant_isDone = true;
		}

		/// <summary>Update all marginals, by iterating message passing the given number of times</summary>
		/// <param name="numberOfIterations">The number of times to iterate each loop</param>
		/// <param name="initialise">If true, messages that initialise loops are reset when observed values change</param>
		private void Execute(int numberOfIterations, bool initialise)
		{
			if (numberOfIterations!=this.numberOfIterationsDone) {
				if (numberOfIterations<this.numberOfIterationsDone) {
					this.numberOfIterationsDone = 0;
					this.Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isDone = false;
				}
				this.Changed_matchTime_numberOfIterations_PlayerIndex_PlayerInTeam_playerTime_vdouble__345_isDone = false;
			}
			this.Changed_vdouble__345();
			this.Changed_PlayerInTeam_playerTime();
			this.Changed_matchTime();
			this.Changed_PlayerIndex();
			this.Changed_PlayerInTeam();
			this.Constant();
			this.Changed_numberOfIterationsDecreased_Init_matchTime_PlayerIndex_PlayerInTeam_playerTime_vdouble__345(initialise);
			this.Changed_matchTime_numberOfIterations_PlayerIndex_PlayerInTeam_playerTime_vdouble__345(numberOfIterations);
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
			if (variableName=="PlayerIndex") {
				return this.PlayerIndex;
			}
			if (variableName=="matchTime") {
				return this.matchTime;
			}
			if (variableName=="playerTime") {
				return this.playerTime;
			}
			if (variableName=="vdouble__345") {
				return this.vdouble__345;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>Get the marginal distribution (computed up to this point) of a variable</summary>
		/// <param name="variableName">Name of the variable in the generated code</param>
		/// <returns>The marginal distribution computed up to this point</returns>
		/// <remarks>Execute, Update, or Reset must be called first to set the value of the marginal.</remarks>
		public object Marginal(string variableName)
		{
			if (variableName=="vdouble__345") {
				return this.Vdouble__345Marginal();
			}
			if (variableName=="playerTime") {
				return this.PlayerTimeMarginal();
			}
			if (variableName=="matchTime") {
				return this.MatchTimeMarginal();
			}
			if (variableName=="PlayerIndex") {
				return this.PlayerIndexMarginal();
			}
			if (variableName=="PlayerInTeam") {
				return this.PlayerInTeamMarginal();
			}
			if (variableName=="vdouble__347") {
				return this.Vdouble__347Marginal();
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
		/// Returns the marginal distribution for 'matchTime' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public DistributionStructArray<Gaussian,double> MatchTimeMarginal()
		{
			return this.matchTime_marginal_F;
		}

		private void OnMessageUpdated(MessageUpdatedEventArgs e)
		{
			// Make a temporary copy of the event to avoid a race condition
			// if the last subscriber unsubscribes immediately after the null check and before the event is raised.
			EventHandler<MessageUpdatedEventArgs> handler = this.MessageUpdated;
			if (handler!=null) {
				handler(this, e);
			}
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
		/// Returns the marginal distribution for 'PlayerIndex' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public PointMass<int[][][]> PlayerIndexMarginal()
		{
			return this.PlayerIndex_marginal_F;
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

		/// <summary>
		/// Returns the marginal distribution for 'playerTime' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public DistributionRefArray<DistributionRefArray<DistributionStructArray<Gaussian,double>,double[]>,double[][]> PlayerTimeMarginal()
		{
			return this.playerTime_marginal_F;
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
			if (variableName=="PlayerIndex") {
				this.PlayerIndex = (int[][][])value;
				return ;
			}
			if (variableName=="matchTime") {
				this.matchTime = (double[])value;
				return ;
			}
			if (variableName=="playerTime") {
				this.playerTime = (double[][][])value;
				return ;
			}
			if (variableName=="vdouble__345") {
				this.vdouble__345 = (double[])value;
				return ;
			}
			throw new ArgumentException("Not an observed variable name: "+variableName);
		}

		/// <summary>Update all marginals, by iterating message-passing an additional number of times</summary>
		/// <param name="additionalIterations">The number of iterations that should be executed, starting from the current message state.  Messages are not reset, even if observed values have changed.</param>
		public void Update(int additionalIterations)
		{
			this.Execute(checked(this.numberOfIterationsDone+additionalIterations), false);
		}

		/// <summary>
		/// Returns the marginal distribution for 'vdouble__345' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public DistributionStructArray<Gaussian,double> Vdouble__345Marginal()
		{
			return this.vdouble__345_marginal_F;
		}

		/// <summary>
		/// Returns the marginal distribution for 'vdouble__347' given by the current state of the
		/// message passing algorithm.
		/// </summary>
		/// <returns>The marginal distribution</returns>
		public DistributionStructArray<Gaussian,double> Vdouble__347Marginal()
		{
			return ArrayHelper.MakeCopy<DistributionStructArray<Gaussian,double>>(this.vdouble__347_marginal_F);
		}

		#endregion

		#region Events
		/// <summary>Event that is fired when the progress of inference changes, typically at the end of one iteration of the inference algorithm.</summary>
		public event EventHandler<ProgressChangedEventArgs> ProgressChanged;
		/// <summary>Event that is fired when a message that is being monitored is updated.</summary>
		public event EventHandler<MessageUpdatedEventArgs> MessageUpdated;
		#endregion

	}

}
