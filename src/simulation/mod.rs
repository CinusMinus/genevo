
pub mod ga;

use chrono::{DateTime, Duration, Local};
use futures::{Future, Stream};
use genetic::{CrossoverOp, Fitness, FitnessEvaluation, Genotype, MutationOp, Phenotype,
                      Population, Breeding, SelectionOp};
use std::marker::PhantomData;

/// A `Simulation` is the execution of a genetic algorithm.
pub trait Simulation<'a, T, G, F, E, S, Q, C, M, P>
    where T: 'a + Phenotype<G>, G: Genotype, F: Fitness, P: Breeding<G>,
          E: FitnessEvaluation<G, F>, S: SelectionOp<T, G, P>, Q: Termination<'a, T, G, F>,
          C: CrossoverOp<P, G>, M: MutationOp<G>
{
    /// Start building a new instance of a `Simulation`.
    fn builder<B>(evaluator: E, selector: S, breeder: C, mutator: M, termination: Vec<Q>) -> B
        where B: SimulationBuilder<'a, Self, T, G, F, E, S, Q, C, M, P>, Self: Sized;

    /// Runs this simulation completely.
    fn run(&mut self) -> Future<Item=SimResult<'a, T, G, F>, Error=SimError>;

    /// Makes one step in this simulation.
    fn step(&mut self) -> Future<Item=SimResult<'a, T, G, F>, Error=SimError>;

    /// Runs the simulation while streaming the results of each step.
    /// The simulation runs without stopping after each step but the
    /// results of each step are provided as a `Stream`.
    fn stream(&mut self) -> Stream<Item=SimResult<'a, T, G, F>, Error=SimError>;

    /// Resets the simulation to rerun it again. This methods resets the
    /// simulation in its initial state, as if its just newly created.
    fn reset(&mut self);

    //TODO should we have statistics? what should they be?
    // Returns the `SimStatistics` of the last run of the simulation.
//    fn statistics(&self) -> SimStatistics;
}

/// The `SimulationBuilder` creates a new `Simulation` with given parameters
/// and options.
pub trait SimulationBuilder<'a, Sim, T, G, F, E, S, Q, C, M, P>
    where Sim: Simulation<'a, T, G, F, E, S, Q, C, M, P>,
          T: 'a + Phenotype<G>, G: Genotype, F: Fitness, P: Breeding<G>,
          E: FitnessEvaluation<G, F>, S: SelectionOp<T, G, P>, Q: Termination<'a, T, G, F>,
          C: CrossoverOp<P, G>, M: MutationOp<G>
{
    /// Finally initializes the `Simulation` with the given `Population`
    /// and returns the newly created `Simulation`.
    ///
    /// Note: This operation is made the last operation in the chain of
    /// configuration option methods to be able to reuse a previously
    /// configured `SimulationBuilder` with a different initial population.
    fn initialize(&self, population: Population<T, G>) -> Sim;
}

/// A `PopulationGenerator` creates a new `Population` with a number of newly
/// created individuals or just individual `Phenotype`s.
///
/// Typically the `PopulationGenerator` is used to create the initial
/// population with randomly created individuals.
pub trait PopulationGenerator<T, G>
    where T: Phenotype<G>, G: Genotype
{
    /// Generates a new `Population` containing the given number of individuals.
    fn generate_population(&self, size: usize) -> Population<T, G> {
        let individuals = (0..size).map(|_| {
            self.generate_phenotype()
        }).collect::<Vec<T>>();
        Population::new(individuals)
    }

    /// Generates a new `Phenotype`.
    ///
    /// An implementation typically generates a randomly created `Phenotype`.
    fn generate_phenotype(&self) -> T;
}

/// A `Termination` defines a condition when the `Simulation` shall stop.
/// Common termination conditions are:
/// * A solution is found that satisfies minimum criteria
/// * A fixed number of generations is reached
/// * An allocated budget (computation time/money) is reached
/// * The highest ranking solution's fitness is reaching or has reached a
///   plateau such that successive iterations no longer produce better results
/// ...or a combination of termination conditions.
pub trait Termination<'a, T, G, F>
    where T: 'a + Phenotype<G>, G: Genotype, F: Fitness
{
    /// Evaluates whether the termination condition is met and returns true
    /// if the simulation shall be stopped or false if it shall continue.
    fn evaluate(&state: SimState<'a, T, G, F>) -> bool;
}

#[derive(Debug, Eq, PartialEq)]
pub struct SimState<'a, T, G, F>
    where T: 'a + Phenotype<G>, G: Genotype, F: Fitness
{
    /// The local time when this simulation started.
    started_at: DateTime<Local>,
    /// The number of the generation currently evaluated.
    generation: u64,
    /// Time spent for the current generation.
    time: Duration,
    /// Average fitness value of the current generation.
    average_fitness: F,
    /// Best solution of this generation.
    best_solution: BestSolution<'a, T, G, F>
}

/// The best solution found by the `Simulation`. If the simulation is not
/// finished this is the best solution of the generation currently evaluated.
/// If the solution is finished this is the overall best solution found by the
/// simulation.
#[derive(Debug, Eq, PartialEq)]
pub struct BestSolution<'a, T, G, F>
    where T: 'a + Phenotype<G>, G: Genotype, F: Fitness
{
    /// The local time at which this solution is found.
    found_at: DateTime<Local>,
    /// The number of the generation in which this solution is found.
    generation: u64,
    /// The `Fitness` value of this solution which is considered to be best
    /// so far.
    fitness: F,
    /// The `Phenotype` that is considered to be best so far.
    best_solution: &'a T,
    // Needed to calm down the compiler ;-)
    phantom_type: PhantomData<G>,
}

/// The result of running a step in the `Simulation`.
#[derive(PartialEq, Eq, Debug)]
pub enum SimResult<'a, T, G, F>
    where T: 'a + Phenotype<G>, G: Genotype, F: Fitness
{
    /// The step was successful, but the simulation has not finished.
    Intermediate(SimState<'a, T, G, F>),
    /// The simulation is finished, and this is the final result.
    ///
    /// The `BestSolution` value represents the fittest individual
    /// found during this simulation over all generations.
    Final(BestSolution<'a, T, G, F>),
}

/// An error occurred during `Simulation`.
pub enum SimError<'a> {
    /// The simulation has been created with an empty population.
    EmptyPopulation(&'a str),
    /// It has been tried to call run, step or stream while the simulation
    /// is already running. E.g. the step method has been called and now step,
    /// run or stream is called before the simulation of the previous step is
    /// finished.
    SimulationAlreadyRunning(&'a str),
}
