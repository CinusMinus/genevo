use crate::{
    algorithm::EvaluatedPopulation,
    genetic::{Fitness, FitnessFunction, Genotype, Offspring},
    operator::{GeneticOperator, MultiObjective, ReinsertionOp, SingleObjective},
    random::Prng,
};
use std::marker::PhantomData;
use rayon::slice::ParallelSliceMut;
use rayon::iter::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator};


#[derive(Clone, Debug, PartialEq)]
pub struct ParallelElitistReinserter<G, F, E>
    where
        G: Genotype,
        F: Fitness,
        E: FitnessFunction<G, F>,
{
    /// The `FitnessFunction` to be used to calculate fitness values of
    /// individuals of the offspring.
    fitness_evaluator: Box<E>,
    /// `offspring_has_precedence` defines whether individuals from offspring
    /// with lower fitness should possible replace better performing ones from
    /// the old population.
    offspring_has_precedence: bool,
    /// The `replace_ratio` defines the fraction of the population size that
    /// is going to be replaced by individuals from the offspring.
    replace_ratio: f64,
    // phantom types
    _g: PhantomData<G>,
    _f: PhantomData<F>,
}

impl<G, F, E> ParallelElitistReinserter<G, F, E>
    where
        G: Genotype,
        F: Fitness,
        E: FitnessFunction<G, F>,
{
    /// Constructs a new instance of the `ElitistReinserter`.
    pub fn new(fitness_evaluator: E, offspring_has_precedence: bool, replace_ratio: f64) -> Self {
        Self {
            fitness_evaluator: Box::new(fitness_evaluator),
            offspring_has_precedence,
            replace_ratio,
            _g: PhantomData,
            _f: PhantomData,
        }
    }

    /// Returns true if the offspring should take precedence over better
    /// performing individuals from the old population.
    pub fn is_offspring_has_precedence(&self) -> bool {
        self.offspring_has_precedence
    }

    /// Sets whether the offspring should have precedence over better
    /// performing individuals from the old population.
    pub fn set_offspring_has_precedence(&mut self, value: bool) {
        self.offspring_has_precedence = value;
    }

    /// Returns the `replace_ratio` of this `ElitistReinserter`.
    pub fn replace_ratio(&self) -> f64 {
        self.replace_ratio
    }

    /// Set the `replace_ratio` of this `ElitistReinserter` to the given
    /// value. The value must be between 0 and 1.0 (inclusive).
    pub fn set_replace_ratio(&mut self, value: f64) {
        self.replace_ratio = value;
    }
}

/// Can be used for single-objective optimization
impl<G, F, E> SingleObjective for ParallelElitistReinserter<G, F, E>
    where
        G: Genotype,
        F: Fitness,
        E: FitnessFunction<G, F>,
{
}
/// Can be used for multi-objective optimization
impl<G, F, E> MultiObjective for ParallelElitistReinserter<G, F, E>
    where
        G: Genotype,
        F: Fitness,
        E: FitnessFunction<G, F>,
{
}


impl<G, F, E> GeneticOperator for ParallelElitistReinserter<G, F, E>
    where
        G: Genotype,
        F: Fitness,
        E: FitnessFunction<G, F>,
{
    fn name(&self) -> String {
        "Elitist-Reinserter".to_string()
    }
}


impl<G, F, E> ReinsertionOp<G, F> for ParallelElitistReinserter<G, F, E>
    where
        G: Genotype,
        F: Fitness + Send + Sync,
        E: FitnessFunction<G, F> + Sync + Send,
{
    fn combine(&self, offspring: &mut Offspring<G>, population: &EvaluatedPopulation<G, F>, _rng: &mut Prng) -> Vec<G> {
        let old_individuals = population.individuals();
        let old_fitness = population.fitness_values();
        let mut old_indices: Vec<_> = (0..old_fitness.len()).into_iter().collect();
        old_indices.par_sort_unstable_by(|x, y| old_fitness[*x].cmp(&old_fitness[*y]).reverse());
        let population_size = old_individuals.len();
        let mut new_population = Vec::with_capacity(population_size);

        let num_offspring = (population_size as f64 * self.replace_ratio + 0.5).floor() as usize;
        //evaluate offspring fitness
        let mut offspring_fitness = Vec::with_capacity(offspring.len());
        offspring.into_par_iter().map(|x| {let fitness = self.fitness_evaluator.fitness_of(&x); (x, fitness)}).collect_into_vec(&mut offspring_fitness);
        offspring_fitness.par_sort_unstable_by(|x, y| x.1.cmp(&y.1).reverse());

        if self.offspring_has_precedence {
            let old = old_indices.into_par_iter().map(|i| old_individuals[i].clone());
            let new = offspring_fitness.into_par_iter().take(num_offspring).map(|(o,_)| o.clone());
            new.chain(old).take(population_size).collect_into_vec(&mut new_population);
        } else {
            todo!()
        }

        new_population
    }
}