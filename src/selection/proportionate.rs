//! The `proportionate` module provides `operator::SelectionOp`s that implement
//! stochastic fitness proportionate selection strategies. Individuals are
//! randomly selected. Individuals with a higher `genetic::Fitness` value are
//! having a higher probability to be selected.
//!
//! How is this achieved?
//! In fitness proportionate selection each individual gets assigned a weight
//! that is equal to its fitness value plus the sum of fitness values of all
//! individuals in the list before him (cumulative weight distribution). Then
//! a uniform random number between 0 and the sum of all weights is used to
//! select a candidate.
//!
//! The provided **fitness proportionate selection** operators are:
//! * `RouletteWheelSelector` - no bias - does not guarantee minimal spread.
//! * `UniversalSamplingSelector` - no bias - minimal spread.

use crate::{
    algorithm::EvaluatedPopulation,
    genetic::{AsScalar, Fitness, Genotype, Parents},
    operator::{GeneticOperator, SelectionOp, SingleObjective},
    random::{random_probability, WeightedDistribution, Prng},
};
use std::sync::{Arc, Mutex};
use rayon::iter::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator};
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use std::borrow::Borrow;

/// The `RouletteWheelSelector` implements stochastic fitness proportionate
/// selection. Each candidate is picked randomly with a probability of being
/// picked that is proportional to its fitness value.
///
/// Characteristics: no bias, does not guarantee minimal spread.
#[allow(missing_copy_implementations)]
#[derive(Clone, Debug, PartialEq)]
pub struct RouletteWheelSelector {
    /// The fraction of number of parents to select in relation to the
    /// number of individuals in the population.
    selection_ratio: f64,
    /// The number of individuals per parents.
    num_individuals_per_parents: usize,
}

impl RouletteWheelSelector {
    /// Constructs a new instance of `RouletteWheelSelector`.
    pub fn new(selection_ratio: f64, num_individuals_per_parents: usize) -> Self {
        RouletteWheelSelector {
            selection_ratio,
            num_individuals_per_parents,
        }
    }

    /// Returns the selection ratio.
    ///
    /// The selection ratio is the fraction of number of parents that are
    /// selected on every call of the `select_from` function and the number
    /// of individuals in the population.
    pub fn selection_ratio(&self) -> f64 {
        self.selection_ratio
    }

    /// Sets the selection ratio to a new value.
    ///
    /// The selection ratio is the fraction of number of parents that are
    /// selected on every call of the `select_from` function and the number
    /// of individuals in the population.
    pub fn set_selection_ratio(&mut self, value: f64) {
        self.selection_ratio = value;
    }

    /// Returns the number of individuals per parents use by this selector.
    pub fn num_individuals_per_parents(&self) -> usize {
        self.num_individuals_per_parents
    }

    /// Sets the number of individuals per parents to the given value.
    pub fn set_num_individuals_per_parents(&mut self, value: usize) {
        self.num_individuals_per_parents = value;
    }
}

impl SingleObjective for RouletteWheelSelector {}

impl GeneticOperator for RouletteWheelSelector {
    fn name(&self) -> String {
        "Roulette-Wheel-Selection".to_string()
    }
}

impl<G, F> SelectionOp<G, F> for RouletteWheelSelector
where
    G: Genotype,
    F: Fitness + AsScalar + Default + Sync,
{
    fn select_from(&self, evaluated: &EvaluatedPopulation<G, F>, rng: &mut Prng) -> Vec<Parents<G>> {
        let rc_individuals = evaluated.individuals();
        let individuals: &Vec<_> = rc_individuals.borrow();
        let num_parents_to_select =
            (individuals.len() as f64 * self.selection_ratio + 0.5).floor() as usize;
        let mut parents: Vec<Parents<G>> = Vec::with_capacity(num_parents_to_select);
        let weighted_distribution = WeightedIndex::new(evaluated.fitness_values().iter().map(|x| x.as_scalar())).unwrap();
            //WeightedDistribution::from_scalar_values(evaluated.fitness_values());
        {
            let arc = Arc::new(Mutex::new(rng));
            (0..num_parents_to_select).into_par_iter().map_init(|| {
                let mut rng = arc.lock().unwrap();
                let rng1 = rng.clone();
                rng.jump();
                rng1
            }, |rng, _| {
                let mut tuple = Vec::with_capacity(self.num_individuals_per_parents);
                for _ in 0..self.num_individuals_per_parents {
                    let selected = weighted_distribution.sample(rng);
                    tuple.push(individuals[selected].clone());
                }
                tuple
            }).collect_into_vec(&mut parents);
        }
        parents
    }
}

/// The `UniversalSamplingSelector` implements stochastic fitness proportionate
/// selection. The first candidate is picked randomly. All other candidates are
/// picked by equidistant jumps.
///
/// Characteristics: no bias, minimal spread.
#[allow(missing_copy_implementations)]
#[derive(Clone, Debug, PartialEq)]
pub struct UniversalSamplingSelector {
    /// The fraction of number of parents to select in relation to the
    /// number of individuals in the population.
    selection_ratio: f64,
    /// The number of individuals per parents.
    num_individuals_per_parents: usize,
}

impl UniversalSamplingSelector {
    /// Constructs a new instance of `UniversalSamplingSelector`.
    pub fn new(selection_ratio: f64, num_individuals_per_parents: usize) -> Self {
        UniversalSamplingSelector {
            selection_ratio,
            num_individuals_per_parents,
        }
    }

    /// Returns the selection ratio.
    ///
    /// The selection ratio is the fraction of number of parents that are
    /// selected on every call of the `select_from` function and the number
    /// of individuals in the population.
    pub fn selection_ratio(&self) -> f64 {
        self.selection_ratio
    }

    /// Sets the selection ratio to a new value.
    ///
    /// The selection ratio is the fraction of number of parents that are
    /// selected on every call of the `select_from` function and the number
    /// of individuals in the population.
    pub fn set_selection_ratio(&mut self, value: f64) {
        self.selection_ratio = value;
    }

    /// Returns the number of individuals per parents use by this selector.
    pub fn num_individuals_per_parents(&self) -> usize {
        self.num_individuals_per_parents
    }

    /// Sets the number of individuals per parents to the given value.
    pub fn set_num_individuals_per_parents(&mut self, value: usize) {
        self.num_individuals_per_parents = value;
    }
}

impl SingleObjective for UniversalSamplingSelector {}

impl GeneticOperator for UniversalSamplingSelector {
    fn name(&self) -> String {
        "Stochastic-Universal-Sampling-Selection".to_string()
    }
}

impl<G, F> SelectionOp<G, F> for UniversalSamplingSelector
where
    G: Genotype,
    F: Fitness + AsScalar,
{
    fn select_from(&self, evaluated: &EvaluatedPopulation<G, F>, rng: &mut Prng) -> Vec<Parents<G>> {
        let individuals = evaluated.individuals();
        let num_parents_to_select =
            (individuals.len() as f64 * self.selection_ratio + 0.5).floor() as usize;
        let mut parents = Vec::with_capacity(num_parents_to_select);
        let weighted_distribution =
            WeightedDistribution::from_scalar_values(evaluated.fitness_values());
        let distance = weighted_distribution.sum()
            / (num_parents_to_select * self.num_individuals_per_parents) as f64;
        let mut pointer = random_probability(rng) * weighted_distribution.sum();
        for _ in 0..num_parents_to_select {
            let mut tuple = Vec::with_capacity(self.num_individuals_per_parents);
            for _ in 0..self.num_individuals_per_parents {
                let selected = weighted_distribution.select(pointer);
                tuple.push(individuals[selected].clone());
                pointer += distance;
            }
            parents.push(tuple);
        }
        parents
    }
}
