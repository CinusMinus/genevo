//! The `tournament` module.
//!
//! The provided `SelectionOp` implementations are:
//! * `TournamentSelector`

use crate::{
    algorithm::EvaluatedPopulation,
    genetic::{Fitness, Genotype, Parents},
    operator::{GeneticOperator, MultiObjective, SelectionOp, SingleObjective},
    random::Prng,
};
use rand::seq::IteratorRandom;

/// The `TournamentSelector` implements the tournament selection method.
/// It runs tournaments with a small size of participants and picks the best
/// performing individual from each tournament.
///
/// The number of participants in each tournament is configurable by the `tournament_size` field.
/// A tournament size of 1 is called 1-way tournament and is equivalent to random selection.
///
/// To avoid that candidates chosen once are selected again they are removed
/// from the list of candidates. Though this can be configured as well. The
/// field `remove_selected_individuals` controls whether selected candidates
/// are removed or not.
///
/// This `TournamentSelector` can be used for single-objective fitness values
/// as well as multi-objective fitness values.
#[allow(missing_copy_implementations)]
#[derive(Clone, Debug, PartialEq)]
pub struct TournamentSelector {
    /// The fraction of number of parents to select in relation to the
    /// number of individuals in the population.
    selection_ratio: f64,
    /// The number of individuals per parents.
    num_individuals_per_parents: usize,
    /// The number of participants on each tournament.
    tournament_size: usize,
    /// Remove chosen individuals from the list of candidates to avoid that
    /// they can be picked again.
    remove_selected_individuals: bool,
}

impl TournamentSelector {
    /// Constructs a new instance of the `TournamentSelector`.
    pub fn new(
        selection_ratio: f64,
        num_individuals_per_parents: usize,
        tournament_size: usize,
        remove_selected_individuals: bool,
    ) -> Self {
        TournamentSelector {
            selection_ratio,
            num_individuals_per_parents,
            tournament_size,
            remove_selected_individuals,
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

    /// Returns the size of one tournament.
    pub fn tournament_size(&self) -> usize {
        self.tournament_size
    }

    /// Sets the size of one tournament to a given value. The value must be
    /// a positive integer greater 0.
    ///
    /// A tournament size of 1 is called 1-way tournament and is
    /// equivalent to random selection.
    pub fn set_tournament_size(&mut self, value: usize) {
        self.tournament_size = value;
    }

    /// Returns whether individuals are removed from the list of candidates
    /// after they have been picked once.
    pub fn is_remove_selected_individuals(&self) -> bool {
        self.remove_selected_individuals
    }

    /// Sets whether individuals shall be removed from the list of candidates
    /// after they have been picked once.
    pub fn set_remove_selected_individuals(&mut self, value: bool) {
        self.remove_selected_individuals = value;
    }
}

/// Can be used for single-objective optimization
impl SingleObjective for TournamentSelector {}
/// Can be used for multi-objective optimization
impl MultiObjective for TournamentSelector {}

impl GeneticOperator for TournamentSelector {
    fn name(&self) -> String {
        "Tournament-Selection".to_string()
    }
}

impl<G, F> SelectionOp<G, F> for TournamentSelector
where
    G: Genotype,
    F: Fitness,
{
    fn select_from(&self, evaluated: &EvaluatedPopulation<G, F>, rng: &mut Prng) -> Vec<Parents<G>> {
        let individuals = evaluated.individuals();
        let population_size = individuals.len();
        let fitness = evaluated.fitness_values();
        let num_parent_tuples = (population_size as f64 * self.selection_ratio).ceil() as usize;
        (0..num_parent_tuples).map(|_| {
            (0..self.num_individuals_per_parents).filter_map(|_| {
                (0..population_size)
                    .choose_multiple(rng, self.tournament_size)
                    .into_iter()
                    .max_by(|a, b| fitness[*a].cmp(&fitness[*b]))
                    .map(|n| individuals[n].clone())
            }).collect()
        }).collect()
    }
}
