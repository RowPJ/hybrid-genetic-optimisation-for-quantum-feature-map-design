# This is the main file to load when running experiments.
# It loads most/all of the other files and contains the high level
# functions that depend on the other parts of the code.

include("datasets.jl")
include("population_evaluators.jl")
include("worker_management.jl")
include("nsga2.jl")
include("parameter_optimization.jl")
include("graphing.jl")
include("tests.jl")

using PyCall

# TODO: add additional genetic termination criteria such as minimum fitness
# or rate of improvement. look at the reference paper for options they list.
# this isn't a requirement but would conveniently make the experiments
# run faster. it could be useful in analysis to show which methods
# converge sooner. alternatively implementing objective tracking could be
# used for convergence analysis

#TODO: track how the genetic objective functions change
# over generations for convergence analysis. return the
# history of objective values as another result

#TODO: for datasets where the number of samples used in training is purposely limited to
# improve runtime, the remaining samples can be used for validation of the final kernels.
# A more general principle is to have a training and test set split before the optimization
# process to determine whether the trained kernels can classify data not involved in their
# training. this validation could also use k-fold cross validation, the kernels wouldn't have
# to be trained multiple times but the validation could be done with multiple ways of splitting
# the final test set.
# Example diagram of how this would work:
#|#####################################Full data set###############################################|
#|##Genetic training data (limited for run speed)#|#####Final test data (for validating results)###|
#|######train data###########|######test data#####|###(Used in C.V. tests after genetic training)##|
#NOTE: with cross validation in genetic training, the above diagram is the same except the
# genetic training data is split in multiple ways

# define python functions to interface with pymoo
# and perform the genetic optimization
py"""
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from sklearn.datasets import make_moons, load_iris, load_breast_cancer, load_digits
from sklearn.decomposition import PCA
from pymoo.optimize import minimize
from pymoo.factory import get_termination
from pymoo.core.mutation import Mutation # to implement custom mutation
from pymoo.visualization.scatter import Scatter  # use to draw solutions on a scatter plot when using 2 objective functions
import numpy as np

#TODO: check that bitflip mutation probability actually does what it should
# The probability of a mutation occuring should be 70%, and the probability
# for each individual bit to flip should be 20%

# TODO: print out the best fitness value in the population
# generated so far (can do in generation advancing loop)

# TODO: consider adding a classical data encoding function to
# the circuits. each set of gate bits could have an additional
# bit determining whether or not the encoding function is used
# or the data is encoded without one.


def load_checkpoint(name):
    return np.load(name + ".npy", allow_pickle=True).flatten()[0]


# pymoo doesn't have a mutation operator that operates like the
# one in the reference paper, so a custom operator is implemented here
class CustomMutation(Mutation):
    # This class is based on the built in BinaryBitflipMutation
    # class provided by pymoo, but adjusted to match the reference
    # paper
    def __init__(self, prob=None, index_prob=None):
        assert(not (prob is None or index_prob is None))
        self.prob=prob
        self.index_prob=index_prob
    def _do(self, problem, X, **kwargs):
        # consider population as boolean array
        X = X.astype(bool)
        # create copy of population to put mutated values in
        _X = np.full(X.shape, np.inf)
        ## create random numbers to decide which rows (individuals) should mutate
        # create 1 random number for each row
        M = np.random.random(X.shape[0])
        # then expand it to a 2d array by repeating the number for each row
        M = np.array([[M[i]] * X.shape[1] for i in range(X.shape[0])])
        mutate_row = M < self.prob
        # create random numbers to decide which bits should flip
        B = np.random.random(X.shape)
        mutate_bit = B < self.index_prob

        # combine row and entry decisions to get which bits to change
        bits_to_mutate = np.logical_and(mutate_row, mutate_bit)
        bits_to_keep = np.logical_not(bits_to_mutate)
        
        # apply mutation
        _X[bits_to_mutate] = np.logical_not(X[bits_to_mutate])
        _X[bits_to_keep] = X[bits_to_keep]

        return _X.astype(bool)

# general version of experiment functions taking the pre-processed training data set in the arguments
def genetic_solve_dataset_classification(samples, labels, feature_count, population_evaluator, seed=22, qubit_count=6, depth=6):
    #NSGA2 defaults to tournament selection
    #and rank and crowding survival
    algorithm = NSGA2(pop_size=100,
                      n_offsprings=15,
                      sampling=get_sampling("bin_random"),
                      crossover=get_crossover("bin_two_point", prob=0.3),
                      mutation=CustomMutation(prob=0.7, index_prob=0.2), #get_mutation("bin_bitflip", prob=0.7),
                      eliminate_duplicates=True)
    problem = KernelCircuitProblem(evaluator=population_evaluator,
                                   feature_count=feature_count,
                                   qubit_count=qubit_count,
                                   depth=depth,
                                   problem_data=(samples,labels),
                                   seed=seed)
    termination = get_termination("n_gen", 1200)
    algorithm.setup(problem, seed=seed, termination=termination, verbose=True)
    fitness_history = []
    while algorithm.has_next():
        algorithm.next()
        fitness_history.append(algorithm.result().F)
    result = algorithm.result()
    
    solutions = result.X
    fitnesses = result.F
    return solutions, fitnesses, result, fitness_history
"""

"Generic experiment function."
function genetic_solve_dataset_classification(dataset, population_evaluator; qubit_count=6, depth=6, seed=22)
    population_matrix, fitness_matrix, pymoo_result, fitness_history = py"genetic_solve_dataset_classification"(dataset.training_samples,
                                                                                               dataset.training_labels,
                                                                                               dataset.feature_count,
                                                                                               population_evaluator,
                                                                                               depth=depth,
                                                                                               qubit_count=qubit_count,
                                                                                               seed=seed)
    row(m, i) = @view m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    return (to_rows(population_matrix), to_rows(fitness_matrix), pymoo_result, map(to_rows, fitness_history))
end

"Allows partially specifying arguments to a function, and returns a function that
takes the rest of the arguments and performs the function call."
function curry(fn, args...; kwargs...)
    function curried(rest_args...; rest_kwargs...)
        fn(args..., rest_args...; kwargs..., rest_kwargs...)
    end
    curried
end

"Returns a variant of the argument that times its executions."
function timed(fn)
    (args...; kwargs...) -> @time fn(args...; kwargs...)
end

"For a given data set struct, generates the functions that could be called to perform
the genetic circuit optimization experiments with each version of the accuracy metric
and its substitutes."
function generate_genetic_metric_variation_experiment_functions(dataset)
    acc_experiment = timed(curry(genetic_solve_dataset_classification, dataset, evaluate_population_yao))
    margin_experiment = timed(curry(genetic_solve_dataset_classification, dataset, evaluate_population_yao_margin_metric))
    cross_validation_experiment = timed(curry(genetic_solve_dataset_classification, dataset, evaluate_population_yao_cross_validation))
    acc_genetic_and_parameter_training_experiment = timed(curry(genetic_solve_dataset_classification, dataset, evaluate_population_yao_parameter_training_accuracy))
    return (acc_experiment, margin_experiment, cross_validation_experiment, acc_genetic_and_parameter_training_experiment)
end

# define experiment functions for the convenience of running specific experiments individually
solve_moons_accuracy, solve_moons_margin, solve_moons_cross_validation, solve_moons_genetic_and_parameter_training_accuracy = generate_genetic_metric_variation_experiment_functions(moons_dataset)
solve_cancer_accuracy, solve_cancer_margin, solve_cancer_cross_validation, solve_cancer_genetic_and_parameter_training_accuracy = generate_genetic_metric_variation_experiment_functions(cancer_dataset)
solve_iris_accuracy, solve_iris_margin, solve_iris_cross_validation, solve_iris_genetic_and_parameter_training_accuracy = generate_genetic_metric_variation_experiment_functions(iris_dataset)
solve_digits_accuracy, solve_digits_margin, solve_digits_cross_validation, solve_digits_genetic_and_parameter_training_accuracy = generate_genetic_metric_variation_experiment_functions(digits_dataset)


"Given a list of population individuals and a list of their corresponding fitness values,
returns the index of the best performing individual measured by their accuracy metric
or its substitute."
function best_individual_index(population, fitnesses)
    # get the highest-accuracy individual
    highest_accuracy = fitnesses[1][1]
    smallest_size = fitnesses[1][2]
    result_index = 1
    for i in 2:length(population)
        next_accuracy = fitnesses[i][1]
        next_size = fitnesses[i][2]
        # remember, use < to compare if an accuracy value is better, since
        # more negative fitness corresponds to higher accuracy as the accuracy
        # metrics were all negated to make the minimizing optimizer into a maximizer
        # Conditional check passes if accuracy is better or if accuracy is the same but size is better
        if next_accuracy < highest_accuracy || (next_accuracy == highest_accuracy && next_size < smallest_size)
            highest_accuracy = next_accuracy
            smallest_size = next_size
            result_index = i
        end
    end
    return result_index
end

#TODO: finish a generalised version of this function definition that works with an arbitrary data set
"Trains circuits genetically to have high accuracy, then replaces the genetically-determined
proportionality parameters with trainable real-valued parameters and trains them to maximise
kernel target alignment. Saves the genetically trained results as well as the parameter optimized
results."
function moons_parameterised_genetic_combination_experiment(;seed=22)
    row(m, i) = @view m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]

    # perform genetic optimization
    (population, fitnesses, pymoo_result, fit_history) = solve_moons_classification(;seed=seed)

    highest_accuracy_individual = population[best_individual_index(population, fitnesses)]

    #train its parameters to maximise kernel target alignment
    parameterised_kernel, initial_parameters = decode_chromosome_parameterised_yao(chromosome, 2, 6, 6) # for moons training, assuming 2 features, 6 qubits, and depth 6


    # generate the validation data set
    sample_count = 500
    samples, labels = make_moons(n_samples=sample_count,
                                 random_state=seed)
end

#TODO: 1. create a general function for testing a kernel on the validation part of a data set
# 2. create a dispatch version that takes a chromosome and calls the first version with its kernel
# 3. create a dispatch version that takes a chromosome and parameters and calls the first version with the parameters substituted into its kernel

function parameter_training_experiment(dataset::Dataset; qubit_count=6, depth=6, max_evaluations=60, seed=22, genetic_metric_type="accuracy", parameter_metric_type="accuracy")
    # load the genetic training results for the given data set
    population, fitnesses, genetic_fitness_histories = load_results(dataset.name, genetic_metric_type)
    # train the population with parameter based training
    population_optimized_parameters, population_parameter_objective_histories = population_parameterised_training(population,
                                                                                                        dataset;
                                                                                                        qubit_count=qubit_count,
                                                                                                        depth=depth,
                                                                                                        max_evaluations=max_evaluations,
                                                                                                        seed=seed,
                                                                                                        metric_type=parameter_metric_type)
    # return optimized parameters for use,
    # return genetic fitness histories for graphing,
    # and return parameter training objective histories for graphing
    return population_optimized_parameters, genetic_fitness_histories, population_parameter_objective_histories
end

# run a bunch of experiments, saving results
function main()
    
end
