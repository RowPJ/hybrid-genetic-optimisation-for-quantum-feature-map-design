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
def genetic_solve_dataset_classification(samples, labels, feature_count, population_evaluator, circuit_equality_test, seed=22, qubit_count=6, depth=6):
    duplicate_elimination = KernelCircuitDuplicateElimination(qubit_count, feature_count, circuit_equality_test)
    #NSGA2 defaults to tournament selection
    #and rank and crowding survival
    algorithm = NSGA2(pop_size=100,
                      n_offsprings=15,
                      sampling=get_sampling("bin_random"),
                      crossover=get_crossover("bin_two_point", prob=0.3),
                      mutation=CustomMutation(prob=0.7, index_prob=0.2), #get_mutation("bin_bitflip", prob=0.7),
                      eliminate_duplicates=duplicate_elimination)
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
                                                                                               circuit_equals, # function for checking 2 circuits for equality
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
    alignment_experiment = timed(curry(genetic_solve_dataset_classification, dataset, evaluate_population_yao_kernel_target_alignment))
    alignment_approximation_experiment = timed(curry(genetic_solve_dataset_classification, dataset, evaluate_population_yao_kernel_target_alignment_split))
    acc_genetic_and_parameter_training_experiment = timed(curry(genetic_solve_dataset_classification, dataset, evaluate_population_yao_parameter_training_rmse))
    acc_genetic_and_parameter_training_target_alignment_experiment = timed(curry(genetic_solve_dataset_classification, dataset, evaluate_population_yao_parameter_training_target_alignment))
    acc_dynamic_dataset_size_experiment = timed(curry(genetic_solve_dataset_classification, dataset, evaluate_population_yao_dynamic_dataset_size))
    
    return (acc_experiment, margin_experiment, cross_validation_experiment, alignment_experiment, alignment_approximation_experiment, acc_genetic_and_parameter_training_experiment, acc_genetic_and_parameter_training_target_alignment_experiment, acc_dynamic_dataset_size_experiment)
end

# define experiment functions for the convenience of running specific experiments individually
solve_moons_accuracy, solve_moons_margin, solve_moons_cross_validation, solve_moons_alignment, solve_moons_alignment_approximation, solve_moons_genetic_and_parameter_training_rmse, solve_moons_genetic_and_parameter_training_target_alignment, solve_moons_dynamic_dataset_size = generate_genetic_metric_variation_experiment_functions(moons_dataset)
solve_cancer_accuracy, solve_cancer_margin, solve_cancer_cross_validation, solve_cancer_alignment, solve_cancer_alignment_approximation, solve_cancer_genetic_and_parameter_training_rmse, solve_cancer_genetic_and_parameter_training_target_alignment, solve_cancer_dynamic_dataset_size = generate_genetic_metric_variation_experiment_functions(cancer_dataset)
solve_iris_accuracy, solve_iris_margin, solve_iris_cross_validation, solve_iris_alignment, solve_iris_alignment_approximation, solve_iris_genetic_and_parameter_training_rmse, solve_iris_genetic_and_parameter_training_target_alignment, solve_iris_dynamic_dataset_size = generate_genetic_metric_variation_experiment_functions(iris_dataset)
solve_digits_accuracy, solve_digits_margin, solve_digits_cross_validation, solve_digits_alignment, solve_digits_alignment_approximation, solve_digits_genetic_and_parameter_training_rmse, solve_digits_genetic_and_parameter_training_target_alignment, solve_digits_dynamic_dataset_size = generate_genetic_metric_variation_experiment_functions(digits_dataset)
solve_blobs_accuracy, solve_blobs_margin, solve_blobs_cross_validation, solve_blobs_alignment, solve_blobs_alignment_approximation, solve_blobs_genetic_and_parameter_training_rmse, solve_blobs_genetic_and_parameter_training_target_alignment, solve_blobs_dynamic_dataset_size = generate_genetic_metric_variation_experiment_functions(blobs_dataset)
solve_circles_accuracy, solve_circles_margin, solve_circles_cross_validation, solve_circles_alignment, solve_circles_alignment_approximation, solve_circles_genetic_and_parameter_training_rmse, solve_circles_genetic_and_parameter_training_target_alignment, solve_circles_dynamic_dataset_size = generate_genetic_metric_variation_experiment_functions(circles_dataset)
solve_adhoc_accuracy, solve_adhoc_margin, solve_adhoc_cross_validation, solve_adhoc_alignment, solve_adhoc_alignment_approximation, solve_adhoc_genetic_and_parameter_training_rmse, solve_adhoc_genetic_and_parameter_training_target_alignment, solve_adhoc_dynamic_dataset_size = generate_genetic_metric_variation_experiment_functions(adhoc_dataset)
solve_voice_accuracy, solve_voice_margin, solve_voice_cross_validation, solve_voice_alignment, solve_voice_alignment_approximation, solve_voice_genetic_and_parameter_training_rmse, solve_voice_genetic_and_parameter_training_target_alignment, solve_voice_dynamic_dataset_size = generate_genetic_metric_variation_experiment_functions(voice_dataset)
solve_susy_accuracy, solve_susy_margin, solve_susy_cross_validation, solve_susy_alignment, solve_susy_alignment_approximation, solve_susy_genetic_and_parameter_training_rmse, solve_susy_genetic_and_parameter_training_target_alignment, solve_susy_dynamic_dataset_size = generate_genetic_metric_variation_experiment_functions(susy_dataset)
solve_susy_hard_accuracy, solve_susy_hard_margin, solve_susy_hard_cross_validation, solve_susy_hard_alignment, solve_susy_hard_alignment_approximation, solve_susy_hard_genetic_and_parameter_training_rmse, solve_susy_hard_genetic_and_parameter_training_target_alignment, solve_susy_hard_dynamic_dataset_size = generate_genetic_metric_variation_experiment_functions(susy_hard_dataset)


# run a bunch of experiments, saving results
function main(seed=22)
    function dynamic_dataset_size_approach_runs()
        for (fn, name) in zip([
                                solve_moons_dynamic_dataset_size,
                                solve_cancer_dynamic_dataset_size,
                                #solve_iris_dynamic_dataset_size,
                                solve_digits_dynamic_dataset_size,
                                #solve_blobs_dynamic_dataset_size,
                                solve_circles_dynamic_dataset_size,
                                solve_adhoc_dynamic_dataset_size,
                                #solve_voice_dynamic_dataset_size,
                                solve_susy_dynamic_dataset_size,
                                solve_susy_hard_dynamic_dataset_size
                                ],
                                [
                                "moons",
                                "cancer",
                                #"iris",
                                "digits",
                                #"blobs",
                                "circles",
                                "adhoc",
                                #"voice", # doesn't work with approach 0 for dynamic dataset size due to a freeze for unknown reasons
                                "susy",
                                "susy_hard"
                                ])
            println("Solving $name, dynamic dataset size approach")
            results = fn(;seed=seed)
            println("Finished $name dynamic dataset size")
            save_results(name, "dynamic_dataset_size", results)
        end
    end
    function original_approach_runs()
        # run all experiments of purely genetic training, saving results
        for (fn, name) in zip([
                                #solve_moons_accuracy,
                                #solve_cancer_accuracy,
                                #solve_iris_accuracy,
                                solve_digits_accuracy,
                                #solve_blobs_accuracy,
                                #solve_circles_accuracy,
                                #solve_adhoc_accuracy,
                                #solve_voice_accuracy,
                                #solve_susy_accuracy,
                                #solve_susy_hard_accuracy
                                ],
                                [
                                #"moons",
                                #"cancer",
                                #"iris",
                                "digits",
                                #"blobs",
                                #"circles",
                                #"adhoc",
                                #"voice",
                                #"susy",
                                #"susy_hard"
                                ])
            println("Solving $name, original approach")
            results = fn(;seed=seed)
            println("Finished $name accuracy")
            save_results(name, "accuracy", results)
        end
    end
    function alignment_metric_approach_runs()
        # run all experiments of purely genetic training, saving results
        for (fn, name) in zip([
                                #solve_moons_alignment,
                                #solve_cancer_alignment,
                                #solve_iris_alignment,
                                solve_digits_alignment,
                                #solve_blobs_alignment,
                                #solve_circles_alignment,
                                #solve_adhoc_alignment,
                                #solve_voice_alignment,
                                #solve_susy_alignment,
                                #solve_susy_hard_alignment
                                ],
                                [
                                #"moons",
                                #"cancer",
                                #"iris",
                                "digits",
                                #"blobs",
                                #"circles",
                                #"adhoc",
                                #"voice",
                                #"susy",
                                #"susy_hard"
                                ])
            println("Solving $name, alignment metric approach")
            results = fn(;seed=seed)
            println("Finished $name alignment metric")
            save_results(name, "alignment", results)
        end
    end
    function alignment_approximation_metric_approach_runs()
        # run all experiments of purely genetic training, saving results
        for (fn, name) in zip([
                                #solve_moons_alignment_approximation,
                                #solve_cancer_alignment_approximation,
                                #solve_iris_alignment_approximation,
                                solve_digits_alignment_approximation,
                                #solve_blobs_alignment_approximation,
                                #solve_circles_alignment_approximation,
                                #solve_adhoc_alignment_approximation,
                                #solve_voice_alignment_approximation,
                                #solve_susy_alignment_approximation,
                                #solve_susy_hard_alignment_approximation
                                ],
                                [
                                #"moons",
                                #"cancer",
                                #"iris",
                                "digits",
                                #"blobs",
                                #"circles",
                                #"adhoc",
                                #"voice",
                                #"susy",
                                #"susy_hard"
                                ])
            println("Solving $name, alignment approximation metric approach")
            results = fn(;seed=seed)
            println("Finished $name alignment approximation metric")
            save_results(name, "alignment_approximation", results)
        end
    end
    function rmse_approach_runs()
        # run all experiments with the genetic training including parameter optimization for minimizing rmse
        for (fn, name) in zip([
                                solve_moons_genetic_and_parameter_training_rmse,
                                solve_cancer_genetic_and_parameter_training_rmse,
                                #solve_iris_genetic_and_parameter_training_rmse,
                                solve_digits_genetic_and_parameter_training_rmse,
                                #solve_blobs_genetic_and_parameter_training_rmse,
                                solve_circles_genetic_and_parameter_training_rmse,
                                solve_adhoc_genetic_and_parameter_training_rmse,
                                #solve_voice_genetic_and_parameter_training_rmse,
                                solve_susy_genetic_and_parameter_training_rmse,
                                solve_susy_hard_genetic_and_parameter_training_rmse
                                ],
                                [
                                "moons",
                                "cancer",
                                #"iris",
                                "digits",
                                #"blobs",
                                "circles",
                                "adhoc",
                                #"voice",
                                "susy",
                                "susy_hard"
                                ])
        println("Solving $name, rmse")
        results = fn(;seed=seed)
        println("Finished $name accuracy with parameter training for RMSE minimization")
        save_results(name, "rmse_parameter_training", results)
        end
    end
    function alignment_approach_runs()
        # run all experiments with the genetic training including parameter optimization for maximizing target alignment
        for (fn, name) in zip([
                                solve_moons_genetic_and_parameter_training_target_alignment,
                                solve_cancer_genetic_and_parameter_training_target_alignment,
                                #solve_iris_genetic_and_parameter_training_target_alignment,
                                solve_digits_genetic_and_parameter_training_target_alignment,
                                #solve_blobs_genetic_and_parameter_training_target_alignment,
                                solve_circles_genetic_and_parameter_training_target_alignment,
                                solve_adhoc_genetic_and_parameter_training_target_alignment,
                                #solve_voice_genetic_and_parameter_training_target_alignment,
                                solve_susy_genetic_and_parameter_training_target_alignment,
                                solve_susy_hard_genetic_and_parameter_training_target_alignment
                                ],
                                [
                                "moons",
                                "cancer",
                                #"iris",
                                "digits",
                                #"blobs",
                                "circles",
                                "adhoc",
                                #"voice",
                                "susy",
                                "susy_hard"
                                ])
            println("Solving $name, alignment")
            results = fn(;seed=seed)
            println("Finished $name accuracy with parameter training for target alignment maximization")
            save_results(name, "alignment_parameter_training", results)
        end
    end
    function classical_rbf_approach()
        for name in [
                    "moons",
                    "cancer",
                    #"iris",
                    "digits",
                    #"blobs",
                    "circles",
                    "adhoc",
                    #"voice",
                    "susy",
                    "susy_hard"
                    ]
            dataset::Dataset = dataset_map[name]
            train_samples, test_samples, train_labels, test_labels = py"train_test_split"(dataset.training_samples, dataset.training_labels, train_size=0.7, random_state=seed, shuffle=true)
            model = SVC(kernel="rbf", class_weight="balanced")
            #fit!(model, train_samples, train_labels) #use same training data as genetically produced QSVM kernels
            fit!(model, dataset.training_samples, dataset.training_labels) #use all data used in the genetic creation of QSVM kernels
            println("Dataset: ", name)
            println("Train acc: ", score(model,  train_samples, train_labels))
            println("Test acc: ", score(model, test_samples, test_labels))
            println("Validation acc: ", score(model, dataset.validation_samples, dataset.validation_labels))
            println()
        end
    end
    @time begin
        # start approaches in parallel
        #a = Dagger.@spawn original_approach_runs()
        #b = Dagger.@spawn rmse_approach_runs()
        #c = Dagger.@spawn alignment_approach_runs()
        # wait for results
        #fetch(a)
        #fetch(b)
        #fetch(c)
        #dynamic_dataset_size_approach_runs()
        original_approach_runs()
        #rmse_approach_runs()
        #alignment_approach_runs()
        #classical_rbf_approach()
        alignment_metric_approach_runs()
        alignment_approximation_metric_approach_runs()
    end
end
