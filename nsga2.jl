using PyCall

py"""
# This file implements the NSGA2 optimization problem definition for
# use with pymoo.

# The genetic kernel circuit design optimization problem in words:
# * minimize the negative of the accuracy of the model trained from the kernel circuit
# * minimize weighted size metric
# * constraints are that each variable has a value of either 0 or 1

from pymoo.core.problem import Problem  # problem base class
from pymoo.core.duplicate import DuplicateElimination, ElementwiseDuplicateElimination # base class for eliminating duplicate solutions
from pymoo.core.population import Population
from sklearn.model_selection import train_test_split
import numpy as np
import random

class KernelCircuitDuplicateElimination(ElementwiseDuplicateElimination):
    def __init__(self, qubit_count, feature_count, equality_checker):
        super().__init__()
        self.qubit_count = qubit_count
        self.feature_count = feature_count
        self.equality_checker = equality_checker
    #def do(self, population, n_offsprings=None, algorithm=None):
    #    duplicate_indices = []
    #    for i1 in range(len(population) - 1):
    #        c1 = population[i1]
    #        for i2 in range(i1+1, len(population)):
    #            c2 = population[i2]
    #            # quick check for inequality
    #            if c1.F is not None and c2.F is not None: #if both fitness values have already been computed, do a quick test
    #               if not np.arrayequals(c1.F, c2.F): # test if the fitness values are different, in which case inequality is trivial
    #                    continue
    #            print("Possible equality:", c1.F, c2.F, i1)
    #            # long check to confirm equality
    #            if self.equality_checker(c1.X, c2.X, self.qubit_count, self.feature_count):
    #                duplicate_indices.append(i1)
    #                break
    #    # return non-duplicates
    #    if duplicate_indices != []:
    #        print("Removing duplicates:", duplicate_indices)
    #        return Population([population[i] for i in range(len(population)) if i not in duplicate_indices])
    #    else:
    #        return population
    def is_equal(self, c1, c2):
        # early exit case where inputs cannot be equal
        if c1.F is not None and c2.F is not None:
            if not np.arrayequals(c1.F, c2.F):
                return False
        # fallback to long check in Julia
        return self.equality_checker(c1.X, c2.X, self.qubit_count, self.feature_count)

class KernelCircuitProblem(Problem):
    def __init__(self, evaluator, feature_count, qubit_count,
                 depth, problem_data, seed=22,
                 **kwargs):
        super().__init__(n_var=qubit_count*depth*5,  # n_var is chromosome length
                         n_obj=2,  # 2 objectives, weighted size metric and accuracy
                         n_constr=0,  # no constraints
                         xl=0,          # lower bound on variables
                         xu=1,          # upper bound on variables
                         **kwargs)

        # set properties for use later in _evaluate
        self.feature_count = feature_count
        self.qubit_count = qubit_count
        self.depth = depth
        self.seed = seed
        self.evaluator = evaluator
        self.train_percent = 0.7

        # Perform splitting and scaling of the data once here so that
        # it doesn't need to be done for every call of the fitness
        # function.
        self.samples, self.labels = problem_data  # unpack feature vectors and their labels
        random.seed(self.seed)
        self.seeds = [random.randint(0, 1000000) for i in range(5000)] # for shuffling samples and labels each iteration
        self.generation = 0


    def _evaluate(self, xs, out, *args, **kwargs):

        # split data into training and testing subsets.
        # The form of problem_data was previously (samples, labels)
        # but after this line it is (train_samples, train_labels,
        # test_samples, test_labels).
        # use a new seed and problem_data split each generation
        # to avoid overfitting to a specific test data selection
        # as generations pass in the genetic algorithm
        generation_seed = self.seed #self.seeds[self.generation]
        self.generation += 1
        problem_data = train_test_split(self.samples,
                                        self.labels,
                                        train_size=self.train_percent,
                                        random_state=generation_seed,
                                        shuffle=True)

        # evaluate population on the new problem data split
        metrics = self.evaluator(xs, self.feature_count,
                                 self.qubit_count, self.depth,
                                 problem_data, self.generation,
                                 generation_seed)
        # First objective value is the negative of the accuracy
        # since accuracy should be maximised rather than minimized,
        # and second is the weighted size which should be minimized.
        # The margin metric is ignored here since this is copying
        # the paper's objective values.
        first_two_metrics = np.array([[-m[0], m[1]] for m in metrics]) #m[0] is accuracy or its substitute, m[1] is size metric
        # Set objective values.
        out["F"] = first_two_metrics
"""