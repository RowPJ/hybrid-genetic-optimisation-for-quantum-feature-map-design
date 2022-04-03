using PyCall

py"""
# This file implements the NSGA2 optimization problem definition for
# use with pymoo.

# The genetic kernel circuit design optimization problem in words:
# * minimize the negative of the accuracy of the model trained from the kernel circuit
# * minimize weighted size metric
# * constraints are that each variable has a value of either 0 or 1

from pymoo.core.problem import Problem  # problem base class
from sklearn.model_selection import train_test_split
import numpy as np


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

        # Perform splitting and scaling of the data once here so that
        # it doesn't need to be done for every call of the fitness
        # function.
        train_percent = 0.7     # fraction of data to use for training
        samples, labels = problem_data  # unpack feature vectors and their labels

        # split data into training and testing subsets.
        # The form of problem_data was previously (samples, labels)
        # but after this line it is (train_samples, train_labels,
        # test_samples, test_labels).
        self.problem_data = train_test_split(samples,
                                             labels,
                                             train_size=train_percent,
                                             random_state=seed,
                                             shuffle=True)

    def _evaluate(self, xs, out, *args, **kwargs):
        metrics = self.evaluator(xs, self.feature_count,
                                 self.qubit_count, self.depth,
                                 self.problem_data, self.seed)
        # First objective value is the negative of the accuracy
        # since accuracy should be maximised rather than minimized,
        # and second is the weighted size which should be minimized.
        # The margin metric is ignored here since this is copying
        # the paper's objective values.
        first_two_metrics = np.array([[-m[0], m[1]] for m in metrics]) #m[0] is accuracy or its substitute, m[1] is size metric
        # Set objective values.
        out["F"] = first_two_metrics
"""