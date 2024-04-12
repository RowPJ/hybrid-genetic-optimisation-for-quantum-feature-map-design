#TODO: replace population evaluator functions with a generic version that takes
# a preprocessing step and the fitness function to use on individuals

using Dagger

"Calculates fitness of a multiple solutions in parallel using circuit size
and classification accuracy as fitness metrics."
function evaluate_population_yao(population, feature_count,
                                 qubit_count, depth,
                                 problem_data, generation,
                                 seed=22)
    # extracts a view of ith row (the views share
    # storage with the matrix m)
    row(m, i) = @view m[i, :]
    # takes a matrix and returns a vector of views of
    # its rows
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    # convert population to an array of bool arrays
    # rather than a matrix of bools
    population_converted = to_rows(population)
    
    # make a task for each individual's fitness
    row_to_task = (row -> Dagger.@spawn fitness_yao(row,
                                                    feature_count,
                                                    qubit_count,
                                                    depth,
                                                    problem_data,
                                                    seed))
    
    tasks = row_to_task.(population_converted)
    # wait for the tasks to complete
    metrics = fetch.(tasks)
    # return the metrics for each individual
    return metrics
end

#TODO: add a flag for not restricting test set size for a better accuracy metric without increasing runtime so much
"Calculates fitness of a multiple solutions in parallel using circuit size
and classification accuracy as fitness metrics."
function evaluate_population_yao_dynamic_dataset_size(population, feature_count,
                                                      qubit_count, depth,
                                                      problem_data, generation,
                                                      seed=22)
    # extracts a view of ith row (the views share
    # storage with the matrix m)
    row(m, i) = @view m[i, :]
    # takes a matrix and returns a vector of views of
    # its rows
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    # convert population to an array of bool arrays
    # rather than a matrix of bools
    population_converted = to_rows(population)
    restricted_size_fraction = min(max(0.2, generation / 1200.0), 1.0)
    restrict_size(vector) = vector[1:max(1, trunc(Int64, restricted_size_fraction*length(vector)))]
    problem_data_converted = restrict_size.(problem_data)
    # make a task for each individual's fitness
    row_to_task = (row -> Dagger.@spawn fitness_yao(row,
                                                    feature_count,
                                                    qubit_count,
                                                    depth,
                                                    problem_data_converted,
                                                    seed))
    tasks = row_to_task.(population_converted)
    # wait for the tasks to complete
    metrics = fetch.(tasks)
    # return the metrics for each individual
    return metrics
end


"Calculates fitness of a multiple solutions in parallel using margin size
instead of accuracy as a fitness metric."
function evaluate_population_yao_margin_metric(population, feature_count,
                                               qubit_count, depth,
                                               problem_data, generation,
                                               seed=22)
    # extracts a view of ith row (the views share
    # storage with the matrix m)
    row(m, i) = @view m[i, :]
    # takes a matrix and returns a vector of views of
    # its rows
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    # convert population to an array of bool arrays
    # rather than a matrix of bools
    population_converted = to_rows(population)

    # make a task for each individual's fitness
    row_to_task = (row -> Dagger.@spawn fitness_yao_margin(row,
                                                    feature_count,
                                                    qubit_count,
                                                    depth,
                                                    problem_data,
                                                    seed))
    tasks = row_to_task.(population_converted)
    # wait for the tasks to complete
    metrics = fetch.(tasks)
    # return the metrics for each individual
    return metrics
end

"Calculates fitness of a multiple solutions in parallel using kernel-target 
alignment instead of accuracy as the fitness metric to maximise."
function evaluate_population_yao_kernel_target_alignment(population, feature_count,
                                               qubit_count, depth,
                                               problem_data, generation,
                                               seed=22)
    # extracts a view of ith row (the views share
    # storage with the matrix m)
    row(m, i) = @view m[i, :]
    # takes a matrix and returns a vector of views of
    # its rows
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    # convert population to an array of bool arrays
    # rather than a matrix of bools
    population_converted = to_rows(population)

    samples = problem_data[1]
    labels = problem_data[3]
    oracle_matrix = create_oracle_matrix(labels)

    # make a task for each individual's fitness
    row_to_task = (row -> Dagger.@spawn fitness_yao_kernel_target_alignment(row,
                                                                            feature_count,
                                                                            qubit_count,
                                                                            depth,
                                                                            samples,
                                                                            labels,
                                                                            oracle_matrix,
                                                                            seed))
    tasks = row_to_task.(population_converted)
    # wait for the tasks to complete
    metrics = fetch.(tasks)
    # return the metrics for each individual
    return metrics
end
"Splits training data into 5 subsets and returns the average alignment of models trained with the subsets."
function evaluate_population_yao_kernel_target_alignment_split(population, feature_count,
                                                                qubit_count, depth,
                                                                problem_data, generation,
                                                                seed=22)
    # extracts a view of ith row (the views share
    # storage with the matrix m)
    row(m, i) = @view m[i, :]
    # takes a matrix and returns a vector of views of
    # its rows
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    # convert population to an array of bool arrays
    # rather than a matrix of bools
    population_converted = to_rows(population)

    samples = problem_data[1]
    labels = problem_data[3]
    #oracle_matrix = create_oracle_matrix(labels)

    split_count = 5
    split_size = trunc(Int64, length(samples)/split_count)
    sample_splits = @views [samples[start:start+split_size-1] for start in 1:split_size:(split_size*split_count)]
    label_splits = @views [labels[start:start+split_size-1] for start in 1:split_size:(split_size*split_count)]
    oracle_matrices = [create_oracle_matrix(labels) for labels in label_splits]

    # make a task for each individual's fitness
    row_to_task = (row -> Dagger.@spawn fitness_yao_kernel_target_alignment_split(row,
                                                                                    feature_count,
                                                                                    qubit_count,
                                                                                    depth,
                                                                                    sample_splits,
                                                                                    label_splits,
                                                                                    oracle_matrices,
                                                                                    seed))
    tasks = row_to_task.(population_converted)
    # wait for the tasks to complete
    metrics = fetch.(tasks)
    # return the metrics for each individual
    return metrics
end

"Calculates fitness of a multiple solutions in parallel, using 5-fold
cross-validation when calculating the accuracy metrics of the population."
function evaluate_population_yao_cross_validation(population, feature_count,
                                                  qubit_count, depth,
                                                  problem_data, generation,
                                                  seed=22)
    # extracts a view of ith row (the views share
    # storage with the matrix m)
    row(m, i) = @view m[i, :]
    # takes a matrix and returns a vector of views of
    # its rows
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    # convert population to an array of bool arrays
    # rather than a matrix of bools
    population_converted = to_rows(population)

    # recombine training and testing data so that it can be split again in cross validations
    # (use cat to recombine separated training and testing data. should maybe fix this in the python side
    # if it becomes a problem)
    problem_data_converted = (cat(problem_data[1], problem_data[2], dims=1),
                              cat(problem_data[3], problem_data[4], dims=1))
    # make a task for each individual's fitness
    row_to_task = (row -> Dagger.@spawn fitness_yao_cross_validation(row,
                                                                     feature_count,
                                                                     qubit_count,
                                                                     depth,
                                                                     problem_data_converted,
                                                                     seed))
    tasks = row_to_task.(population_converted)
    # wait for the tasks to complete
    metrics = fetch.(tasks)
    # return the metrics for each individual
    return metrics
end

function evaluate_population_yao_parameter_training_rmse(population, feature_count,
                                                        qubit_count, depth,
                                                        problem_data, generation,
                                                        seed=22,
                                                        max_parameter_training_evaluations=100) #parameter_iterations controls how many iterations of parameter based training to apply to each individual when evaluating their fitness
    # extracts a view of ith row (the views share
    # storage with the matrix m)
    row(m, i) = @view m[i, :]
    # takes a matrix and returns a vector of views of
    # its rows
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    # convert population to an array of bool arrays
    # rather than a matrix of bools
    population_converted = to_rows(population)

    # make a task for each individual's fitness
    row_to_task = (row -> Dagger.@spawn fitness_yao_parameter_training_rmse(row,
                                                                           feature_count,
                                                                           qubit_count,
                                                                           depth,
                                                                           problem_data,
                                                                           seed,
                                                                           max_parameter_training_evaluations))
    tasks = row_to_task.(population_converted)
    # wait for the tasks to complete
    metrics = fetch.(tasks)
    # return the metrics for each individual
    return metrics
end

function evaluate_population_yao_parameter_training_target_alignment(population, feature_count,
                                                                     qubit_count, depth,
                                                                     problem_data, generation,
                                                                     seed=22,
                                                                     max_parameter_training_evaluations=100) #parameter_iterations controls how many iterations of parameter based training to apply to each individual when evaluating their fitness
    # extracts a view of ith row (the views share
    # storage with the matrix m)
    row(m, i) = @view m[i, :]
    # takes a matrix and returns a vector of views of
    # its rows
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    # convert population to an array of bool arrays
    # rather than a matrix of bools
    population_converted = to_rows(population)

    # ensure labels are 1 and -1 for target alignment definition to work
    (_, _, train_labels, test_labels) = problem_data
    if !all(x -> x == 1 || x == -1, train_labels)
        println(train_labels)
        error("Some training set labels from pymoo are not -1 or 1.")
    end
    if !all(x -> x == 1 || x == -1, test_labels)
        println(test_labels)
        error("Some testing set labels from pymoo are not -1 or 1.")
    end

    # make a task for each individual's fitness
    row_to_task = (row -> Dagger.@spawn fitness_yao_parameter_training_target_alignment(row,
                                                                                        feature_count,
                                                                                        qubit_count,
                                                                                        depth,
                                                                                        problem_data,
                                                                                        seed,
                                                                                        max_parameter_training_evaluations))
    tasks = row_to_task.(population_converted)
    # wait for the tasks to complete
    metrics = fetch.(tasks)
    # return the metrics for each individual
    return metrics
end
