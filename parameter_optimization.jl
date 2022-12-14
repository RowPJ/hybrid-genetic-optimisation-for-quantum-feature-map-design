using NLopt, LinearAlgebra


#NOTE: compute_kernel_matrix_parallel is no longer used since parameter training
# now also takes place in the genetic optimization loop, which means threading
# the matrix calculation probably just adds overhead unless the matrix size
# grows a lot

"Given a chromosome and circuit properties, returns two values. The first is a function
accepting circuit parameters that itself returns a circuit when given feature values.
The second is an array of initial parameter values that can be used when optimizing the
circuit."
function decode_chromosome_parameterised_yao(chromosome, feature_count, qubit_count, depth)

    next_qubit_index(q) = q % qubit_count + 1
    angle_map = [pi, pi/2, pi/4, pi/8]

    

    block_cases = [0, 3, 2, 1, 1, 2, 2, 1]
    block_appliers = [q -> put(qubit_count, q=>H),
                      q -> cnot(qubit_count, q, next_qubit_index(q)),
                      nothing,
                      (angle, q) -> put(qubit_count, q=>Rx(angle)),
                      (angle, q) -> put(qubit_count, q=>Rz(angle)),
                      nothing,
                      nothing,
                      (angle, q) -> put(qubit_count, q=>Ry(angle))]

    "Returns the values of parameters in the parameterised gates of the chromosome,
    in the order in which the gates would be chained."
    function extract_initial_parameters()
        initial_parameters::Vector{Float64} = []
        bits_per_gate = 5
        for (i, h) in enumerate(range(start=1, step=bits_per_gate, stop=length(chromosome)))
            j = (i-1) % qubit_count + 1
            k = (i-1) % feature_count + 1
            mapping_index = (chromosome[h]*4 + chromosome[h+1]*2 + chromosome[h+2]) + 1
            block_case = @inbounds block_cases[mapping_index]
            block_applier = @inbounds block_appliers[mapping_index]
            # only account for block case 1, the case of parameterised gates
            if block_case == 1
                proportionality_parameter_index = (chromosome[h+3]*2 + chromosome[h+4]) + 1
                proportionality_parameter = angle_map[proportionality_parameter_index]
                push!(initial_parameters, proportionality_parameter)
            end
        end
        return initial_parameters
    end

    "Defines the function which can be called to substitute parameter values into the circuit.
    It returns a function that takes feature values and returns the YAO circuit that computes
    for the kernel output for that pair of feature values."
    function substitute_parameters(parameters)
        # taking parameters as an argument allows the inner functions
        # to form a closure over the variable, thereby using the specified
        # parameters when they do eventually get run
        function feature_map(features)
            # this variable is new compared to the unparameterised version
            # of the chromosome decoder. it is used to select a parameter
            # value from the parameters variable in the enclosing scope.
            # it should be incremented whenever a parameter is consumed.
            current_parameter_index = 1
            block_chain = chain(qubit_count)
            bits_per_gate = 5
            for (i, h) in enumerate(range(start=1, step=bits_per_gate, stop=length(chromosome)))
                j = (i-1) % qubit_count + 1
                k = (i-1) % feature_count + 1
                mapping_index = (chromosome[h]*4 + chromosome[h+1]*2 + chromosome[h+2]) + 1
                block_case = @inbounds block_cases[mapping_index]
                block_applier = @inbounds block_appliers[mapping_index]
                if (block_case == 0) || ((block_case == 3) && (qubit_count != 1))
                    push!(block_chain, block_applier(j))
                elseif block_case == 1
                    # this is the case of a parameterised gate.
                    # we need to take the next unused parameter
                    # value using current_parameter_index, rather
                    # than using the value determined by the chromosome
                    parameter = parameters[current_parameter_index]
                    # make sure to increment the current_parameter_index value
                    current_parameter_index += 1
                    # calculate rotation angle using the parameter and feature value
                    angle = parameter * features[k]
                    # apply gate as normal
                    push!(block_chain, block_applier(angle, j))
                else
                    # do nothing for identity operation
                end
            end
            block_chain
        end
        # the adjoint function doesn't need to be modified in this version
        # of the decoder, since it just uses feature_map to construct the gates
        # and just modifies the gate order of the returned circuit.
        function adjoint_feature_map(features)
            adjoints_chain = feature_map(0 .- features)
            chain_len = length(adjoints_chain)
            chain_len_succ = chain_len + 1
            @inbounds for i in 1:(chain_len ÷ 2)
                other_index = chain_len_succ - i
                carry = adjoints_chain[i]
                adjoints_chain[i] = adjoints_chain[other_index]
                adjoints_chain[other_index] = carry
            end
            adjoints_chain
        end

        # this function takes data values and returns a YAO kernel circuit
        # using the parameters in the enclosing scope
        construct_kernel_circuit(data1, data2) = chain(feature_map(data1), adjoint_feature_map(data2))
        return construct_kernel_circuit
    end
    
    # the return value of the decoder is the entry point
    # in the circuit construction process of substituting
    # the trainable circuit parameters
    return substitute_parameters, extract_initial_parameters()
end

"Optimizes a kernel to maximise kernel-target alignment. maxeval determines the maximum number of
target alignment calls the optimizer can make."
function optimize_kernel_target_alignment(parameterised_kernel, initial_parameters, problem_data; max_evaluations=100, seed=22, verbose=false)
    # get the training data from the dataset
    #samples, labels = dataset.training_samples, dataset.training_labels
    
    #(train_samples, test_samples, train_labels, test_labels) = py"train_test_split"(samples,
                                                                                   # labels,
                                                                                    #train_size=0.7,
                                                                                    #random_state=seed,
                                                                                   # shuffle=true)

    (train_samples, test_samples, train_labels, test_labels) = problem_data

    # calculate the oracle matrix once here, since it doesn't
    # depend on the parameter values and so doesn't change with
    # optimization.
    oracle_matrix = create_oracle_matrix(train_labels)

    # output initial alignment
    #initial_alignment = kernel_target_alignment(parameterised_kernel(initial_parameters),
    #                                                train_samples,
    #                                                train_labels,
    #                                                oracle_matrix)
    #println("Initial parameters and kernel target alignment:\n$initial_parameters\n$initial_alignment\n")

    # define the NLopt optimization problem
    opt = Opt(:LN_COBYLA, length(initial_parameters))
    

    evaluation_counter = 0
    objective_history = []
    # A gradient vector to manually fill is supplied by the algorithm 
    # to the objective function, but it can be ignored in the case
    # of gradient-free optimizers like COBYLA.
    "Computes objective function while printing out progress
    of optimization."
    function progress_objective(parameters, gradients)
        evaluation_counter += 1
        
        objective = kernel_target_alignment(parameterised_kernel(parameters),
                                            train_samples,
                                            train_labels,
                                            oracle_matrix)

        # record the objective value for convergence analysis
        push!(objective_history, objective)
        if verbose
            println("Evaluation: $evaluation_counter\nAlignment: $objective\nParameters: $parameters\n")
        end
        return objective
    end
    opt.max_objective = progress_objective
    # set maximum number of fitness evaluations
    opt.maxeval = max_evaluations
    # optimize parameters
    (final_objective_value, final_parameters, return_code) = optimize(opt, initial_parameters)
    # return the relevant values
    return (final_parameters, final_objective_value, opt.numevals, objective_history, return_code)
end

function model_root_mean_squared_error(model::TrainedModel, problem_data)
    (train_samples, test_samples, train_labels, test_labels) = problem_data
    squared_errors_sum::Float64 = 0.0
    num_samples::Int64 = length(train_labels)# + length(test_labels)
    num_plus::Float64 = count(==(1), train_labels)# + count(==(1), test_labels)
    num_minus::Float64 = num_samples - num_plus

    function accumulate_errors(decision_function_outputs, labels)
        for (output, label) in zip(decision_function_outputs, labels)
            # only count errors from samples not reaching the output value of the correct label.
            # scale errors down by the number of members in the class to balance contributions by each class.
            if (label == -1 && output > -1)
                squared_errors_sum += (label - output)^2 / num_minus
            elseif (label == 1 && output < 1)
                squared_errors_sum += (label - output)^2 / num_plus
            end
        end
    end
    train_set_outputs = decision_function(model.classifier, model.gram_matrix)
    #test_kernel_outputs = compute_kernel_matrix(model.kernel, model.training_set, test_samples)
    #test_set_outputs = decision_function(model.classifier, test_kernel_outputs)
    accumulate_errors(train_set_outputs, train_labels)
    #accumulate_errors(test_set_outputs, test_labels)
    # scale up error by the number of samples to reverse class-based weighting
    squared_errors_sum *= num_samples
    # calculate root of mean error
    return sqrt(squared_errors_sum / num_samples)
end

"Optimizes a kernel to minimize root mean squared error (RMSE) between the classifier outputs and the data labels for training data,
assuming the class labels are -1 and 1. max_evaluations determines the maximum number of parameter tests the optimizer can perform.
In this case that is the number of times the kernel can be used to train and test a model."
function optimize_kernel_rmse(parameterised_kernel, initial_parameters, problem_data; max_evaluations=100, seed=22, verbose=false)

    (train_samples, test_samples, train_labels, test_labels) = problem_data

    # ensure labels are positive and negative 1
    if !all(x -> x == 1 || x == -1, train_labels)
        error("Some training set labels are not -1 or 1.")
    end
    if !all(x -> x == 1 || x == -1, test_labels)
        error("Some testing set labels are not -1 or 1.")
    end

    # define the NLopt optimization problem
    opt = Opt(:LN_COBYLA, length(initial_parameters))

    evaluation_counter = 0
    objective_history = []
    # A gradient vector to manually fill is supplied by the algorithm 
    # to the objective function, but it can be ignored in the case
    # of gradient-free optimizers like COBYLA.
    "Computes objective function while printing out progress
    of optimization."
    function progress_objective(parameters, gradients)
        evaluation_counter += 1
        
        kernel = parameterised_kernel(parameters)
        model = train_model(problem_data, kernel)
        objective = model_root_mean_squared_error(model, problem_data)

        # record the objective value for convergence analysis
        push!(objective_history, objective)
        if verbose
            println("Evaluation: $evaluation_counter\nRMSE: $objective\nParameters: $parameters\n")
        end
        return objective
    end
    opt.min_objective = progress_objective
    # set maximum number of fitness evaluations
    opt.maxeval = max_evaluations
    # optimize parameters
    (final_objective_value, final_parameters, return_code) = optimize(opt, initial_parameters)
    # return the relevant values
    return (final_parameters, final_objective_value, opt.numevals, objective_history, return_code)
end

"Takes a vector of individuals and trains them using parameter
based training to better classify the dataset. Returns the trained
parameter values and fitness evaluation histories of the individuals."
function population_parameterised_training(population, dataset, feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=22, metric_type="rmse")
    #NOTE: max_evaluations specifies the maximum fitness evaluations per individual, not over the whole
    #population.
    #NOTE: although the argument name is dataset, it takes a value of problem_data form

    metrics_to_target_optimizers = Dict("rmse"=>optimize_kernel_rmse, "target_alignment"=>optimize_kernel_target_alignment)
    target_optimizer = metrics_to_target_optimizers[metric_type]
    
    function process_individual(chromosome)
        parameterised_kernel, initial_parameters = decode_chromosome_parameterised_yao(chromosome,
                                                                                       feature_count,
                                                                                       qubit_count,
                                                                                       depth)
        optimized_parameters, final_objective, num_evals, history, return_code = target_optimizer(parameterised_kernel,
                                                                                                          initial_parameters,
                                                                                                          dataset;
                                                                                                          max_evaluations=max_evaluations,
                                                                                                          seed=seed)
        if return_code == :MAXEVAL_REACHED
            return optimized_parameters, history
        else
            # in the case of failed optimization, just record repeated copies of the initial objective
            # to represent no improvement and return the initial parameter values
            return initial_parameters, fill(final_objective, max_evaluations)
        end
    end

    tasks = [Dagger.@spawn process_individual(c) for c in population]
    outputs = fetch.(tasks)
    population_final_parameters::Vector{Vector{Float64}} = []
    histories::Vector{Vector{Float64}} = []
    for (params, hist) in outputs
        push!(population_final_parameters, params)
        push!(histories, hist)
    end
    if metric_type == "rmse"
        println("Min RMSE: $(minimum(x->x[end], histories))")
    elseif metric_type == "target_alignment"
        println("Max kernel-target alignment: $(maximum(x->x[end], histories))")
    end

    return population_final_parameters, histories
end
