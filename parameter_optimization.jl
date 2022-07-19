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

"Returns the alignment of two matrices.
Check paper on training kernels to maximise target alignment
for details."
function matrix_alignment(a, b)
    # NOTE on implementation: frobenius inner product operation is simply
    # the sum of the product of the corresponding entries of 2 matrices,
    # which is already what the julia dot (_⋅_) operator returns for matrices.
    (a ⋅ b) / √((a ⋅ a) * (b ⋅ b))
end

#TODO: increase cpu utilization in this function (was low with a 6 qubit 6 depth circuit)
# only 1 core was at max load, probably the one creating the tasks
"Like compute_kernel_matrix but only for the symmetric case, and computing
entries in parallel."
function compute_kernel_matrix_parallel(kernel_block, samples)
    dimension = length(samples)

    # create matrix to store tasks in.
    # the bottom half is not used, maybe this could
    # be optimised to halve matrix memory use if it
    # becomes an issue
    tasks::Matrix{Dagger.EagerThunk} = Matrix(undef, dimension, dimension)
    # create matrix to store result in (created here before
    # the task submission loop as the main diagonal results
    # are written directly in that loop)
    result = zeros(Float64, (dimension, dimension))

    # create kernel calculation tasks for half the matrix
    @inbounds for (j, s1) in enumerate(samples)
        # main diagonal is all 1.0's
        result[j, j] = 1.0
        # upper half of matrix is explicitly computed
        @inbounds for i in 1:(j-1)
            s2 = samples[i]
            output_task = Dagger.@spawn apply_kernel(kernel_block, s2, s1)
            tasks[i,j] = output_task
        end
    end

    # wait on tasks and fill the result matrix
    @inbounds for j in 1:dimension
        @inbounds for i in 1:(j-1)
            result[i, j] = fetch(tasks[i, j])
            result[j, i] = result[i, j]
        end
    end
    return result
end

"Scales labels by dividing them by the number of instances in their class.
This is used when computing the kernel target alignment to correct for unbalanced
datasets. Compare to the pennylane target alignment calculation to check correctness."
function scale_labels(labels)
    n_minus = count(==(-1), labels)
    n_plus = length(labels) - n_minus
    return [label == -1 ? label/n_minus : label/n_plus for label in labels]
end

"Returns the gram matrix of a hypothetical kernel that would optimally classify the labels."
function create_oracle_matrix(labels, scale=true)
    scaled_labels = scale ? scale_labels(labels) : labels
    return scaled_labels * scaled_labels'
end

"Calculates and returns the target alignment for the argument kernel
on the given samples and corresponding labels."
function kernel_target_alignment(kernel, samples, labels, oracle_matrix=create_oracle_matrix(labels)) # here oracle_matrix is the outer product of the label row vector and label column vector
    gram_matrix = compute_kernel_matrix(kernel, samples, samples)
    return matrix_alignment(gram_matrix, oracle_matrix)
end

"Two argument version of the function that takes the matrix arguments directly to
allow them to be re-used from computations elsewhere."
function kernel_target_alignment(gram_matrix, oracle_matrix=nothing)
    return matrix_alignment(gram_matrix, oracle_matrix)
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

"Optimizes a kernel to maximise classification accuracy. max_evaluations determines the maximum number of
parameter tests the optimizer can perform. In this case that is the number of times the kernel can be
used to train and test a model."
function optimize_kernel_accuracy(parameterised_kernel, initial_parameters, problem_data; max_evaluations=100, seed=22, verbose=false)

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
        objective = accuracy_metric_yao(model, problem_data)

        # record the objective value for convergence analysis
        push!(objective_history, objective)
        if verbose
            println("Evaluation: $evaluation_counter\nAccuracy: $objective\nParameters: $parameters\n")
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

#TODO: find a way to define the Dataset struct in all workers without duplicating the definition from datasets.jl
#=
function optimize_kernel_accuracy(parameterised_kernel, initial_parameters, dataset::Dataset; seed=22, kwargs...)
    # get the training data from the dataset
    samples, labels = dataset.training_samples, dataset.training_labels
    (train_samples, test_samples, train_labels, test_labels) = py"train_test_split"(samples,
                                                                                    labels,
                                                                                    train_size=0.7,
                                                                                    random_state=seed,
                                                                                    shuffle=true)
    problem_data = (train_samples, test_samples, train_labels, test_labels)
    # forward to the method that takes the samples and labels directly
    return optimize_kernel_accuracy(parameterised_kernel, initial_parameters, problem_data; kwargs...)
end
=#

#TODO: try optimizing target alignment or average margin size instead of accuracy
# so that there is more oppurtunity for improvement on smaller data sets.
# accuracy converges too quickly for small data sets since outliers can make
# up a relatively large percent of the data with small data sets
"Takes a vector of individuals and trains them using parameter
based training to better classify the dataset. Returns the trained
parameter values and fitness evaluation histories of the individuals."
function population_parameterised_training(population, dataset, feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=22, metric_type="accuracy")
    #NOTE: max_evaluations specifies the maximum fitness evaluations per individual, not over the whole
    #population.
    #NOTE: although the argument name is dataset, it takes a value of problem_data form

    metrics_to_target_optimizers = Dict("accuracy"=>optimize_kernel_accuracy, "target_alignment"=>optimize_kernel_target_alignment)
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
    #TODO: parallelise this loop
    population_final_parameters::Vector{Vector{Float64}} = []
    histories::Vector{Vector{Float64}} = []
    for (params, hist) in outputs
        push!(population_final_parameters, params)
        push!(histories, hist)
    end
    #=
    for c in population
        params, hist = process_individual(c)
        push!(population_final_parameters, params)
        push!(histories, hist)
    end
    =#

    return population_final_parameters, histories
end
