#TODO: ideas to speed up the code:
# 1. evaluate the generated circuit with symbolic arguments, then extract the real part of the first amplitude's
#    symbolic expression. this expression can be compiled to a fast function with Symbolics.jl in order to get a
#    highly optimized function taking the data points as parameters and computing the kernel output of the circuit
#    if it had run with those parameters.
#    PROS:
#       1. it won't need to perform any calculations that aren't necessary for computing the zero basis state's
#          amplitude which will speed up execution.
#       2. the circuit generation code will only run once per chromosome with symbolic arguments, not once for each
#          pair of data points the kernel is computed for
#       3. the generated function will only evaluate a single large mathematical expression, not perform matrix
#          operations or other operations. this could maybe be sped up even more by compiling it with the @fastmath
#          macro, although the outputs will have to be checked for correctness if @fastmath is used
#   CONS:
#      1. Compiling the symbolic expression may take a long time and may evaluate a lot of the circuit operations
#         again anyway, so the speedup may not be much
#      2. Need to get symbolic complex number circuit registers working with all required operations
#   DETAILS:
#     1. would create symbolic circuit parameters like this: @variables (d1::Float64)[2] (d2::Float64)[2]
#     2. can create symbolic type zero statevector with: symbolic_zero_state(n) = append!([Complex{Num}(1.0)], [Complex{Num}(0.0) for i in 2:2^n])
#     3. can create symbolic ArrayReg register with: symbolic_register(n) = ArrayReg(symbolic_zero_state(n))
#     4. need to implement functions that 

#TODO: incorporate class weighting into model training and fitness 
# calculation for unbalanced data sets. this would need to be managed
# for the average margin metric as well as SVM creation, maybe elsewhere
# too
# Only training with balanced data sets is not an alternative since the train/test split could be unbalanced.

###########################################################################
#- This file contains the code that all workers need to load to function -#
###########################################################################
using Yao, CuYao, GPUArrays
using Distributed, Dagger
using PyCall
using ScikitLearn

# import SVC class from python
@sk_import svm: (SVC) # equivalent to "from sklearn.svm import SVC" in python
import ScikitLearn: CrossValidation # for train_test_split

# import numpy from python
np = pyimport("numpy") # import numpy module to set seed used by sklearn

"The variable to check the process environment for
to determine which register type (cpu or gpu) to
use."
const REGISTER_VARIABLE = "YAO_REGISTER_TYPE"
const GPU_REGISTER_TYPE = "GPU"
const CPU_REGISTER_TYPE = "CPU"


"This function creates a Yao register on the cpu
or gpu based on the value of the environment variable
YAO_REGISTER_TYPE."
function make_register(qubit_count::Integer)
    register_type = ENV[REGISTER_VARIABLE]
    if register_type == GPU_REGISTER_TYPE
        return cu(zero_state(qubit_count))
    elseif register_type == CPU_REGISTER_TYPE
        return zero_state(qubit_count)
    else
        error("Tried to make a register with"*
              " register type \"$register_type\","*
              " only \"$GPU_REGISTER_TYPE\" and "*
              "\"$CPU_REGISTER_TYPE\" are valid.")
    end
end

#This variable is made for workers to re-use registers if they run multiple circuits
#with the same number of qubits in a row. The register is made a local variable so that
#it can't be modified from outside of the enclosed function. This could speed up the code
#since it can assume that worker_register won't be modified randomly be other threads or functions.
let worker_register = nothing
    global ensure_register
    "This function should be called to make sure that worker_register contains a register
    with the correct number of qubits. It also returns the valid worker_register value
    for convenience."
    function ensure_register(qubit_count::Integer)
        if isnothing(worker_register) || (nqubits(worker_register) != qubit_count)
            worker_register = make_register(qubit_count)
        end
        return worker_register
    end
end

"This function returns a callable function taking two parameters as input:
the feature vectors of two data points to compute the kernel output for."
function decode_chromosome_yao(chromosome, feature_count, qubit_count, depth; plotting=false)

    # function used when creating the cnot block.
    # it is separate since 1-based array indexing
    # complicates the modulus calculation.
    # this is trivially correct for q = 1 to qubit_count-1
    # inclusive, and for q=qubit_count, the modulus is 0 and
    # adding 1 gives the correct answer of 1
    next_qubit_index(q) = q % qubit_count + 1

    # for the case of decoding with symbolic features, the angles should be
    # rounded to 2 digits so they fit better in the plot boxes. they still
    # won't fully fit but will be easier to read.
    angle_map = plotting ? round.([pi, pi/2, pi/4, pi/8], digits=2) : [pi, pi/2, pi/4, pi/8]
    
    block_cases = [0, 3, 2, 1, 1, 2, 2, 1]
    block_appliers = [q -> put(qubit_count, q=>H),
                      q -> cnot(qubit_count, q, next_qubit_index(q)),
                      nothing,
                      (angle, q) -> put(qubit_count, q=>Rx(angle)),
                      (angle, q) -> put(qubit_count, q=>Rz(angle)),
                      nothing,
                      nothing,
                      (angle, q) -> put(qubit_count, q=>Ry(angle))]
    
    # define feature map part of kernel circuit
    function feature_map(features)
        # initial circuit is just an identity gate.
        # this ensures it isn't empty and gives a
        # base to chain onto
        block_chain = chain(qubit_count)
        bits_per_gate = 5
        # i is the identifier for the group of five bits,
        # h is the index within chromosome that the group starts at.
        for (i, h) in enumerate(range(start=1, step=bits_per_gate, stop=length(chromosome)))
            # calculate qubit to act on and feature to depend on.
            # contrasts with python in that indices need to be adjusted
            # before modular operations are performed
            j = (i-1) % qubit_count + 1
            k = (i-1) % feature_count + 1
            
            # calculate which block to apply
            mapping_index = (chromosome[h]*4 + chromosome[h+1]*2 + chromosome[h+2]) + 1 #+1 for 1 based indexing
            
            block_case = @inbounds block_cases[mapping_index]
            block_applier = @inbounds block_appliers[mapping_index]
            
            # apply the block
            # (only do cnot gate if there is more than 1 qubit,
            # otherwise do an identity)
            if (block_case == 0) || ((block_case == 3) && (qubit_count != 1))
                push!(block_chain, block_applier(j))
            elseif block_case == 1
                proportionality_parameter_index = (chromosome[h+3]*2 + chromosome[h+4]) + 1 #+1 for 1 based indexing
                proportionality_parameter = angle_map[proportionality_parameter_index]
                angle = proportionality_parameter * features[k]
                push!(block_chain, block_applier(angle, j))
            else
                # do nothing for identity operation (block_case == 2).
                # this could make generated circuits a bit smaller
                # but should be equivalent to applying the identity
                # gate anyway
            end
        end
        # return the chain of operation blocks
        block_chain
    end

    # Yao adjoint operation was slow, so simply run feature map operation
    # again using the negatives of the feature values, then reverse the order
    # in which the gates are applied. This works since the adjoint of a
    # rotation is a rotation in the opposite direction of the same magnitude,
    # and the non-parameterised gates (H, CNOT) are their own adjoints.
    function adjoint_feature_map(features)
        adjoints_chain = feature_map(0 .- features)
        # reverse the chain elements with the indexing operation
        chain_len = length(adjoints_chain)
        chain_len_succ = chain_len + 1
        # for even length lists, swaps all elements on left with
        # one on right. for odd length lists, does the same but
        # stops before the middle element
        @inbounds for i in 1:(chain_len ÷ 2)
            other_index = chain_len_succ - i
            carry = adjoints_chain[i]
            adjoints_chain[i] = adjoints_chain[other_index]
            adjoints_chain[other_index] = carry
        end
        adjoints_chain
    end
            

    # Define the kernel function in terms of the
    # feature map: apply the feature map to encode
    # the first data point, then apply the adjoint
    # of the encoding of the second data point.
    # Return as a function so data can be supplied later.
    (data1, data2) -> chain(feature_map(data1), adjoint_feature_map(data2))
end


# convenience for getting zero amplitude no matter
# whether the register's statevector is stored on
# cpu or gpu memory
@inline function _zero_amplitude(statevector)
    @inbounds @allowscalar begin
        statevector[1,1]
    end
end


"Sets the amplitudes of the state of the register
to the zero state's amplitudes."
function _reset_register!(register)
    # use complex number type determined
    # by the register statevector contents
    statevector = state(register)
    complex_type = typeof(_zero_amplitude(statevector))
    # reset every element of the state to zero
    # (using .= assignment rather than individually
    # assigning by index works for both cpu and gpu)
    statevector .= zero(complex_type)
    # Now set the first amplitude to 1.
    # (@allowscalar allows setting the first
    # element of the vector if it's stored on
    # the gpu)
    @inbounds @allowscalar begin
        statevector[1,1] = one(complex_type)
    end
    register
end


"Use this to simulate the circuit described by the kernel block
for the given input data arguments. Will run on cpu or gpu depending
on how the worker that executes it is configured to run tasks (see
make_register for how this is implemented).
An optional register can be given to avoid register reallocation,
and it will be automatically reset to the zero state before the
gate is run."
function apply_kernel(kernel, d1, d2, qreg=nothing)

    # Create the kernel block from the parameterised
    # kernel function by applying the parameters.
    kernel_block = kernel(d1, d2)

    qubit_count = nqubits(kernel_block)

    # Create quantum register for
    # the circuit to operate on.
    # Use the provided register after
    # resetting it or use the worker's
    # default one if nothing was given,
    # after ensuring it's the correct
    # size
    if isnothing(qreg)
        # currently uses ComplexF64 amplitudes, but
        # ComplexF32 could be faster if the gates
        # are also converted to the same format before
        # execution.
        reg = ensure_register(qubit_count)
        _reset_register!(reg)
    else
        reg = qreg
        _reset_register!(reg)
    end
    # apply kernel circuit to the state,
    # parameterised by the data points
    reg |> kernel_block
    # Get the register's zero amplitude after
    # circuit execution.
    statevector = state(reg)
    # Return the real part of the amplitude
    # of the zero basis state (this is how
    # kernel output is defined in reference
    # paper)
    real(_zero_amplitude(statevector))
end

"Computes the kernel matrix in the special case that the vectors
of samples are the same. In this case, only half of the matrix
needs to be calculated which should greatly reduce computational
load since the circuit simulations dominate the run time."
function _compute_symmetric_kernel(kernel_block, samples, register)
    # Preallocate matrix of dimension*dimension size 
    # for storing kernel outputs.
    dimension = length(samples)

    # must be created before kernel calculation loop
    # as the main diagonal entries write directly
    # to this
    result = zeros(Float64, (dimension, dimension))

    @inbounds for (i, s1) in enumerate(samples)
        # main diagonal is all 1.0's
        result[i, i] = 1.0
        # lower half of matrix is explicitly
        # computed
        @inbounds for j in 1:(i-1)
            s2 = samples[j]
            output = apply_kernel(kernel_block, s1, s2, register)
            result[i, j] = output
            result[j, i] = output
        end
    end
    return result
end


"Computes the kernel matrix for a set of training points and
other samples."
function compute_kernel_matrix(kernel_block, training_samples, other_samples)
    # ensure that each worker process has
    # a register of the correct size
    n_qubits = nqubits(kernel_block(training_samples[1], training_samples[1]))
    # make sure a register of the correct size is available for computing this
    # kernel matrix
    register = ensure_register(n_qubits)
    # handle special case of computing gram matrix
    if training_samples == other_samples
        return _compute_symmetric_kernel(kernel_block, training_samples, register)
    end
    # handle general case for train and test samples
    M = length(other_samples)
    N = length(training_samples)
    result = zeros(Float64, (M, N))
    # calculate kernel outputs
    @inbounds for (j, s2) in enumerate(training_samples)
        for (i, s1) in enumerate(other_samples)
            output = apply_kernel(kernel_block, s1, s2, register)
            result[i, j] = output
        end
    end
    # return result kernel matrix
    return result
end


"Calculates size penalty for the chromosome.
The formula is (N_Local + 2*N_CNOT) / N_Qubits,
where N_Local is the number of local gates, N_CNOT
is the number of CNOT gates, and N_Qubits is the
number of qubits in the circuit."
function size_metric(chromosome, qubit_count, count_identities=false)
    n_cnot = 0
    n_local = 0
    
    # this array is copied from decode_chromosome_yao.
    # it can be used to determine if a gate is local,
    # entangling, or an identity
    block_cases = [0, 3, 2, 1, 1, 2, 2, 1]
    
    # convenience function for the case of a gate
    # based on the index it starts at
    function block_case(i) 
        index = chromosome[i]*4 + chromosome[i+1]*2 + chromosome[i+2] + 1
        @inbounds block_cases[index]
    end

    bits_per_gate = 5
    for i in range(start=1, step=bits_per_gate, stop=length(chromosome))
        case = block_case(i)
        if (case == 0) || (case == 1) #(hadamard or parameterised rotation)
            n_local += 1
        elseif case == 3 #cnot case
            n_cnot += 1
        elseif count_identities # identity
            n_local += 1
        end
    end
    return (n_local + 2*n_cnot) / qubit_count
end


function weighted_size_metric(size_metric, accuracy)
    return size_metric + size_metric * accuracy * accuracy
end

"This struct can be used to group together
all state needed to apply a trained classifier
to new data."
struct TrainedModel
    classifier::PyObject
    kernel::Any # real type is a function of 2 arguments that returns a circuit block
    gram_matrix::Matrix{Float64}
    training_set::Vector{Vector{Float64}}
end

"Returns the classifier outputs from a trained model
given other points for input as well as the kernel
block and training set needed for computing kernel values."
function classify(model::TrainedModel, points)
    matrix = compute_kernel_matrix(model.kernel,
                                   model.training_set,
                                   points)
    return predict(model.classifier, matrix)
end

"Converts Int64 labels to Float64 values of +1.0 and -1.0."
function replace_labels(labels)
    label_set = [l for l in Set(labels)]
    label_map = Dict(label_set[1]=>-1.0, label_set[2]=>1.0)
    return [label_map[l] for l in labels]
end

"Used to create a trained model given a train/test
sample split and a kernel circuit block."
function train_model(problem_data, kernel, seed=22)
    # set python's random seed
    np.random.seed(seed)
    # unpack problem data
    (train_samples, test_samples, train_labels, test_labels) = problem_data
    # compute gram matrix
    gram_matrix = compute_kernel_matrix(kernel, train_samples, train_samples)
    # create model with sklearn
    model = SVC(kernel="precomputed")
    # fit model
    fit!(model, gram_matrix, train_labels)
    # group together the data needed
    # to classify with the model.
    return TrainedModel(model, kernel, gram_matrix, train_samples)
end


"Calculates the accuracy of a kernel circuit."
function accuracy_metric_yao(model_struct, problem_data, kernel=model_struct.kernel)
    # unpack problem data for test points and labels
    (train_samples, test_samples, train_labels, test_labels) = problem_data
    # compute kernel outputs for test set points
    test_sample_kernel_values =  compute_kernel_matrix(kernel, train_samples, test_samples)
    # score model
    accuracy = score(model_struct.classifier, test_sample_kernel_values, test_labels)
    return accuracy
end


# NOTE/TODO: could use average of margin sizes of each test
# data point to allow for gradual improvement of the metric,
# although optimizing the minimum also works
"Return minimum of the margin sizes for each test set data
point. The margin size for a data point is calculated as the
product of the data point label (-1 or 1) and the output of the
classifier on the data. In this way, incorrect classifications
have negative margin size, and correct classifications have
positive margin size."
function margin_width_metric_yao(model_struct, problem_data)
    # unpack problem data to get train_labels
    (train_samples, test_samples, train_labels, test_labels) = problem_data
    # classify training data points to get the margin width.
    # reuse gram matrix calculated with the model for speed.
    outputs = decision_function(model_struct.classifier, model_struct.gram_matrix)
    confidences = [label * output for (label, output) in zip(train_labels, outputs)]
    #return sum(confidences) / length(confidences)  #this line was tested before and worked to train good classifiers
    return minimum(confidences)
end


"Calculates fitness of a single solution."
function fitness_yao(chromosome, feature_count, qubit_count, depth, problem_data, seed=22)
    # create the kernel block constructor from the chromosome and circuit details
    kernel = decode_chromosome_yao(chromosome, feature_count, qubit_count, depth)

    # train a model using the kernel, problem data, and seed
    model_struct = train_model(problem_data, kernel, seed)
    # calculate some fitness metrics
    sm = size_metric(chromosome, qubit_count)
    acc = accuracy_metric_yao(model_struct, problem_data, kernel)
    margin_metric = margin_width_metric_yao(model_struct, problem_data)
    # return the metrics
    return (acc, weighted_size_metric(sm, acc), margin_metric)
end

using SymEngine
using YaoSym
"Calculates fitness of a single solution. Evaluates the kernel once symbolically to save time."
function fitness_yao_symbolic(chromosome, feature_count, qubit_count, depth, problem_data, seed=22)
    # create the kernel block constructor from the chromosome and circuit details
    kernel = decode_chromosome_yao(chromosome, feature_count, qubit_count, depth)
    # substitute symbolic feature values into the circuit
    data_1 = [SymEngine.symbols("x_$i") for i in 1:feature_count]
    data_2 = [SymEngine.symbols("y_$i") for i in 1:feature_count]
    symbolic_kernel_circuit = kernel(data_1, data_2)

    # create a symbolic-compatible register
    register = zero_state(Basic, qubit_count)

    # run the symbolic circuit
    register |> symbolic_kernel_circuit

    # get the zero state amplitude as a symbolic expression
    # in terms of the symbolic feature values
    zero_amplitude_expression = state(register)[1,1]
    # compile the expression into a function returning the definite zero amplitude
    # when given feature values
    lambdified_fn = eval(Expr(:function,
                         Expr(:call, gensym(), map(Symbol,vcat(data_1, data_2))...),
                         convert(Expr, zero_amplitude_expression)))
    fast_zero_amplitude_calculator(args...) = Base.invokelatest(lambdified_fn, args...)
    # use that to define a function directly producing the kernel output
    fast_kernel_function(d1, d2) = real(fast_zero_amplitude_calculator(d1..., d2...))

    # hacked together symbolic-compatible local versions of global functions
    function compute_symmetric_kernel_matrix_symbolic(training_samples)
        dimension = length(training_samples)
        result = zeros(Float64, (dimension, dimension))
        @inbounds for (i, s1) in enumerate(training_samples)
            result[i,i] = 1.0
            @inbounds for j in 1:(i-1)
                s2 = training_samples[j]
                output = fast_kernel_function(s1, s2)
                result[i,j] = output
                result[j,i] = output
            end
        end
        return result
    end
    function compute_kernel_matrix_symbolic(training_samples, other_samples)
        if training_samples == other_samples
            return compute_symmetric_kernel_matrix_symbolic(training_samples)
        end
        M = length(other_samples)
        N = length(training_samples)
        result = zeros(Float64, (M, N))
        @inbounds for (j, s2) in enumerate(training_samples)
            for (i, s1) in enumerate(other_samples)
                output = fast_kernel_function(s1, s2)
                result[i,j] = output
            end
        end
        return result
    end
    function train_model_symbolic(problem_data, seed=22)
        np.random.seed(seed)
        (train_samples, test_samples, train_labels, test_labels) = problem_data
        gram_matrix = compute_kernel_matrix_symbolic(train_samples, train_samples)
        model = SVC(kernel="precomputed")
        fit!(model, gram_matrix, train_labels)
        return TrainedModel(model, kernel, gram_matrix, train_samples)
    end
    function accuracy_metric_yao_symbolic(model_struct, problem_data)
        (train_samples, test_samples, train_labels, test_labels) = problem_data
        test_sample_kernel_values =  compute_kernel_matrix_symbolic(train_samples, test_samples)
        accuracy = score(model_struct.classifier, test_sample_kernel_values, test_labels)
        return accuracy
    end

    # train a model using the fast kernel, problem data, and seed.
    # the fast kernel should be advantageous since many kernel evaluations
    # will be performed to construct the kernel matrix
    model_struct = train_model_symbolic(problem_data, seed)
    # calculate some fitness metrics
    sm = size_metric(chromosome, qubit_count)
    acc = accuracy_metric_yao_symbolic(model_struct, problem_data)
    margin_metric = margin_width_metric_yao(model_struct, problem_data)
    # return the metrics
    return (acc, weighted_size_metric(sm, acc), margin_metric)
end

"Calculates fitness of a single solution."
function fitness_yao_margin(chromosome, feature_count, qubit_count, depth, problem_data, seed=22)
    # create the kernel block constructor from the chromosome and circuit details
    kernel = decode_chromosome_yao(chromosome, feature_count, qubit_count, depth)
    # train a model using the kernel, problem data, and seed
    model_struct = train_model(problem_data, kernel, seed)
    # calculate some fitness metrics
    sm = size_metric(chromosome, qubit_count)
    acc = accuracy_metric_yao(model_struct, problem_data, kernel)
    margin_metric = margin_width_metric_yao(model_struct, problem_data)
    # return the metrics
    return (margin_metric, weighted_size_metric(sm, acc), acc)
end

"Calculates the fitness of a chromosome given problem data being a tuple with a vector of all
data set samples and a vector of all corresponding labels. The accuracy metric is calculated
using 5-fold cross-validation."
function fitness_yao_cross_validation(chromosome, feature_count, qubit_count, depth,
                                      problem_data, seed=22)
    # helper function for computing averages (means)
    mean(xs) = sum(xs) / length(xs)
    # unpack problem data into
    # an array of all samples
    # and an array of all labels
    (samples, labels) = problem_data
    # decode the chromosome
    kernel_block = decode_chromosome_yao(chromosome, feature_count, qubit_count, depth)
    # compute the kernel values for every pair of data points in the data set.
    # this is faster than splitting the data set into training and testing sets
    # multiple times and then computing kernel matrices for each split
    sample_kernel_matrix = compute_kernel_matrix(kernel_block, samples, samples)
    # use 5-fold cross validation
    n_folds = 5
    n_samples = length(samples)
    # create the iterable of train and test set indices for each
    # of the splits
    kf = CrossValidation.KFold(n_samples, n_folds=n_folds, shuffle=true, random_state=seed)
    # iterate through each choice of split, recording the accuracy and margin metrics
    # for each choice
    metric_vectors::Vector{Vector{Float64}} = [[], [], []] # this stores the metrics for each split
    for (train_indices, test_indices) in kf
        # extract the gram matrix for this split from sample_kernel_matrix
        gram_matrix = [[sample_kernel_matrix[i, j] for j in train_indices]
                        for i in train_indices]
        # extract labels
        train_labels = [labels[i] for i in train_indices]
        # create and train the model for this split
        np.random.random(seed)
        model = SVC(kernel="precomputed")
        fit!(model, gram_matrix, train_labels)
        # compute the margin metric
        train_set_outputs = decision_function(model, gram_matrix)
        train_set_confidences = [l * o for (l, o) in zip(train_labels, train_set_outputs)]
        #margin_metric = mean(train_set_confidences)
        margin_metric = minimum(train_set_confidences)
        # compute the accuracy metric
        test_labels = [labels[i] for i in test_indices]
        test_kernel_outputs = [[sample_kernel_matrix[i, j] for j in train_indices]
                                for i in test_indices]
        accuracy_metric = score(model, test_kernel_outputs, test_labels)
        # compute the weighted size metric.
        # this must be computed for each data split
        # since it depends on the accuracy metric.
        sm = size_metric(chromosome, qubit_count)
        wsm = weighted_size_metric(sm, accuracy_metric)
        # store the metrics for this split
        push!(metric_vectors[1], accuracy_metric)
        push!(metric_vectors[2], wsm)
        push!(metric_vectors[3], margin_metric)
    end
    # average the metrics to return a final metric
    return (mean(metric_vectors[1]), mean(metric_vectors[2]), mean(metric_vectors[3]))
end

# NOTE:
# Workers could also be configured to compute with multiple GPUs.
# By using the CUDA.jl package ("using CUDA"), the "device!" function
# can be used with a do block to run code with the specified device as
# the default for running tasks. could try running gpu tasks with
# "device!(id) do ... end"  on each worker, where id is the GPU's index,
# with GPU's indexed from 0
