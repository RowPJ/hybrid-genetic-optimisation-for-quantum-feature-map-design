# This file contains some tests functions that can be used to
# manually inspect whether parts of the program function
# correctly or not.

#TODO: fix circuit drawing where parameters are not substituted so that the text fits
# in the boxes

###################################
#- Test / convenience functions. -#
###################################
using YaoPlots
#using SymPy # for drawing plots with symbolic parameter values
import Cairo, Fontconfig # for displaying plots
using JLD2 # for saving and loading objects
using SymPy # for plotting kernels with parameters visible
using ProfileSVG # for visual profiling

"Draws a large random kernel circuit to test plotting."
function plot_test()
    many_qubits = 8
    large_depth = 4
    many_features = 5
    features = 2 .* rand(many_features) .- 1
    one_features = ones(many_features)
    bits_per_gate = 5
    bit_count = bits_per_gate * many_qubits * large_depth
    big_chromosome = Int64.(round.(rand(bit_count)))
    kernel = decode_chromosome_yao(big_chromosome, many_features, many_qubits, large_depth)(one_features, one_features)
    YaoPlots.plot(kernel)
end

"Makes a random chromosome for a kernel on \"size\" qubits."
function random_chromosome(qubits, depth)
    bits_per_gate = 5
    bit_count = bits_per_gate * qubits * depth
    chromosome = Int64.(round.(rand(bit_count)))
    return chromosome
end

"Creates a population of random chromosomes."
function random_population(pop_size, qubits, depth)
    [random_chromosome(qubits, depth) for i in 1:pop_size]
end


"Tests kernel application pipeline for a given size circuit on cpu or gpu."
function kernel_test(size, gpu)
    many_qubits = size
    large_depth = 6
    many_features = 5
    # preallocate register for speed
    qreg = gpu ? cu(zero_state(ComplexF64, many_qubits)) : zero_state(ComplexF64, many_qubits)
    one_features = ones(many_features)
    bits_per_gate = 5
    bit_count = bits_per_gate * many_qubits * large_depth
    big_chromosome = Int64.(round.(rand(bit_count)))
    kernel = decode_chromosome_yao(big_chromosome, many_features, many_qubits, large_depth)
    for i in 1:(150^2)
        #@time begin
            #features = 2 .* rand(many_features) .- 1
            apply_kernel(kernel, one_features, one_features, qreg)
        #end
    end
end


function profile_kernel(size=5)
    @ProfileSVG.profview @time kernel_test(size, false)
end


"Use this to quickly inspect the circuit corresponding to a chromosome."
function draw_chromosome(d1::Vector{Float64}, d2::Vector{Float64}, chromosome::AbstractVector, args...)
    kernel = decode_chromosome_yao(chromosome, args...)(d1, d2)
    YaoPlots.plot(kernel)
end

function plot_test_symbolic()
    many_qubits = 8
    large_depth = 4
    many_features = 2
    bits_per_gate = 5
    bit_count = bits_per_gate * many_qubits * large_depth
    big_chromosome = Bool.(round.(rand(bit_count)))
    draw_chromosome(big_chromosome, many_features, many_qubits, large_depth)
end


"Version for when data points aren't supplied (uses symbolic values instead)."
function draw_chromosome(chromosome::AbstractVector, feature_count, args...)
    kernel = decode_chromosome_yao(chromosome, feature_count, args...; plotting=true)
    draw_kernel(kernel, feature_count)
end

function draw_kernel(kernel, feature_count)
    x = SymPy.symbols("x[0:$feature_count]", real=true)
    y = SymPy.symbols("y[0:$feature_count]", real=true)
    circuit = kernel(x, y)
    YaoPlots.plot(circuit)
end


function profile_kernel_matrix(size=5)
    # generate training and testing set
    samples = 150
    percent_train = 0.7
    num_train = Int64(floor(samples * percent_train))
    num_test = samples -  num_train
    features = 5
    train_set = [rand(features) for i in 1:num_train]
    train_labels = [Int64(round(rand())) for i in 1:num_train]
    test_set = [rand(features) for i in 1:num_test]
    test_labels = [Int64(round(rand())) for i in 1:num_test]
    # make a random kernel block
    depth = 6
    bits_per_gate = 5
    bit_count = bits_per_gate * size * depth
    random_chromosome = Int64.(round.(rand(bit_count)))
    kernel_block = decode_chromosome_yao(random_chromosome, features, size, depth)
    @ProfileSVG.profview @time begin
        # compute gram matrix
        compute_kernel_matrix(kernel_block, train_set, train_set)
        # compute test set matrix
        compute_kernel_matrix(kernel_block, train_set, test_set)
    end
end


"Plots the decision boundary of a classifier for a 2D data set."
function plot_decision_boundary(model_struct, samples, labels)
end


"Saves an experiment output to files named based on dataset_string and
metric_string. EG: dataset_string == \"moons\" and metric_string == \"margin\"
will save \"moons margin final_population.jld2\" and \"moons margin final_fitnesses.jld2\"
files."
function save_results(dataset_string, metric_string, result_tuple)
    population = result_tuple[1]
    fitnesses = result_tuple[2]
    fitness_history = result_tuple[4]

    row(m, i) = @view m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]

    population = to_rows(population)
    fitnesses = to_rows(fitnesses)
    fitness_history = convert(Vector{Vector{Vector{Float64}}}, fitness_history)

    jldsave("./results/$dataset_string $metric_string final_population.jld2"; population)
    jldsave("./results/$dataset_string $metric_string final_fitnesses.jld2"; fitnesses)
    jldsave("./results/$dataset_string $metric_string fitness_history.jld2"; fitness_history)
    
    nothing
end

function save_parameter_results(dataset_string, metric_string, results)
    jldsave("./results/$dataset_string $metric_string parameter_training_results.jld2"; results)
end
function load_parameter_results(dataset_string, metric_string)
    JLD2.load("./results/$dataset_string $metric_string parameter_training_results.jld2")["results"]
end

"Like save_results but takes only the first two arguments and returns the loaded
population and fitness values."
function load_results(dataset_string, metric_string)
    population = JLD2.load("./results/$dataset_string $metric_string final_population.jld2")["population"]
    population = [p[1] for p in population]
    fitnesses = JLD2.load("./results/$dataset_string $metric_string final_fitnesses.jld2")["fitnesses"]
    fitnesses = [f[1] for f in fitnesses]
    fitness_history = JLD2.load("./results/$dataset_string $metric_string fitness_history.jld2")["fitness_history"]
    (population, fitnesses, fitness_history)
end

###OUTDATED, expects to send a data set struct to optimize_kernel_target_alignment rather than a problem_data tuple
#=
"Use to test optimizing kernel target alignment.
Depends on cancer accuracy results with 4 features,
6 qubits, and depth 6 existing for them to be loaded."
function target_alignment_optimization_test(seed=22)
    row(m, i) = @view m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    
    # load an optimized chromosome and create a parameterised kernel
    cancer_acc_results = load_results("cancer", "accuracy")
    best_individual = cancer_acc_results[1][4]
    best_fitness = cancer_acc_results[2][4]
    n_features = 4
    parameterised_kernel, initial_parameters = decode_chromosome_parameterised_yao(best_individual, n_features, 6, 6)

    # optimize kernel
    results = optimize_kernel_target_alignment(parameterised_kernel, initial_parameters, cancer_dataset)
    return results
end


"Like target_alignment_optimization_test, but tests optimizing parameters for test set classification accuracy."
function accuracy_optimization_test(seed=22)
    row(m, i) = @view m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    
    # load an optimized chromosome and create a parameterised kernel
    cancer_acc_results = load_results("cancer", "accuracy")
    best_individual = cancer_acc_results[1][4]
    best_fitness = cancer_acc_results[2][4]
    n_features = 4
    parameterised_kernel, initial_parameters = decode_chromosome_parameterised_yao(best_individual, n_features, 6, 6)

    # optimize kernel
    results = optimize_kernel_accuracy(parameterised_kernel, initial_parameters, cancer_dataset)
    return results
end
=#

#TODO: complete this function
function roc_curve_test()
    pop, fit = load_results("digits", "accuracy")
end
