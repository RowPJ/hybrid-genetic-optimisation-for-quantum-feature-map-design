# TODO: create animation of decision boundary graph changing with
# threshold choice

using Plots, ScikitLearn
using Compose
using StatsPlots # for groups bar graph
using Statistics # for mean and std deviation functions

py"""
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
"""

function roc_curve(model_struct::TrainedModel, dataset::Dataset; approach_number=0)
    # compute kernel values for model's training data and the dataset's training samples
    kernel_outputs = compute_kernel_matrix(model_struct.kernel, model_struct.training_set, dataset.validation_samples)
    # compute decision function outputs (range from -1 to 1)
    df_outputs = decision_function(model_struct.classifier, kernel_outputs)
    # record number of true and false positives for various selections of a thresholding boundary
    true_positive_rates, false_positive_rates = [], []
    width=0.001
    for boundary in -3:width:3
        # label samples using the current boundary
        predictions = [o < boundary ? -1 : 1 for o in df_outputs]
        # count true and false positives
        true_positive_p = ((true_label, predicted_label),) -> predicted_label == 1 && true_label == 1
        false_positive_p = ((true_label, predicted_label),) -> predicted_label == 1 && true_label == -1
        tp = count(true_positive_p, zip(dataset.validation_labels, predictions))
        fp = count(false_positive_p, zip(dataset.validation_labels, predictions))
        tpr = tp / dataset.num_positive_validation_instances
        fpr = fp / dataset.num_negative_validation_instances
        push!(true_positive_rates, tpr)
        push!(false_positive_rates, fpr)
    end

    # calculate AUC (area under curve)
    # sort false positive rates (and true positive rates correspondingly), then sum rectangular areas to approximate
    #auc = 0.0
    #for pair in zip(true_positive_rates, false_positive_rates)
            #auc += pair[1] * width / length
    #end
    #auc = round(auc, digits=4)
    dataset_name = dataset.name
    title = "ROC Curve, ($dataset_name, Approach $approach_number)"#, AUC=$auc"
    Plots.plot(false_positive_rates, true_positive_rates, plot_title=title, xlabel="FP rate", ylabel="TP rate")
end

###UNUSED
#=
# dispatch version that takes a chromosome
function roc_curve(chromosome::AbstractVector{Bool}, dataset::Dataset, n_qubits, depth; approach_number=0, seed=22)
    n_features = dataset.feature_count
    kernel = decode_chromosome_yao(chromosome, n_features, n_qubits, depth)
    roc_curve(kernel, dataset; approach_number=approach_number, seed=seed)
end

# dispatch version that takes a kernel and first creates the model
function roc_curve(kernel, dataset::Dataset; approach_number=0, seed=22)
    problem_data = py"train_test_split"(dataset.training_samples, dataset.training_labels, train_size=0.7, random_state=seed)
    model = train_model(problem_data, kernel)
    roc_curve(model, dataset; approach_number=approach_number)
end
=#

function decision_boundary(model_struct, dataset; approach_number=0)
    function point_to_output(x1, x2)
        # take a coordinate and output its classification
        kernel_output = compute_kernel_matrix(model_struct.kernel,
                                              model_struct.training_set,
                                              [[x1, x2]])
        prediction = predict(model_struct.classifier, kernel_output)[1]
        # convert prediction to 1 or 2 for input to Plots
        return prediction == -1 ? 1 : 2
    end

    axis_interval = -1.2:.01:1.2

    ccol = cgrad([RGB(1,.2,.2), RGB(.2,.2,1)])
    #mcol = [RGB(1,.1,.1) RGB(.1,.1,1)]

    name = dataset.name
    contour(axis_interval, axis_interval, point_to_output,
            f=true, nlev=2, c=ccol, leg=:none, title="Decision Boundary ($name, Approach $approach_number)",
            xlabel="Feature 1", ylabel="Feature 2")
    scatter!([s[1] for s in dataset.validation_samples],
             [s[2] for s in dataset.validation_samples],
             m=[:rect :circle],
             color=[label == -1 ? "red" : "blue" for label in dataset.validation_labels],
             lims=(-1.2, 1.2))
end

function plot_final_fitness_pareto_front(final_fitnesses; dataset_name="undef", training_type="undef", approach_number=0)
    xs = [final_fitnesses[i][2] for i in 1:length(final_fitnesses)] #size values
    ys = [-final_fitnesses[i][1] for i in 1:length(final_fitnesses)] #accuracy values
    # plot pareto front with 
    Plots.plot(xs, ys,
               seriestype=:scatter,
               title="Final generation pareto front ($dataset_name, Approach $approach_number)",
               xlim=(0, 6), ylim=(0.35, 1),
               xlabel="Size metric", ylabel="Accuracy metric",
               legend=false)
end

#TODO: compute xlim bounds by checking the max and min of xs over all generations
"Animates the history of multi-objective fitness values changing
over genetic optimization generations. This can be used to visualize
the convergence of the genetic optimization."
function animate_genetic_fitness_history(fitness_history; dataset_name="undef", training_type="undef", ylabel="Accuracy metric", ylim=(0.35, 1))
    target_seconds = 10
    frame_rate = 30
    target_frames = frame_rate * target_seconds
    available_frames = length(fitness_history)
    if target_frames > available_frames
        error("Decrease frame rate or decrease the desired animation length; " +
              "there are insufficient generations for the current values.")
    end
    step = length(fitness_history) ÷ target_frames
    animation = @animate for (gen, fitnesses) in enumerate(fitness_history)
        xs = [fitnesses[i][2] for i in 1:length(fitnesses)] #size values
        ys = [-fitnesses[i][1] for i in 1:length(fitnesses)] #accuracy values
        # plot pareto front with 
        plt = Plots.plot(xs, ys,
                         seriestype=:scatter,
                         title="Pareto front, generation $gen",#"Change of pareto front with genetic optimization",
                         xlim=(0, 6), ylim=ylim,
                         xlabel="Size metric", ylabel=ylabel,
                         legend=false)
        # annotate with generation number
        annotate!([(10, 10, Plots.text("Generation: $gen", :black, :right, 3))])
        plt
    end every step
    gif(animation, "./diagrams/fitness_history_$(dataset_name)_$(training_type).gif", fps=frame_rate)
end

#OUTDATED SINCE ACCURACY ISN'T DIRECTLY TRAINED ANYMORE
#=
#TODO: add support for parameter metrics other than accuracy; take the parameter values and chromosomes 
# as an argument as well and calculate the accuracy metric for the data set, or just only accept
# accuracy for the parameter fitness and do the conversion of fitness histories elsewhere
function visualize_genetic_and_parameter_training(genetic_fitness_history, parameter_fitness_history, dataset, genetic_training_type, parameter_training_type, seed=22)
    target_seconds = 6
    frame_rate = 15
    target_frames = frame_rate * target_seconds
    genetic_training_iterations = length(genetic_fitness_history)
    total_parameter_training_evaluations = maximum(map(length, parameter_fitness_history))
    available_frames = genetic_training_iterations + total_parameter_training_evaluations
    if target_frames > available_frames
        error("Decrease frame rate or decrease the desired animation length; " +
              "there are insufficient generations and/or parameter-training " +
              "fitness evaluations for the current values.")
    end
    step = length(genetic_fitness_history) ÷ target_frames
    final_wait_seconds = 5
    final_wait_frames = final_wait_seconds * step * frame_rate
    animation = @animate for iteration in 1:(available_frames+final_wait_frames)
        if iteration <= genetic_training_iterations
            fitnesses = genetic_fitness_history[iteration]
            # case of animating genetic fitness changes
            xs = [fitnesses[i][2] for i in 1:length(fitnesses)] #size values
            ys = [-fitnesses[i][1] for i in 1:length(fitnesses)] #accuracy values
            # plot pareto front with 
            plt = Plots.plot(xs, ys,
                             seriestype=:scatter,
                             title="Pareto front, generation $iteration",
                             xlim=(0, 6), ylim=(0.35, 1),
                             xlabel="Size metric", ylabel="Accuracy metric",
                             legend=false)
            # annotate with generation number
            annotate!([(10, 10, Plots.text("Generation: $iteration", :black, :right, 3))])
            plt
        elseif iteration <= genetic_training_iterations + total_parameter_training_evaluations
            # case of animating parameter training fitness changes
            parameter_training_evaluations = iteration - genetic_training_iterations
            final_genetic_fitnesses = genetic_fitness_history[end]
            xs = [g[2] for g in final_genetic_fitnesses]
            ys = [g[parameter_training_evaluations] for g in parameter_fitness_history]
            plt = Plots.plot(xs, ys,
                             seriestype=:scatter,
                             title="Pareto front, $parameter_training_evaluations parameter fitness evaluations",
                             xlim=(0,6), ylim=(0.35, 1),
                             xlabel="Size metric", ylabel="Accuracy metric",
                             legend=false)
            # annotate with generation number
            annotate!([(10, 10, Plots.text("Fitness evaluations: $parameter_training_evaluations", :black, :right, 3))])
            plt
        else
            # case of inserting waiting frames to extend animation length
            #(do nothing)
        end
    end every step
    dataset_name = dataset.name
    gif(animation, "./diagrams/pareto_front_change_with_parameter_training $dataset_name $genetic_training_type $parameter_training_type.gif", fps=frame_rate)
end
=#

"Returns a latex string for drawing a chromosome with Quantikz"
function draw_chromosome_latex(chromosome, feature_count, qubit_count, depth; parameters=nothing, rounding=4) # rounding specifies how many digits to round proportionality parameters to
    next_qubit_index(q) = q % qubit_count + 1
    block_cases = [0, 3, 2, 1, 1, 2, 2, 1]
    angle_map = [pi, pi/2, pi/4, pi/8]
    function extract_initial_parameters()
        initial_parameters::Vector{Float64} = []
        bits_per_gate = 5
        for (i, h) in enumerate(range(start=1, step=bits_per_gate, stop=length(chromosome)))
            j = (i-1) % qubit_count + 1
            k = (i-1) % feature_count + 1
            mapping_index = (chromosome[h]*4 + chromosome[h+1]*2 + chromosome[h+2]) + 1
            block_case = @inbounds block_cases[mapping_index]
            # only account for block case 1, the case of parameterised gates
            if block_case == 1
                proportionality_parameter_index = (chromosome[h+3]*2 + chromosome[h+4]) + 1
                proportionality_parameter = angle_map[proportionality_parameter_index]
                push!(initial_parameters, proportionality_parameter)
            end
        end
        return initial_parameters
    end
    # if parameters weren't supplied, just extract the defaults from the chromosome
    if isnothing(parameters)
        parameters = extract_initial_parameters()
    end

    # entries of cases for string placement. not building directly to a string allows simplifying circuits before saving them
    qubit_entries::Vector{Vector{Tuple}} = [[] for i in 1:qubit_count]

    # functions for each case that take a qubit number, proportionality parameter, and parameter name for
    # the encoded data point and edit the qubit strings to add that gate to the circuit
    function apply_hadamard(qubit, prop_param, data_param)
        push!(qubit_entries[qubit], ("H",))
    end
    function apply_cnot(qubit, prop_param, data_param)
        target_qubit = next_qubit_index(qubit)
        # calculate offset of the target qubit
        offset = target_qubit - qubit # this will either be 1 or -(qubit_count-1).
        if offset < 0
            push!(qubit_entries[qubit], ("Empty",))# empty
            push!(qubit_entries[qubit], ("Ctrl", offset))# target
        else
            push!(qubit_entries[qubit], ("Ctrl", offset))
        end 
        push!(qubit_entries[target_qubit], ("Targ",))
    end
    function apply_empty_gate(qubit, prop_param, data_param)
        push!(qubit_entries[qubit], ("Empty",))
    end
    function apply_rz(qubit, prop_param, data_param)
        push!(qubit_entries[qubit], ("Rz", prop_param, data_param))
    end
    function apply_rx(qubit, prop_param, data_param)
        push!(qubit_entries[qubit], ("Rx", prop_param, data_param))
    end
    function apply_ry(qubit, prop_param, data_param)
        push!(qubit_entries[qubit], ("Ry", prop_param, data_param))
    end
    # mapping from block case to applier function
    block_appliers = [apply_hadamard, # hadamard
                        apply_cnot, #cnot
                        apply_empty_gate, #empty block case
                        apply_rx, #Rx
                        apply_rz, #Rz
                        apply_empty_gate, #empty block case
                        apply_empty_gate, #empty block case
                        apply_ry] #Ry
    
    # define function that applies the gates of the feature map to the circuit
    function feature_map()
        current_parameter_index = 1
        bits_per_gate = 5
        for (i, h) in enumerate(range(start=1, step=bits_per_gate, stop=length(chromosome)))
            j = (i-1) % qubit_count + 1
            k = (i-1) % feature_count + 1
            mapping_index = (chromosome[h]*4 + chromosome[h+1]*2 + chromosome[h+2]) + 1
            block_case = @inbounds block_cases[mapping_index]
            block_applier = @inbounds block_appliers[mapping_index]
            
            if block_case == 1
                # if the gate is parameterised, consume a parameter
                # then apply the gate
                parameter = parameters[current_parameter_index]
                current_parameter_index += 1

                block_applier(j, round(parameter, digits=rounding), "x_{$(k-1)}") # data parameter value is x_k for kth data value
            else
                # otherwise, just apply the gate without parameters
                block_applier(j, nothing, nothing)
            end
        end
    end

    # trigger qubit_entries to be filled in
    feature_map()
    
    wires_to_remove = []
    layers_to_remove = []
    "Remove empty layers and wires from qubit_entries, in place."
    function trim_qubit_entries()
        # remove empty wires
        for (wire, wire_ops) in enumerate(qubit_entries)
            # check if a wire has no operations
            if all(op->op[1] == "Empty", wire_ops)
                push!(wires_to_remove, wire)
            end
        end
        # remove empty layers
        layer_counts = [length(ops) for ops in qubit_entries]
        largest_layer_size = maximum(layer_counts)
        for layer in 1:largest_layer_size
            layer_ops = [layer_counts[qubit] < layer ? ("Empty",) : qubit_entries[qubit][layer] for qubit in 1:qubit_count]
            if all(op->op[1] == "Empty", layer_ops)
                push!(layers_to_remove, layer)
            end
        end
        return [[qubit_entries[qubit][layer] for layer in 1:length(qubit_entries[qubit]) if !(layer in layers_to_remove)] for qubit in 1:qubit_count if !(qubit in wires_to_remove)]
    end
    qubit_entries = trim_qubit_entries()

    # strings for each qubit's gates.
    # these need to be combined when building the result
    qubit_strings = [i in wires_to_remove ? "" : raw"    \lstick{$\ket{0}$} " for i in 1:qubit_count] # initially have a 0 ket for each used qubit, and an empty string otherwise
    "Use qubit_entries to fill in qubit_strings."
    function build_qubit_strings()
        for (qubit, wire_ops) in zip([i for i in 1:qubit_count if !(i in wires_to_remove)], qubit_entries)
            for op in wire_ops
                type = op[1]
                if type == "H"
                    qubit_strings[qubit] *= raw"& \gate{H} "
                elseif type == "Empty"
                    qubit_strings[qubit] *= raw"& \qw "
                elseif type == "Targ"
                    qubit_strings[qubit] *= raw"& \targ{} "
                elseif type == "Ctrl"
                    offset = op[2]
                    qubit_strings[qubit] *= "& \\ctrl{$offset} " # if the offset is less than 0, a gate has already been applied to the target, so this gate must shift down one layer for the target to be free
                elseif type == "Rx"
                    prop_param, data_param = op[2], op[3]
                    qubit_strings[qubit] *= "& \\gate{R_x($prop_param * $data_param)} "
                elseif type == "Ry"
                    prop_param, data_param = op[2], op[3]
                    qubit_strings[qubit] *= "& \\gate{R_y($prop_param * $data_param)} "
                elseif type == "Rz"
                    prop_param, data_param = op[2], op[3]
                    qubit_strings[qubit] *= "& \\gate{R_z($prop_param * $data_param)} "
                else
                    error("build_qubit_strings: unknown operation type $type.")
                end
            end
        end
    end
    build_qubit_strings()

    final_string = "\\begin{quantikz}\n"
    # combine qubit_strings into the final latex string
    for (qubit, qubit_ops) in enumerate(qubit_strings)
        #only add wires that are used
        if !(qubit in wires_to_remove)
            final_string *= qubit_ops * "& \\qw \\\\\n" # append an empty wire and \\ to end the qubit line, then add a new line character
        end
    end
    final_string *= "\\end{quantikz}\n"
    return final_string
end

#NOTE: uses python's matplotlib, pandas, and seaborn modules for drawing the plots
function confusion_matrix(model_struct::TrainedModel, test_set_samples, test_set_labels)
    # record variables for false positives, true positives, false negatives, and true negatives
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    predictions = classify(model_struct, test_set_samples)
    for (prediction, label) in zip(predictions, test_set_labels)
        if prediction != label
            # if prediction is wrong
            if label > 0
                # and sample label is positive
                # record a false negative
                fn += 1
            else
                # and sample label is negative
                # record a false positive
                fp += 1
            end
        else
            # if prediction is correct
            if label > 0
                # and sample label is positive
                # record a true positive
                tp += 1
            else
                # and sample label is negative
                # record a true negative
                tn += 1
            end
        end
    end
    # calculate total number of positive and negative samples in the test set
    total_positives = tp + fn
    total_negatives = tn + fp

    # draw grid plot with the information
    row_labels = ["Positive Prediction",
                  "Negative Prediction"]
    column_labels = ["Positive Label", "Negative Label"]
    array = [[tp, fp],
             [fn, tn]]
    dataframe = py"pd.DataFrame"(array, index=row_labels, columns=column_labels)
    #py"sn.set(font_scale=20)" # set axis label size (need to adjust to find the right scale, 20 is far too large)
    py"plt.clf()" # clear old figure
    result = py"sn.heatmap"(dataframe, linecolor="black", fmt="d", annot=true, annot_kws=py"{'fontsize':16}", cbar=false)#, annot_kws=Dict("size"=>20)) # set font size
    return result
end

function confusion_matrix(model_struct::TrainedModel, dataset::Dataset)
    confusion_matrix(model_struct, dataset.validation_samples, dataset.validation_labels)
end

### Code that executes the graph generation for each experiment configuration using the above functions goes below
# The label #SINGLE_CLASSIFIER_TYPE is for functions that generate graphs using the best individual of the final population
# of each experiment configuration. #POPULATION_TYPE is the label for functions that plot the entire final population on their output graph(s)

#=
#SINGLE_CLASSIFIER_TYPE
function generate_roc_curve_graphs(configuration)

end

#POPULATION_TYPE
function generate_pareto_front_graphs(configuration)

end

#POPULATION_TYPE
function generate_pareto_front_animations(configuration)

end

#SINGLE_CLASSIFIER_TYPE
function generate_parameter_training_accuracy_graphs(configuration)

end

#SINGLE_CLASSIFIER_TYPE
function generate_decision_boundary_graphs(configuration)
    #NOTE: this only applies to 2D data sets since they
    # are simplest to draw the graph for.
    # Currently the only 2D data set is the moons data set.
end
=#


function save_text(text, filename)
    outfile = open(filename, "w")
    write(outfile, text)
    close(outfile)
end


"Given a list of population individuals and a list of their corresponding fitness values,
returns the index of the best performing individual measured by their accuracy metric
or its substitute."
function best_individual_index(population, fitnesses) #NOTE: could remove population argument from the argument list if it won't be needed in the future
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

#=
#POPULATION_TYPE
#SINGLE_CLASSIFIER_TYPE
"Load results, then create and save graphs for each data set and approach being compared (original, parameter refinement, trained rmse metric)."
function generate_graphs_old(seed=22)
    # for each data set
    for dataset_name in [
                         "moons",
                         "digits",
                         "cancer",
                         "iris",
                         #"blobs",
                         "circles",
                         "adhoc",
                         "voice",
                         "susy",
                         "susy_hard"
                         ]
        # retrieve the dataset object corresponding to the name
        dataset::Dataset = dataset_map[dataset_name]

        # STEPS:
        # 1. load results from original genetic optimization method
        #    1.1 draw graphs for original optimization method
        #        1.1.1 final population fitnesses pareto front
        #        1.1.2 final population fitnesses pareto front animation
        #        1.1.3 final population best individual circuit
        #        1.1.4 final population best individual roc curve
        #        1.1.5 final population best individual confusion matrix
        #        1.1.6 final population best individual decision boundary (if 2D dataset)
        #    1.2 train final parameters to get 2nd approach results,
        #        and draw graphs for the 2nd approach
        #       1.2.1-1.2.6 same as 1.1.1-1.1.6
        #       1.2.7 final population individuals change in accuracy metric with parameter training (maybe can't use best individual since they already have 100% accuracy. try training all individuals and show a few of the graphs)
        #    1.3 repeat 1.2 but training parameters for target alignment instead of accuracy
        # 2. load results from genetic optimization that use trained accuracy
        #    in genetic fitness function. if the final parameters weren't saved,
        #    re-train them to retrieve the original models. the saved fitness values will
        #    still be correct
        #    2.1 draw graphs for this approach like for steps 1.1 and 1.2
        # 3. repeat 2 for the genetic optimization that optimized alignment

        # GRAPHS TO DRAW IN EACH STEP:
        # draw pareto fronts for each dataset and configuration
        # draw accuracy over pareto fronts change with successive parameter training
        # draw decision boundaries
        # draw roc curves

        println("Dataset: $dataset_name")
        println("Step 1: Approaches 1 and 2")
        # step 1
        println("Step 1.1: Approach 1 - Reproduction of referenced paper's approach")
        population, fitnesses, history = load_results(dataset_name, "accuracy")
        # step 1.1
        println("Step 1.1.1: Final pareto front")
        # 1.1.1 Pareto front
        fig = plot_final_fitness_pareto_front(fitnesses; dataset_name=dataset_name, training_type="accuracy", approach_number=1)
        savefig(fig, "./diagrams/$dataset_name accuracy final_fitness_pareto_front.pdf")
        # 1.1.2 Pareto front animation
        println("Step 1.1.2: Pareto front animation")
        animate_genetic_fitness_history(history; dataset_name=dataset_name, training_type="accuracy")
        # 1.1.3 Best individual circuit
        println("Step 1.1.3: Best individual circuit")
        best_chromosome_index = best_individual_index(population, fitnesses)
        best_chromosome = population[best_chromosome_index]
        figure = draw_chromosome_latex(best_chromosome, dataset.feature_count, 6, 6) # the draw_chromosome function is defined in tests.jl, maybe move the definition to this file
        save_text(figure, "./diagrams/$(dataset_name)_accuracy_best_individual_circuit.tex")
        
        # for the further stages of step 1, the best individual model must be instantiated from the chromosome and training dataset
        kernel = decode_chromosome_yao(best_chromosome, dataset.feature_count, 6, 6)
        
        train_percent = 0.7     # fraction of data to use for training, check nsga2.jl for the correct value to use (maybe change for flexibility or to read a variable instead of using a constant literal)
        problem_data = py"train_test_split"(dataset.training_samples,
                                            dataset.training_labels,
                                            train_size=train_percent,
                                            random_state=seed,
                                            shuffle=true)
        model_struct = train_model(problem_data, kernel, seed)
        validation_kernel_outputs = compute_kernel_matrix(model_struct.kernel, model_struct.training_set, dataset.validation_samples)
        validation_accuracy = score(model_struct.classifier, validation_kernel_outputs, dataset.validation_labels)
        print("Validation data accuracy: $validation_accuracy")
        
        # 1.1.4 Best individual ROC curve
        println("Step 1.1.4: Best individual ROC curve")
        figure = roc_curve(model_struct, dataset; approach_number=1)
        savefig(figure, "./diagrams/$dataset_name accuracy best_individual_roc_curve.pdf")
        # 1.1.5 Best individual confusion matrix
        println("Step 1.1.5: Best individual confusion matrix")
        cm = confusion_matrix(model_struct, dataset)
        # The matrix cm is not passed to the python call since the implicit pyplot state
        # still holds it. the pyplot savefig function will save the cm figure when called
        py"plt.savefig"("./diagrams/$dataset_name accuracy best_individual_confusion_matrix.pdf")
        # 1.1.6 Best individual decision boundary (for 2D datasets)
        if dataset.feature_count == 2
            println("Step 1.1.6: Best individual decision boundary")
            figure = decision_boundary(model_struct, dataset; approach_number=1)
            savefig(figure, "./diagrams/$dataset_name accuracy best_individual_decision_boundary.pdf")
        end

        println("Best individual metric values ($dataset_name approach 1):", fitnesses[best_chromosome_index])

        # Step 1.2: perform parameter training to get 2nd approach results (maybe separate training into experiments.jl and just load the results here?)
        println("Step 1.2: Approach 2.1 - rmse training")
        population_final_parameters, parameter_training_fitness_history = population_parameterised_training(population, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=seed, metric_type="rmse")
        # This below variable is the recreation of the fitnesses variable for the second approach (parameter refinement approach).
        # it holds the multi-objective fitness values for the final population after parameter training. It uses the last recorded
        # accuracy from parameter training as the accuracy and the size metric is copied from the first approach fitnesses.
        # Note that the parameterised training doesn't return negated accuracy, so it must be manually negated here to be consistent
        # with the form of the accuracies of the first approach.
        # calculate accuracy of final models when using the trained parameters
        population_trained_accuracies_rmse = []
        for (c, c_final_params) in zip(population, population_final_parameters)
            c_parameterised_kernel, c_initial_params = decode_chromosome_parameterised_yao(c, dataset.feature_count, 6, 6)
            c_model = train_model(problem_data, c_parameterised_kernel(c_final_params), seed)
            push!(population_trained_accuracies_rmse, accuracy_metric_yao(c_model, problem_data))
        end
        fitnesses_2 = [[-population_trained_accuracies_rmse[i], fitnesses[i][2]] for i in 1:length(population)]
        # 1.2.1-1.2.6 follow similarly to 1.1.1-1.1.6, so some repeated comments are left out
        # 1.2.1 Pareto front
        println("Step 1.2.1: Final pareto front")
        figure = plot_final_fitness_pareto_front(fitnesses_2; dataset_name=dataset_name, training_type="parameters rmse trained", approach_number=2.1)
        savefig(figure, "./diagrams/$dataset_name rmse trained final_fitness_pareto_front.pdf")
        # 1.2.2 Pareto front animation
        println("Step 1.2.2: Pareto front animation")
        # visualize_genetic_and_parameter_training is outdated since it only supports direct training of accuracy metric, not alignment or rmse
        #visualize_genetic_and_parameter_training(history, parameter_training_fitness_history, dataset, "accuracy", "rmse", seed)
        # 1.2.3 Best individual circuit
        println("Step 1.2.3: Best individual circuit")
        best_chromosome_index_2 = best_individual_index(population, fitnesses_2)
        best_chromosome_2 = population[best_chromosome_index_2]
        best_chromosome_trained_parameters = population_final_parameters[best_chromosome_index_2]
        best_parameterised_kernel, best_initial_parameters = decode_chromosome_parameterised_yao(best_chromosome_2, dataset.feature_count, 6, 6)
        # DEBUG PRINTING
        #=
        for (index, (chromosome, trained_params)) in enumerate(zip(population, population_final_parameters))
            _, initial_parameters = decode_chromosome_parameterised_yao(chromosome, dataset.feature_count, 6, 6)
            if length(trained_params) != length(initial_parameters)
                println("Index $index: Mismatch in trained and initial parameter lengths")
                println(chromosome)
                println(trained_params)
                println(length(trained_params))
                println(initial_parameters)
                println(length(initial_parameters))
                println()
            end
        end
        println()
        println("Best individual:")
        println(best_chromosome_2)
        println(best_initial_parameters)
        println(length(best_initial_parameters))
        println(best_chromosome_trained_parameters)
        println(length(best_chromosome_trained_parameters))
        =#
        best_kernel = best_parameterised_kernel(best_chromosome_trained_parameters)
        figure = draw_chromosome_latex(best_chromosome_2, dataset.feature_count, 6, 6; parameters=best_chromosome_trained_parameters)
        save_text(figure, "./diagrams/$(dataset_name)_rmse_trained_best_individual_circuit.tex")

        # problem_data variable is defined when graphing first approach
        model_struct_2 = train_model(problem_data, best_kernel, seed)
        validation_kernel_outputs = compute_kernel_matrix(model_struct_2.kernel, model_struct.training_set, dataset.validation_samples)
        validation_accuracy = score(model_struct_2.classifier, validation_kernel_outputs, dataset.validation_labels)
        print("Validation data accuracy: $validation_accuracy")

        # 1.2.4 Best individual ROC curve
        println("Step 1.2.4: Best individual ROC curve")
        figure = roc_curve(model_struct_2, dataset; approach_number=2.1)
        savefig(figure, "./diagrams/$dataset_name rmse trained best_individual_roc_curve.pdf")
        # 1.2.5 Best individual confusion matrix
        println("Step 1.2.5: Best individual confusion matrix")
        cm = confusion_matrix(model_struct_2, dataset)
        py"plt.savefig"("./diagrams/$dataset_name rmse trained best_individual_confusion_matrix.pdf")
        # 1.2.6 Best individual decision boundary (for 2D datasets)
        if dataset.feature_count == 2
            println("Step 1.2.6: Best individual decision boundary")
            figure = decision_boundary(model_struct_2, dataset; approach_number=2.1)
            savefig(figure, "./diagrams/$dataset_name rmse trained best_individual_decision_boundary.pdf")
        end

        println("Best individual metric values ($dataset_name approach 2):", fitnesses_2[best_chromosome_index_2])

        
        # Step 1.3: repeat 1.2 but training parameters for target alignment
        begin
            println("Step 1.3: Approach 2.2 - alignment training")
            population_final_parameters_alignment, parameter_training_fitness_history_alignment = population_parameterised_training(population, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=seed, metric_type="target_alignment")
            # calculate accuracy of final models when using the trained parameters
            population_trained_accuracies_alignment = []
            for (c, c_final_params) in zip(population, population_final_parameters_alignment)
                c_parameterised_kernel, c_initial_params = decode_chromosome_parameterised_yao(c, dataset.feature_count, 6, 6)
                c_model = train_model(problem_data, c_parameterised_kernel(c_final_params), seed)
                push!(population_trained_accuracies_alignment, accuracy_metric_yao(c_model, problem_data))
            end
            fitnesses_2_alignment = [[-population_trained_accuracies_alignment[i], fitnesses[i][2]] for i in 1:length(population)]
            # 1.3.1-1.3.6 follow similarly to 1.1.1-1.1.6, so some repeated comments are left out
            # 1.3.1 Pareto front
            println("Step 1.3.1: Final pareto front")
            figure_alignment = plot_final_fitness_pareto_front(fitnesses_2_alignment; dataset_name=dataset_name, training_type="parameters alignment trained", approach_number=2.2)
            savefig(figure_alignment, "./diagrams/$dataset_name alignment trained final_fitness_pareto_front.pdf")
            # 1.3.2 Pareto front animation - skipped since training alignment didn't update accuracies for each parameter change
            # maybe replace with a graph showing how training improved alignment?
            # 1.3.3 Best individual circuit
            println("Step 1.3.3: Best individual circuit")
            best_chromosome_index_2_alignment = best_individual_index(population, fitnesses_2_alignment)
            best_chromosome_2_alignment = population[best_chromosome_index_2_alignment]
            best_chromosome_trained_parameters_alignment = population_final_parameters_alignment[best_chromosome_index_2_alignment]
            best_parameterised_kernel_alignment, best_initial_parameters_alignment = decode_chromosome_parameterised_yao(best_chromosome_2_alignment, dataset.feature_count, 6, 6)
            best_kernel_alignment = best_parameterised_kernel_alignment(best_chromosome_trained_parameters_alignment)
            figure_alignment = draw_chromosome_latex(best_chromosome_2_alignment, dataset.feature_count, 6, 6; parameters=best_chromosome_trained_parameters_alignment)
            save_text(figure_alignment, "./diagrams/$(dataset_name)_alignment_trained_best_individual_circuit.tex")

            model_struct_2_alignment = train_model(problem_data, best_kernel_alignment, seed)
            validation_kernel_outputs = compute_kernel_matrix(model_struct_2_alignment.kernel, model_struct.training_set, dataset.validation_samples)
            validation_accuracy = score(model_struct_2_alignment.classifier, validation_kernel_outputs, dataset.validation_labels)
            print("Validation data accuracy: $validation_accuracy")
            
            # 1.3.4 Best individual ROC curve
            println("Step 1.3.4: Best individual ROC curve")
            figure_alignment = roc_curve(model_struct_2_alignment, dataset; approach_number=2.2)
            savefig(figure_alignment, "./diagrams/$dataset_name alignment trained best_individual_roc_curve.pdf")
            # 1.3.5 Best individual confusion matrix
            println("Step 1.3.5: Best individual confusion matrix")
            cm_alignment = confusion_matrix(model_struct_2_alignment, dataset)
            py"plt.savefig"("./diagrams/$dataset_name alignment trained best_individual_confusion_matrix.pdf")
            # 1.3.6 Best individual decision boundary (for 2D datasets)
            if dataset.feature_count == 2
                println("Step 1.2.6: Best individual decision boundary")
                figure_alignment = decision_boundary(model_struct_2_alignment, dataset; approach_number=2.2)
                savefig(figure_alignment, "./diagrams/$dataset_name alignment trained best_individual_decision_boundary.pdf")
            end

            println("Best individual metric values ($dataset_name approach 2 - target alignment):", fitnesses_2_alignment[best_chromosome_index_2_alignment])
        end


        #=
        # step 2
        println("Step 2: Approach 3.1 - rmse training in genetic fitness")
        population_3, fitnesses_3, history_3 = load_results(dataset_name, "rmse_parameter_training")
        population_final_parameters_3, parameter_training_fitness_history_3 = population_parameterised_training(population_3, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=seed, metric_type="rmse")
        # 2.1
        println("Step 2.1: Draw graphs from Step 1.1")
        # 2.1.1 Pareto front
        println("Step 2.1.1: Final pareto front")
        figure = plot_final_fitness_pareto_front(fitnesses_3; dataset_name=dataset_name, training_type="rmse training in fitness", approach_number=3.1)
        savefig(figure, "./diagrams/$dataset_name rmse training in fitness final_fitness_pareto_front.pdf")
        # 2.1.2 Pareto front animation
        println("Step 2.1.2: Pareto front animation")
        animate_genetic_fitness_history(history_3; dataset_name=dataset_name, training_type="rmse training in fitness")
        # 2.1.3 Best individual circuit
        println("Step 2.1.3: Best individual circuit")
        best_chromosome_index_3 = best_individual_index(population_3, fitnesses_3)
        best_chromosome_3 = population_3[best_chromosome_index_3]
        best_chromosome_trained_parameters_3 = population_final_parameters_3[best_chromosome_index_3]
        best_parameterised_kernel_3, best_initial_parameters_3 = decode_chromosome_parameterised_yao(best_chromosome_3, dataset.feature_count, 6, 6)
        best_kernel_3 = best_parameterised_kernel_3(best_chromosome_trained_parameters_3)
        figure = draw_chromosome_latex(best_chromosome_3, dataset.feature_count, 6, 6; parameters=best_chromosome_trained_parameters_3)
        save_text(figure, "./diagrams/$(dataset_name)_rmse_training_in_fitness_best_individual_circuit.tex")

        # problem_data variable is defined when graphing first approach
        model_struct_3 = train_model(problem_data, best_kernel_3, seed)
        validation_kernel_outputs = compute_kernel_matrix(model_struct_3.kernel, model_struct.training_set, dataset.validation_samples)
        validation_accuracy = score(model_struct_3.classifier, validation_kernel_outputs, dataset.validation_labels)
        print("Validation data accuracy: $validation_accuracy")
        
        # 2.1.4 Best individual ROC curve
        println("Step 2.1.4: Best individual ROC curve")
        figure = roc_curve(model_struct_3, dataset; approach_number=3.1)
        savefig(figure, "./diagrams/$dataset_name rmse training in fitness best_individual_roc_curve.pdf")
        # 2.1.5 Best individual confusion matrix
        println("Step 2.1.5: Best individual confusion matrix")
        cm = confusion_matrix(model_struct_3, dataset)
        py"plt.savefig"("./diagrams/$dataset_name rmse training in fitness best_individual_confusion_matrix.pdf")
        # 2.1.6 Best individual decision boundary (for 2D datasets)
        if dataset.feature_count == 2
            println("Step 2.1.6: Best individual decision boundary")
            figure = decision_boundary(model_struct_3, dataset; approach_number=3.1)
            savefig(figure, "./diagrams/$dataset_name rmse training in fitness best_individual_decision_boundary.pdf")
        end

        println("Best individual metric values ($dataset_name approach 3):", fitnesses_3[best_chromosome_index_3])
        
        begin
            # step 3
            println("Step 3: Approach 3.2 - alignment training in genetic fitness")
            population_3_alignment, fitnesses_3_alignment, history_3_alignment = load_results(dataset_name, "alignment_parameter_training")
            population_final_parameters_3_alignment, parameter_training_fitness_history_3_alignment = population_parameterised_training(population_3_alignment, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=seed, metric_type="target_alignment")
            # 3.1
            println("Step 3.1: Draw graphs from Step 1.1")
            # 3.1.1 Pareto front
            println("Step 3.1.1: Final pareto front")
            figure_alignment = plot_final_fitness_pareto_front(fitnesses_3_alignment; dataset_name=dataset_name, training_type="alignment training in fitness", approach_number=3.2)
            savefig(figure_alignment, "./diagrams/$dataset_name alignment training in fitness final_fitness_pareto_front.pdf")
            # 3.1.2 Pareto front animation
            println("Step 3.1.2: Pareto front animation")
            animate_genetic_fitness_history(history_3_alignment; dataset_name=dataset_name, training_type="alignment training in fitness")
            # 3.1.3 Best individual circuit
            println("Step 3.1.3: Best individual circuit")
            best_chromosome_index_3_alignment = best_individual_index(population_3_alignment, fitnesses_3_alignment)
            best_chromosome_3_alignment = population_3_alignment[best_chromosome_index_3_alignment]
            best_chromosome_trained_parameters_3_alignment = population_final_parameters_3_alignment[best_chromosome_index_3_alignment]
            best_parameterised_kernel_3_alignment, best_initial_parameters_3_alignment = decode_chromosome_parameterised_yao(best_chromosome_3_alignment, dataset.feature_count, 6, 6)
            best_kernel_3_alignment = best_parameterised_kernel_3_alignment(best_chromosome_trained_parameters_3_alignment)
            figure_alignment = draw_chromosome_latex(best_chromosome_3_alignment, dataset.feature_count, 6, 6; parameters=best_chromosome_trained_parameters_3_alignment)
            save_text(figure_alignment, "./diagrams/$(dataset_name)_alignment_training_in_fitness_best_individual_circuit.tex")

            # problem_data variable is defined when graphing first approach
            model_struct_3_alignment = train_model(problem_data, best_kernel_3_alignment, seed)
            validation_kernel_outputs = compute_kernel_matrix(model_struct_3_alignment.kernel, model_struct.training_set, dataset.validation_samples)
            validation_accuracy = score(model_struct_3_alignment.classifier, validation_kernel_outputs, dataset.validation_labels)
            print("Validation data accuracy: $validation_accuracy")
            
            # 3.1.4 Best individual ROC curve
            println("Step 3.1.4: Best individual ROC curve")
            figure_alignment = roc_curve(model_struct_3_alignment, dataset; approach_number=3.2)
            savefig(figure_alignment, "./diagrams/$dataset_name alignment training in fitness best_individual_roc_curve.pdf")
            # 3.1.5 Best individual confusion matrix
            println("Step 3.1.5: Best individual confusion matrix")
            cm_alignment = confusion_matrix(model_struct_3_alignment, dataset)
            py"plt.savefig"("./diagrams/$dataset_name alignment training in fitness best_individual_confusion_matrix.pdf")
            # 3.1.6 Best individual decision boundary (for 2D datasets)
            if dataset.feature_count == 2
                println("Step 3.1.6: Best individual decision boundary")
                figure_alignment = decision_boundary(model_struct_3_alignment, dataset; approach_number=3.2)
                savefig(figure_alignment, "./diagrams/$dataset_name alignment training in fitness best_individual_decision_boundary.pdf")
            end

            println("Best individual metric values ($dataset_name approach 3 - alignment):", fitnesses_3_alignment[best_chromosome_index_3_alignment])
        end

        =#


    end
end
=#

#=
"Like generate_graphs_old, but uses validation set accuracy as the metric to compare classifier quality by."
function generate_graphs(seed=22)

    # 1. loop through datasets (for adhoc dataset, there is no validation set, so draw alternatives using training or testing accuracy?)
    # 2. load results for each approach. filter out equivalent and empty circuits. train parameters for all approaches but approach 1. make the trained model instances, ready for classification, with the final parameter values
    # 3.1 calculate validation set accuracy for all approaches using final parameter values (for adhoc, use testing set accuracy to verify the overfitting hypothesis)
    # 3.2 make arrays of corresponding sizes for all approaches
    # 4. draw pareto front with all approaches on it in different colours, using validation data accuracy (for adhoc, use test set accuracy)
    # 5.1 find best individuals by validation accuracy for each approach (for adhoc, use test set accuracy to compare)
    # 5.2 draw best individuals by validation accuracy for each approach (for adhoc, use test set accuracy to compare)
    # 6. draw roc curve with all approaches on in different colours, using validation data accuracy (for adhoc, use test set accuracy)
    # 7. draw a separate confusion matrix for each best classifier (for adhoc, use test set to compare)
    # 8. draw 2 separate decision boundaries for each best classifier, showing training set and testing set on one, and validation set on the other.
    # 9. draw a bar graph of accuracy on each dataset subset (train, test, validation) for each approach (requires calculating train set and test set accuracies)

    classical_colour = "cyan"
    colour_0 = "brown"
    colour_1 = "purple"
    colour_2_1 = "red"
    colour_2_2 = "orange"
    colour_3_1 = "green"
    colour_3_2 = "blue"
    colour_4_1 = "yellow"
    colour_4_2 = "grey"

    #1: go through each dataset
    for dataset_name in [
                        "moons",
                        "cancer",
                        "iris",
                        "digits",
                        #"blobs",
                        "circles",
                        "adhoc",
                        "voice",
                        "susy",
                        "susy_hard"
                        ]
        println("\nProcessing $dataset_name dataset")
        dataset::Dataset = dataset_map[dataset_name]
        dataset_printing_name = dataset_name
        if dataset_printing_name == "susy"
            dataset_printing_name = "SUSY"
        elseif dataset_printing_name == "susy_hard"
            dataset_printing_name = "SUSY reduced features"
        elseif dataset_printing_name == "adhoc"
            dataset_printing_name = "Random"
        else
            dataset_printing_name = uppercasefirst(dataset_name)
        end
        
        #2: load data, create trained models with final parameters
        println("Loading result data")
        # load results for approach 1, 3.1, and 3.2
        #population_0, fitnesses_0, history_0 = load_results(dataset_name, "dynamic_dataset_size")
        population_1, fitnesses_1, history_1 = load_results(dataset_name, "accuracy")
        #population_3_1, fitnesses_3_1, history_3_1 = load_results(dataset_name, "rmse_parameter_training")
        #population_3_2, fitnesses_3_2, history_3_2 = load_results(dataset_name, "alignment_parameter_training")
        population_4_1, fitnesses_4_1, history_4_1 = load_results(dataset_name, "alignment")
        population_4_2, fitnesses_4_2, history_4_2 = load_results(dataset_name, "alignment_approximation")
        
        println("Animating genetic fitness histories")
        #animate_genetic_fitness_history(history_0; dataset_name=dataset_name, training_type="dynamic_dataset_size")
        animate_genetic_fitness_history(history_1; dataset_name=dataset_name, training_type=nothing)
        #animate_genetic_fitness_history(history_3_1; dataset_name=dataset_name, training_type="rmse")
        #animate_genetic_fitness_history(history_3_2; dataset_name=dataset_name, training_type="kernel_target_alignment")
        animate_genetic_fitness_history(history_4_1; ylabel="Kernel-target alignment", dataset_name=dataset_name, training_type="nothing_alignment", ylim=(0, 1)) #unlike accuracy, alignment can commonly be below 0.5
        animate_genetic_fitness_history(history_4_2; ylabel="Kernel-target alignment approximation", dataset_name=dataset_name, training_type="nothing_alignment_approximation", ylim=(0, 1)) #unlike accuracy, alignment can commonly be below 0.5

        # pre-processing: remove empty circuits and circuits with the exact same size metric and accuracy (since they are different binary representations of the same circuit)
        function remove_duplicates(population, fitnesses)
            population_result = []
            fitnesses_result = []
            # for each index
            for i in 1:length(population)
                acc_1, size_1 = fitnesses[i]
                # if circuit is empty (trivial solution)
                if size_1 == 0
                    @goto next_i_iteration
                end
                # check if any elements after are the same
                for j in (i+1):length(population)
                    # if so, don't include this item
                    acc_2, size_2 = fitnesses[j]
                    if (acc_1 == acc_2) && (size_1 == size_2)
                        @goto next_i_iteration
                    end
                end
                # if no duplicates were detected, include this index
                push!(population_result, population[i])
                push!(fitnesses_result, fitnesses[i])
                @label next_i_iteration
            end
            return population_result, fitnesses_result
        end
        "Like remove_duplicates but also checks that initial parameters are the same."
        function remove_duplicates_approach_3(population, fitnesses)
            population_result = []
            fitnesses_result = []
            # for each index
            for i in 1:length(population)
                acc_1, size_1 = fitnesses[i]
                # if circuit is empty (trivial solution)
                if size_1 == 0
                    @goto next_i_iteration
                end
                # otherwise prepare parameters for equivalence checks
                kernel_1, params_1 = decode_chromosome_parameterised_yao(population[i], dataset.feature_count, 6, 6)
                # check if any elements after are the same
                for j in (i+1):length(population)
                    # if so, don't include this item
                    acc_2, size_2 = fitnesses[j]
                    # if accuracies are the same and sizes are the same
                    if (acc_1 == acc_2) && (size_1 == size_2)
                        # check if initial parameters are the same
                        kernel_2, params_2 = decode_chromosome_parameterised_yao(population[j], dataset.feature_count, 6, 6)
                        if params_1 == params_2
                            @goto next_i_iteration
                        end
                    end
                end
                # if no duplicates were detected, include this index
                push!(population_result, population[i])
                push!(fitnesses_result, fitnesses[i])
                @label next_i_iteration
            end
            return population_result, fitnesses_result
        end
        #println("Removing duplicate solutions")
        #population_0, fitnesses_0 = remove_duplicates(population_0, fitnesses_0)
        #println("Unique approach 0 solutions: $(length(population_0))")
        #population_1, fitnesses_1 = remove_duplicates(population_1, fitnesses_1)
        #println("Unique approach 1 solutions: $(length(population_1))")
        #population_3_1, fitnesses_3_1 = remove_duplicates_approach_3(population_3_1, fitnesses_3_1)
        #println("Unique approach 3.1 solutions: $(length(population_3_1))")
        #population_3_2, fitnesses_3_2 = remove_duplicates_approach_3(population_3_2, fitnesses_3_2) println("Unique approach 3.2 solutions: $(length(population_3_2))")
        #population_4_1, fitnesses_4_1 = remove_duplicates(population_4_1, fitnesses_4_1) println("Unique approach 4.1 solutions: $(length(population_4_1))")
        #population_4_2, fitnesses_4_2 = remove_duplicates(population_4_2, fitnesses_4_2) println("Unique approach 4.2 solutions: $(length(population_4_2))")
        

        # re-split the training data to get the train and test subsets
        train_percent = 0.7     # fraction of data to use for training, check nsga2.jl for the correct value to use (maybe change for flexibility or to read a variable instead of using a constant literal)
        problem_data = py"train_test_split"(dataset.training_samples,
                                            dataset.training_labels,
                                            train_size=train_percent,
                                            random_state=seed,
                                            shuffle=true)
        
        println("Creating final models")
        # find trained parameters for approaches 2.1, 2.2, 3.1, and 3.2
        println("Training 2.1 and 2.2 parameters")
        parameters_2_1, parameter_history_2_1 = population_parameterised_training(population_1, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=seed, metric_type="rmse")
        parameters_2_2, parameter_history_2_2 = population_parameterised_training(population_1, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=seed, metric_type="target_alignment")
        #println("Training 3.1 parameters")
        #parameters_3_1, parameter_history_3_1 = population_parameterised_training(population_3_1, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=seed, metric_type="rmse")
        #println("Training 3.2 parameters")
        #parameters_3_2, parameter_history_3_2 = population_parameterised_training(population_3_2, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=seed, metric_type="target_alignment")

        # create kernels of all classifiers
        #kernels_0 = [decode_chromosome_yao(c, dataset.feature_count, 6, 6) for c in population_0]
        kernels_1 = [decode_chromosome_yao(c, dataset.feature_count, 6, 6) for c in population_1]
        kernels_2_1 = [decode_chromosome_parameterised_yao(c, dataset.feature_count, 6, 6)[1](params) for (c, params) in zip(population_1, parameters_2_1)]
        kernels_2_2 = [decode_chromosome_parameterised_yao(c, dataset.feature_count, 6, 6)[1](params) for (c, params) in zip(population_1, parameters_2_2)]
        #kernels_3_1 = [decode_chromosome_parameterised_yao(c, dataset.feature_count, 6, 6)[1](params) for (c, params) in zip(population_3_1, parameters_3_1)]
        #kernels_3_2 = [decode_chromosome_parameterised_yao(c, dataset.feature_count, 6, 6)[1](params) for (c, params) in zip(population_3_2, parameters_3_2)]
        kernels_4_1 = [decode_chromosome_yao(c, dataset.feature_count, 6, 6) for c in population_4_1]
        kernels_4_2 = [decode_chromosome_yao(c, dataset.feature_count, 6, 6) for c in population_4_2]

        # create final trained models of all members of each population
        #models_0 = [train_model(problem_data, k, seed) for k in kernels_0]
        models_1 = [train_model(problem_data, k, seed) for k in kernels_1]
        models_2_1 = [train_model(problem_data, k, seed) for k in kernels_2_1]
        models_2_2 = [train_model(problem_data, k, seed) for k in kernels_2_2]
        #models_3_1 = [train_model(problem_data, k, seed) for k in kernels_3_1]
        #models_3_2 = [train_model(problem_data, k, seed) for k in kernels_3_2]
        models_4_1 = [train_model(problem_data, k, seed) for k in kernels_4_1]
        models_4_2 = [train_model(problem_data, k, seed) for k in kernels_4_2]

        
        # split problem_data
        train_samples, test_samples, train_labels, test_labels = problem_data

        mean(xs) = sum(xs)/length(xs)

        #3.1: get model accuracies on validation data
        println("Calculating training accuracies")
        #train_accuracies_0 = [accuracy(m, dataset.train_samples, dataset.train_labels) for m in models_0]
        #println("Average train accuracy 0: ", mean(train_accuracies_0))
        train_accuracies_1 = [accuracy(m, train_samples, train_labels) for m in models_1]
        println("Average train accuracy 1: ", mean(train_accuracies_1))
        train_accuracies_2_1 = [accuracy(m, train_samples, train_labels) for m in models_2_1]
        println("Average train accuracy 2.1: ", mean(train_accuracies_2_1))
        train_accuracies_2_2 = [accuracy(m, train_samples, train_labels) for m in models_2_2]
        println("Average train accuracy 2.2: ", mean(train_accuracies_2_2))
        #train_accuracies_3_1 = [accuracy(m, train_samples, train_labels) for m in models_3_1]
        #println("Average train accuracy 3.1: ", mean(train_accuracies_3_1))
        #train_accuracies_3_2 = [accuracy(m, train_samples, train_labels) for m in models_3_2]
        #println("Average train accuracy 3.2: ", mean(train_accuracies_3_2))
        train_accuracies_4_1 = [accuracy(m, train_samples, train_labels) for m in models_4_1]
        println("Average train accuracy 4.1: ", mean(train_accuracies_4_1))
        train_accuracies_4_2 = [accuracy(m, train_samples, train_labels) for m in models_4_2]
        println("Average train accuracy 4.2: ", mean(train_accuracies_4_2))
        println("Calculating testing accuracies")
        #test_accuracies_0 = [accuracy(m, test_samples, test_labels) for m in models_0]
        #println("Average test accuracy 0: ", mean(test_accuracies_0))
        test_accuracies_1 = [accuracy(m, test_samples, test_labels) for m in models_1]
        println("Average test accuracy 1: ", mean(test_accuracies_1))
        test_accuracies_2_1 = [accuracy(m, test_samples, test_labels) for m in models_2_1]
        println("Average test accuracy 2.1: ", mean(test_accuracies_2_1))
        test_accuracies_2_2 = [accuracy(m, test_samples, test_labels) for m in models_2_2]
        println("Average test accuracy 2.2: ", mean(test_accuracies_2_2))
        #test_accuracies_3_1 = [accuracy(m, test_samples, test_labels) for m in models_3_1]
        #println("Average test accuracy 3.1: ", mean(test_accuracies_3_1))
        #test_accuracies_3_2 = [accuracy(m, test_samples, test_labels) for m in models_3_2]
        #println("Average test accuracy 3.2: ", mean(test_accuracies_3_2))
        test_accuracies_4_1 = [accuracy(m, test_samples, test_labels) for m in models_4_1]
        println("Average test accuracy 4.1: ", mean(test_accuracies_4_1))
        test_accuracies_4_2 = [accuracy(m, test_samples, test_labels) for m in models_4_2]
        println("Average test accuracy 4.2: ", mean(test_accuracies_4_2))
        println("Calculating validation accuracies")
        #validation_accuracies_0 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_0]
        #println("Average validation accuracy 0: ", mean(validation_accuracies_0))
        validation_accuracies_1 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_1]
        println("Average validation accuracy 1: ", mean(validation_accuracies_1))
        validation_accuracies_2_1 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_2_1]
        println("Average validation accuracy 2.1: ", mean(validation_accuracies_2_1))
        validation_accuracies_2_2 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_2_2]
        println("Average validation accuracy 2.2: ", mean(validation_accuracies_2_2))
        #validation_accuracies_3_1 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_3_1]
        #println("Average validation accuracy 3.1: ", mean(validation_accuracies_3_1))
        #validation_accuracies_3_2 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_3_2]
        #println("Average validation accuracy 3.2: ", mean(validation_accuracies_3_2))
        validation_accuracies_4_1 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_4_1]
        println("Average validation accuracy 4.1: ", mean(validation_accuracies_4_1))
        validation_accuracies_4_2 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_4_2]
        println("Average validation accuracy 4.2: ", mean(validation_accuracies_4_2))

        #3.2: get model size metrics
        unweighted_size(c) = size_metric(c, 6)
        population_sizes(p) = unweighted_size.(p)
        #sizes_0 = population_sizes(population_0)#[fitnesses_0[i][2] for i in 1:length(fitnesses_0)]
        sizes_1 = population_sizes(population_1)#[fitnesses_1[i][2] for i in 1:length(fitnesses_1)]
        sizes_2_1 = sizes_1
        sizes_2_2 = sizes_1
        #sizes_3_1 = population_sizes(population_3_1)#[fitnesses_3_1[i][2] for i in 1:length(fitnesses_3_1)]
        #sizes_3_2 = population_sizes(population_3_2)#[fitnesses_3_2[i][2] for i in 1:length(fitnesses_3_2)]
        sizes_4_1 = population_sizes(population_4_1)
        sizes_4_2 = population_sizes(population_4_2)

        #4: draw pareto front with all approaches listed
        println("Drawing population metrics")
        function add_pareto_plot(accuracies, sizes, colour, label)
            Plots.plot!(sizes, accuracies,
                        seriestype=:scatter,
                        colour=colour,
                        seriesalpha=0.4,
                        label=label)
        end
        # make new empty plot
        fig = Plots.plot(title="$(dataset_printing_name) metrics for each approach's final population", legend=true,
                        xlabel="Size metric", ylabel="Validation set accuracy",
                        xlim=(0, 6), ylim=(0.35, 1.4),
                        labels=[
                            #"0"
                            "1" "2.1" "2.2" "4.1" "4.2"
                            #"3.1"
                            #"3.2"
                            #"4.1"
                            ])
        #add_pareto_plot(validation_accuracies_0, sizes_0, colour_0, "0")
        add_pareto_plot(validation_accuracies_1, sizes_1, colour_1, "1")
        add_pareto_plot(validation_accuracies_2_1, sizes_2_1, colour_2_1, "2.1")
        add_pareto_plot(validation_accuracies_2_2, sizes_2_2, colour_2_2, "2.2")
        #add_pareto_plot(validation_accuracies_3_1, sizes_3_1, colour_3_1, "3.1")
        #add_pareto_plot(validation_accuracies_3_2, sizes_3_2, colour_3_2, "3.2")
        add_pareto_plot(validation_accuracies_4_1, sizes_4_1, colour_4_1, "4.1")
        add_pareto_plot(validation_accuracies_4_2, sizes_4_2, colour_4_2, "4.2")
        savefig(fig, "./diagrams/$(dataset_name)_validation_population_metrics.pdf")

        # make new empty plot
        fig = Plots.plot(title="$(dataset_printing_name) metrics for each approach's final population", legend=true,
                        xlabel="Size metric", ylabel="Test set accuracy",
                        xlim=(0, 6), ylim=(0.35, 1.4),
                        labels=[
                            #"0"
                            "1" "2.1" "2.2" "4.1" "4.2"
                            #"3.1"
                            #"3.2"
                            #"4.1"
                            ])
        #add_pareto_plot(test_accuracies_0, sizes_0, colour_0, "0")
        add_pareto_plot(test_accuracies_1, sizes_1, colour_1, "1")
        add_pareto_plot(test_accuracies_2_1, sizes_2_1, colour_2_1, "2.1")
        add_pareto_plot(test_accuracies_2_2, sizes_2_2, colour_2_2, "2.2")
        #add_pareto_plot(test_accuracies_3_1, sizes_3_1, colour_3_1, "3.1")
        #add_pareto_plot(test_accuracies_3_2, sizes_3_2, colour_3_2, "3.2")
        add_pareto_plot(test_accuracies_4_1, sizes_4_1, colour_4_1, "4.1")
        add_pareto_plot(test_accuracies_4_2, sizes_4_2, colour_4_2, "4.2")
        savefig(fig, "./diagrams/$(dataset_name)_test_population_metrics.pdf")

        #5.1: find best individuals by validation set accuracy
        function best_individual_index(accuracies, sizes)
            # get the highest-accuracy individual
            highest_accuracy = accuracies[1]
            smallest_size = sizes[1]
            result_index = 1
            for i in 2:length(sizes)
                next_accuracy = accuracies[i]
                next_size = sizes[i]
                # Conditional check passes if accuracy is better or if accuracy is the same but size is better
                if next_accuracy > highest_accuracy || (next_accuracy == highest_accuracy && next_size < smallest_size)
                    highest_accuracy = next_accuracy
                    smallest_size = next_size
                    result_index = i
                end
            end
            return result_index
        end
        #best_index_0 = best_individual_index(test_accuracies_0, sizes_0)
        best_index_1 = best_individual_index(test_accuracies_1, sizes_1)
        best_index_2_1 = best_individual_index(test_accuracies_2_1, sizes_2_1)
        best_index_2_2 = best_individual_index(test_accuracies_2_2, sizes_2_2)
        #best_index_3_1 = best_individual_index(test_accuracies_3_1, sizes_3_1)
        #best_index_3_2 = best_individual_index(test_accuracies_3_2, sizes_3_2)
        best_index_4_1 = best_individual_index(test_accuracies_4_1, sizes_4_1)
        best_index_4_2 = best_individual_index(test_accuracies_4_2, sizes_4_2)

        # calculate accuracies of best individuals
        test_set_accuracy(model) = accuracy(model, test_samples, test_labels)
        train_set_accuracy(model) = accuracy(model, train_samples, train_labels)
        
        groups = ["Training", "Testing", "Validation"]
        models = [
            #models_0[best_index_0]
            models_1[best_index_1]
            models_2_1[best_index_2_1]
            models_2_2[best_index_2_2]
            #models_3_1[best_index_3_1]
            #models_3_2[best_index_3_2]
            models_4_1[best_index_4_1]
            models_4_2[best_index_4_2]
         ]
        train_set_accuracies = train_set_accuracy.(models)
        test_set_accuracies = test_set_accuracy.(models)
        validation_accuracies = [
                                 #validation_accuracies_0[best_index_0]
                                 validation_accuracies_1[best_index_1]
                                 validation_accuracies_2_1[best_index_2_1]
                                 validation_accuracies_2_2[best_index_2_2]
                                 #validation_accuracies_3_1[best_index_3_1]
                                 #validation_accuracies_3_2[best_index_3_2]
                                 validation_accuracies_4_1[best_index_4_1]
                                 validation_accuracies_4_2[best_index_4_2]
                                 ]

        improvements_2_1 = validation_accuracies_2_1 .- validation_accuracies_1
        improvements_2_2 = validation_accuracies_2_2 .- validation_accuracies_1
        println("Average validation improvement over approach 1 (approach 2.1):", sum(improvements_2_1) / length(improvements_2_1))
        println("Average validation improvement over approach 1 (approach 2.2):", sum(improvements_2_2) / length(improvements_2_2))

        best_solution_size_metrics = [
                                      #sizes_0[best_index_0],
                                      sizes_1[best_index_1],
                                      sizes_2_1[best_index_2_1],
                                      sizes_2_2[best_index_2_2],
                                      #sizes_3_1[best_index_3_1],
                                      #sizes_3_2[best_index_3_2],
                                      sizes_4_1[best_index_4_1],
                                      sizes_4_2[best_index_4_2]
                                      ]
        # print details on best models for document table
        println("Information about models that achieved the highest validation accuracy (train, test, validation, size metric):")
        data_string(approach_number, approach_index) = "$dataset_printing_name approach $approach_number: ($(train_set_accuracies[approach_index]), $(test_set_accuracies[approach_index]), $(validation_accuracies[approach_index]), $(best_solution_size_metrics[approach_index]))"
        for (i, anr) in enumerate([
            #0,
            1, 2.1, 2.2,
            #3.1,
            #3.2
            4.1,
            4.2
            ])
            println(data_string(anr, i))
        end

        #5.2: draw and save best circuits for each approach
        println("Drawing best circuits")
        #save_text(draw_chromosome_latex(population_0[best_index_0], dataset.feature_count, 6, 6),
                    #"./diagrams/$(dataset_name)_best_circuit_0.tex")
        save_text(draw_chromosome_latex(population_1[best_index_1], dataset.feature_count, 6, 6),
                    "./diagrams/$(dataset_name)_best_circuit_1.tex")
        save_text(draw_chromosome_latex(population_1[best_index_2_1], dataset.feature_count, 6, 6; parameters=parameters_2_1[best_index_2_1]),
                    "./diagrams/$(dataset_name)_best_circuit_2_1.tex")
        save_text(draw_chromosome_latex(population_1[best_index_2_2], dataset.feature_count, 6, 6; parameters=parameters_2_2[best_index_2_2]),
                    "./diagrams/$(dataset_name)_best_circuit_2_2.tex")
        #save_text(draw_chromosome_latex(population_3_1[best_index_3_1], dataset.feature_count, 6, 6; parameters=parameters_3_1[best_index_3_1]),
                    #"./diagrams/$(dataset_name)_best_circuit_3_1.tex")
        #save_text(draw_chromosome_latex(population_3_2[best_index_3_2], dataset.feature_count, 6, 6; parameters=parameters_3_2[best_index_3_2]),
        #            "./diagrams/$(dataset_name)_best_circuit_3_2.tex")
        save_text(draw_chromosome_latex(population_4_1[best_index_4_1], dataset.feature_count, 6, 6),
                    "./diagrams/$(dataset_name)_best_circuit_4_1.tex")
        save_text(draw_chromosome_latex(population_4_2[best_index_4_2], dataset.feature_count, 6, 6),
                    "./diagrams/$(dataset_name)_best_circuit_4_2.tex")
        
        #6: draw roc curves of the approaches on the same graph
        println("Drawing ROC curves")
        function add_roc_curve(model, colour, label)
            kernel_outputs = compute_kernel_matrix(model.kernel, model.training_set, dataset.validation_samples)
            df_outputs = decision_function(model.classifier, kernel_outputs)
            true_positive_rates, false_positive_rates = [], []
            width=0.001
            for boundary in -3:width:3
                # label samples using the current boundary
                predictions = [o < boundary ? -1 : 1 for o in df_outputs]
                # count true and false positives
                true_positive_p = ((true_label, predicted_label),) -> predicted_label == 1 && true_label == 1
                false_positive_p = ((true_label, predicted_label),) -> predicted_label == 1 && true_label == -1
                tp = count(true_positive_p, zip(dataset.validation_labels, predictions))
                fp = count(false_positive_p, zip(dataset.validation_labels, predictions))
                tpr = tp / dataset.num_positive_validation_instances
                fpr = fp / dataset.num_negative_validation_instances
                push!(true_positive_rates, tpr)
                push!(false_positive_rates, fpr)
            end
            Plots.plot!(false_positive_rates, true_positive_rates, colour=colour, label=label)
        end
        # make axis for roc curve plots
        fig = Plots.plot(title="$(dataset_printing_name) ROC curves for the best produced models", legend=true,
                        xlabel="False positive rate", ylabel="True positive rate",
                        xlim=(0, 1), ylim=(0, 1),
                        labels=[
                            #"0"
                            "1" "2.1" "2.2" "4.1" "4.2"
                            #"3.1"
                            #"3.2"
                            #"4.1"
                            ])
        #add_roc_curve(models_0[best_index_0], colour_0, "0")
        add_roc_curve(models_1[best_index_1], colour_1, "1")
        add_roc_curve(models_2_1[best_index_2_1], colour_2_1, "2.1")
        add_roc_curve(models_2_2[best_index_2_2], colour_2_2, "2.2")
        #add_roc_curve(models_3_1[best_index_3_1], colour_3_1, "3.1")
        #add_roc_curve(models_3_2[best_index_3_2], colour_3_2, "3.2")
        add_roc_curve(models_4_1[best_index_4_1], colour_4_1, "4.1")
        add_roc_curve(models_4_2[best_index_4_2], colour_4_2, "4.2")
        savefig(fig, "./diagrams/$(dataset_name)_roc_curves.pdf")

        #7: draw confusion matrices for the best individuals
        println("Drawing confusion matrices")
        #confusion_matrix(models_0[best_index_0], dataset.validation_samples, dataset.validation_labels)
        #py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_0.pdf")
        confusion_matrix(models_1[best_index_1], dataset.validation_samples, dataset.validation_labels)
        py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_1.pdf")
        confusion_matrix(models_2_1[best_index_2_1], dataset.validation_samples, dataset.validation_labels)
        py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_2_1.pdf")
        confusion_matrix(models_2_2[best_index_2_2], dataset.validation_samples, dataset.validation_labels)
        py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_2_2.pdf")
        #confusion_matrix(models_3_1[best_index_3_1], dataset.validation_samples, dataset.validation_labels)
        #py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_3_1.pdf")
        #confusion_matrix(models_3_2[best_index_3_2], dataset.validation_samples, dataset.validation_labels)
        #py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_3_2.pdf")
        confusion_matrix(models_4_1[best_index_4_1], dataset.validation_samples, dataset.validation_labels)
        py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_4_1.pdf")
        confusion_matrix(models_4_2[best_index_4_2], dataset.validation_samples, dataset.validation_labels)
        py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_4_2.pdf")

        #9: graph train, test and validation accuracies for each approach
        println("Plotting classification accuracies graph")
        
        classical_model = SVC(kernel="rbf", class_weight="balanced")
        fit!(classical_model, train_samples, train_labels) #use same training data as genetically produced QSVM kernels
        classical_accuracies = [score(classical_model,  train_samples, train_labels) score(classical_model, test_samples, test_labels) score(classical_model, dataset.validation_samples, dataset.validation_labels)]
        println("Classical accuracies: ", classical_accuracies)

        accuracies(i) = [train_set_accuracies[i] test_set_accuracies[i] validation_accuracies[i]]
        #y00s = accuracies(1)
        y0s = accuracies(1)
        y1s = accuracies(2)
        y2s = accuracies(3)
        #y3s = accuracies(5)
        #y4s = accuracies(5)
        y5s = accuracies(4)
        y6s = accuracies(5)
        ys = [
            classical_accuracies
              #y00s;
              y0s;
              y1s;
              y2s;
              #y3s;
              #y4s;
              y5s
              y6s
             ]'
        fig = Plots.plot(title="$(dataset_printing_name) accuracies for each approach and data subset",
                         legend=true,
                         ylim=(0, 1.4))
        groupedbar!(groups, ys, labels=[
            #"0"
            "SVM RBF kernel" "1" "2.1" "2.2" "4.1" "4.2"
             #"3.1"
             #"3.2"
             #"4.1"
             ], color=[
                #colour_0
                classical_colour colour_1 colour_2_1 colour_2_2 colour_4_1 colour_4_2],# colour_3_1 colour_3_2],
                ylabel="Accuracy on subset")
        """
        xs = ["1"
              "2.1"
              "2.2"
              "3.1"
              "3.2"
              ]
        models = [
                    models_1[best_index_1]
                    models_2_1[best_index_2_1]
                    models_2_2[best_index_2_2]
                    models_3_1[best_index_3_1]
                    models_3_2[best_index_3_2]
                 ]
        y0s = train_set_accuracy.(models)
        y1s = test_set_accuracy.(models)
        y2s = [
                validation_accuracies_1[best_index_1]
                validation_accuracies_2_1[best_index_2_1]
                validation_accuracies_2_2[best_index_2_2]
                validation_accuracies_3_1[best_index_3_1]
                validation_accuracies_3_2[best_index_3_2]
              ]
        # create plot
        fig = Plots.plot(title="$(dataset_printing_name) accuracies for each approach", legend=true, ylim=(0, 1.2)) # set ylim past 100 to make space for the legend
        #annotation_text(x) = Plots.text(x, pointsize=1)
        ys = [y0s y1s y2s]
        groupedbar!(xs, ys, labels=["Train" "Test" "Validation"])#, series_annotations=annotation_text.([string.(y0s) string.(y1s) string.(y2s)]))
        """
        """
        # add train, test and validation accuracies
        #Plots.bar!(xs, y0s, colour="green", label="Train")
        #Plots.bar!(xs, y1s, colour="blue", label="Test")
        #Plots.bar!(xs, y2s, colour="orange", label="Validation")
        """
        savefig(fig, "./diagrams/$(dataset_name)_accuracies.pdf")

        #10: parameter training history (for convergence analysis)
        println("Plotting parameter training convergence")
        function plot_parameter_history(y_axis_label, approach_number, history)
            fig = Plots.plot(1:100, history, xlabel="$y_axis_label evaluations", ylabel=y_axis_label, title="$dataset_printing_name, approach $approach_number parameter training", legend=false)
            savefig(fig, "./diagrams/$(dataset_name)_optimization_history_$(approach_number).pdf")
        end
        best_history_2_1 = parameter_history_2_1[best_index_2_1]
        plot_parameter_history("RMSE", 2.1, best_history_2_1)
        best_history_2_2 = parameter_history_2_2[best_index_2_2]
        plot_parameter_history("Kernel-target alignment", 2.2, best_history_2_2)
        #best_history_3_1 = parameter_history_3_1[best_index_3_1]
        #plot_parameter_history("RMSE", 3.1, best_history_3_1)
        #best_history_3_2 = parameter_history_3_2[best_index_3_2]
        #plot_parameter_history("Kernel-target alignment", 3.2, best_history_3_2)
       
        #8: draw decision boundaries for the best individuals
        if dataset.feature_count == 2
            println("Drawing decision boundaries")
            # compute classification outputs for the contour graph
            axis_interval = -1.2:.01:1.2
            point_to_index(x) = trunc(Int64, (x+1.2)*100)+1
            grid_samples = []
            for x in axis_interval
                for y in axis_interval
                    push!(grid_samples, [x,y])
                end
            end

            "Creates 2 decision boundaries: 1 for training and testing set points, and 1 for validation set points"
            function graph_decision_boundaries(model, approach_number)
                # calculate contour coordinate outputs
                kernel_output_matrix = compute_kernel_matrix(model.kernel, model.training_set, grid_samples)
                outputs = predict(model.classifier, kernel_output_matrix)
                # create contour output matrix and fill in the values
                z_matrix = Matrix(undef, length(axis_interval), length(axis_interval))
                index = 1
                for x in 1:length(axis_interval)
                    for y in 1:length(axis_interval)
                        output = outputs[index]
                        z_matrix[x, y] = (output == -1 ? 1 : 2)
                        index += 1
                    end
                end
                ccol = cgrad([RGB(1,.2,.2), RGB(.2,.2,1)])
                # create first decision boundary plot.
                fig = Plots.plot(leg=:none, title="$(dataset_printing_name) decision boundary, approach $approach_number",
                                lims=(-1.2, 1.2), xlabel="Feature 1", ylabel="Feature 2")
                # draw contour
                contour!(axis_interval, axis_interval, (x,y)->z_matrix[point_to_index(x),point_to_index(y)],
                        f=true, nlev=2, c=ccol)
                # draw training set points
                scatter!([s[1] for s in train_samples],
                        [s[2] for s in train_samples],
                        m=[:rect :circle],
                        color=[label == -1 ? "red" : "blue" for label in train_labels])
                # draw test set points
                scatter!([s[1] for s in test_samples],
                        [s[2] for s in test_samples],
                        m=[:rect :circle],
                        color=[label == -1 ? "pink" : "cyan" for label in test_labels])
                # save to disk
                savefig(fig, "./diagrams/$(dataset_name)_decision_boundary_$(approach_number).pdf")

                # second, plot decision boundary for validation points
                fig2 = Plots.plot(leg=:none, title="$(dataset_printing_name) decision boundary, approach $approach_number, validation data",
                                lims=(-1.2, 1.2), xlabel="Feature 1", ylabel="Feature 2")
                # draw contour again
                contour!(axis_interval, axis_interval, (x,y)->z_matrix[point_to_index(x),point_to_index(y)],
                        f=true, nlev=2, c=ccol)
                # draw validation set points
                scatter!([s[1] for s in dataset.validation_samples],
                        [s[2] for s in dataset.validation_samples],
                        m=[:rect :circle],
                        color=[label == -1 ? "red" : "blue" for label in dataset.validation_labels])
                # save to disk
                savefig(fig2, "./diagrams/$(dataset_name)_decision_boundary_$(approach_number)_validation.pdf")
            end

            #graph_decision_boundaries(models_0[best_index_0], 0)
            graph_decision_boundaries(models_1[best_index_1], 1)
            graph_decision_boundaries(models_2_1[best_index_2_1], 2.1)
            graph_decision_boundaries(models_2_2[best_index_2_2], 2.2)
            #graph_decision_boundaries(models_3_1[best_index_3_1], 3.1)
            #graph_decision_boundaries(models_3_2[best_index_3_2], 3.2)
            graph_decision_boundaries(models_4_1[best_index_4_1], 4.1)
            graph_decision_boundaries(models_4_2[best_index_4_2], 4.2)
        end
    end
end
=#


"Like generate_graphs, but uses new naming scheme for approaches and adds new parameter training approaches (1, 1.1, 1.2, 2, 2.1, 2.2, 3, 3.1, 3.2)."
function generate_graphs_new_naming(seed=22)

    # 1. loop through datasets (for adhoc dataset, there is no validation set, so draw alternatives using training or testing accuracy?)
    # 2. load results for each approach. filter out equivalent and empty circuits. train parameters for all approaches but approach 1. make the trained model instances, ready for classification, with the final parameter values
    # 3.1 calculate validation set accuracy for all approaches using final parameter values (for adhoc, use testing set accuracy to verify the overfitting hypothesis)
    # 3.2 make arrays of corresponding sizes for all approaches
    # 4. draw pareto front with all approaches on it in different colours, using validation data accuracy (for adhoc, use test set accuracy)
    # 5.1 find best individuals by validation accuracy for each approach (for adhoc, use test set accuracy to compare)
    # 5.2 draw best individuals by validation accuracy for each approach (for adhoc, use test set accuracy to compare)
    # 6. draw roc curve with all approaches on in different colours, using validation data accuracy (for adhoc, use test set accuracy)
    # 7. draw a separate confusion matrix for each best classifier (for adhoc, use test set to compare)
    # 8. draw 2 separate decision boundaries for each best classifier, showing training set and testing set on one, and validation set on the other.
    # 9. draw a bar graph of accuracy on each dataset subset (train, test, validation) for each approach (requires calculating train set and test set accuracies)

    classical_colour = "brown"
    colour_1 = "green"
    colour_1_1 = "blue"
    colour_1_2 = "aqua"
    colour_2 = "grey25"
    colour_2_1 = "grey50"
    colour_2_2 = "grey75"
    colour_3 = "red"
    colour_3_1 = "orange"
    colour_3_2 = "yellow"

    #1: go through each dataset
    for dataset_name in [
                        "moons",
                        "cancer",
                        "iris",
                        "digits",
                        #"blobs",
                        "circles",
                        "adhoc",
                        "voice",
                        "susy",
                        "susy_hard"
                        ]
        println("\nProcessing $dataset_name dataset")
        dataset::Dataset = dataset_map[dataset_name]
        dataset_printing_name = dataset_name
        if dataset_printing_name == "susy"
            dataset_printing_name = "SUSY"
        elseif dataset_printing_name == "susy_hard"
            dataset_printing_name = "SUSY reduced features"
        elseif dataset_printing_name == "adhoc"
            dataset_printing_name = "Random"
        else
            dataset_printing_name = uppercasefirst(dataset_name)
        end
        
        #2: load data, create trained models with final parameters
        println("Loading result data")
        # load results for approach 1, 3.1, and 3.2
        population_1, fitnesses_1, history_1 = load_results(dataset_name, "accuracy")
        population_2, fitnesses_2, history_2 = load_results(dataset_name, "alignment")
        population_3, fitnesses_3, history_3 = load_results(dataset_name, "alignment_approximation")
        
        println("Animating genetic fitness histories")
        animate_genetic_fitness_history(history_1; dataset_name=dataset_name, training_type="accuracy")
        animate_genetic_fitness_history(history_2; ylabel="Kernel-target alignment", dataset_name=dataset_name, training_type="alignment", ylim=(0, 1)) #unlike accuracy, alignment can commonly be below 0.5
        animate_genetic_fitness_history(history_3; ylabel="Kernel-target alignment approximation", dataset_name=dataset_name, training_type="alignment_approximation", ylim=(0, 1)) #unlike accuracy, alignment can commonly be below 0.5

        # pre-processing: remove empty circuits and circuits with the exact same size metric and accuracy (since they are different binary representations of the same circuit)
        function remove_duplicates(population, fitnesses)
            population_result = []
            fitnesses_result = []
            # for each index
            for i in 1:length(population)
                acc_1, size_1 = fitnesses[i]
                # if circuit is empty (trivial solution)
                if size_1 == 0
                    @goto next_i_iteration
                end
                # check if any elements after are the same
                for j in (i+1):length(population)
                    # if so, don't include this item
                    acc_2, size_2 = fitnesses[j]
                    if (acc_1 == acc_2) && (size_1 == size_2)
                        @goto next_i_iteration
                    end
                end
                # if no duplicates were detected, include this index
                push!(population_result, population[i])
                push!(fitnesses_result, fitnesses[i])
                @label next_i_iteration
            end
            return population_result, fitnesses_result
        end

        #=
        println("Removing duplicate solutions")
        population_1, fitnesses_1 = remove_duplicates(population_1, fitnesses_1)
        println("Unique approach 1 solutions: $(length(population_1))")
        population_2, fitnesses_2 = remove_duplicates(population_2, fitnesses_2)
        println("Unique approach 2 solutions: $(length(population_2))")
        population_3, fitnesses_3 = remove_duplicates(population_3, fitnesses_3)
        println("Unique approach 3 solutions: $(length(population_3))")
        =#

        # re-split the training data to get the train and test subsets used in genetic training (if the seed is the same)
        train_percent = 0.7     # fraction of data to use for training, check nsga2.jl for the correct value to use (maybe change for flexibility or to read a variable instead of using a constant literal)
        problem_data = py"train_test_split"(dataset.training_samples,
                                            dataset.training_labels,
                                            train_size=train_percent,
                                            random_state=seed,
                                            shuffle=true)
        
        println("Creating final models")
        # find trained parameters for approaches 1.1, 1.2, 2.1, 2.2, 3.1, and 3.2
        println("Training 1.1 and 1.2 parameters")
        parameters_1_1, parameter_history_1_1 = population_parameterised_training(population_1, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=seed, metric_type="rmse")
        parameters_1_2, parameter_history_1_2 = population_parameterised_training(population_1, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=seed, metric_type="target_alignment")
        println("Training 2.1 and 2.2 parameters")
        parameters_2_1, parameter_history_2_1 = population_parameterised_training(population_2, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=seed, metric_type="rmse")
        parameters_2_2, parameter_history_2_2 = population_parameterised_training(population_2, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=seed, metric_type="target_alignment")
        println("Training 3.1 and 3.2 parameters")
        parameters_3_1, parameter_history_3_1 = population_parameterised_training(population_3, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=seed, metric_type="rmse")
        parameters_3_2, parameter_history_3_2 = population_parameterised_training(population_3, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=100, seed=seed, metric_type="target_alignment")

        # create kernels of all classifiers
        kernels_1 = [decode_chromosome_yao(c, dataset.feature_count, 6, 6) for c in population_1]
        kernels_1_1 = [decode_chromosome_parameterised_yao(c, dataset.feature_count, 6, 6)[1](params) for (c, params) in zip(population_1, parameters_1_1)]
        kernels_1_2 = [decode_chromosome_parameterised_yao(c, dataset.feature_count, 6, 6)[1](params) for (c, params) in zip(population_1, parameters_1_2)]
        kernels_2 = [decode_chromosome_yao(c, dataset.feature_count, 6, 6) for c in population_2]
        kernels_2_1 = [decode_chromosome_parameterised_yao(c, dataset.feature_count, 6, 6)[1](params) for (c, params) in zip(population_2, parameters_2_1)]
        kernels_2_2 = [decode_chromosome_parameterised_yao(c, dataset.feature_count, 6, 6)[1](params) for (c, params) in zip(population_2, parameters_2_2)]
        kernels_3 = [decode_chromosome_yao(c, dataset.feature_count, 6, 6) for c in population_3]
        kernels_3_1 = [decode_chromosome_parameterised_yao(c, dataset.feature_count, 6, 6)[1](params) for (c, params) in zip(population_3, parameters_3_1)]
        kernels_3_2 = [decode_chromosome_parameterised_yao(c, dataset.feature_count, 6, 6)[1](params) for (c, params) in zip(population_3, parameters_3_2)]

        # create final trained models of all members of each population
        models_1 = [train_model(problem_data, k, seed) for k in kernels_1]
        models_1_1 = [train_model(problem_data, k, seed) for k in kernels_1_1]
        models_1_2 = [train_model(problem_data, k, seed) for k in kernels_1_2]
        models_2 = [train_model(problem_data, k, seed) for k in kernels_2]
        models_2_1 = [train_model(problem_data, k, seed) for k in kernels_2_1]
        models_2_2 = [train_model(problem_data, k, seed) for k in kernels_2_2]
        models_3 = [train_model(problem_data, k, seed) for k in kernels_3]
        models_3_1 = [train_model(problem_data, k, seed) for k in kernels_3_1]
        models_3_2 = [train_model(problem_data, k, seed) for k in kernels_3_2]

        
        # split problem_data
        train_samples, test_samples, train_labels, test_labels = problem_data

        mean(xs) = sum(xs)/length(xs)

        #3.1: get model accuracies on validation data
        println("Calculating training accuracies")
        train_accuracies_1 = [accuracy(m, train_samples, train_labels) for m in models_1]
        println("Average train accuracy 1: ", mean(train_accuracies_1))
        train_accuracies_1_1 = [accuracy(m, train_samples, train_labels) for m in models_1_1]
        println("Average train accuracy 1.1: ", mean(train_accuracies_1_1))
        train_accuracies_1_2 = [accuracy(m, train_samples, train_labels) for m in models_1_2]
        println("Average train accuracy 1.2: ", mean(train_accuracies_1_2))
        train_accuracies_2 = [accuracy(m, train_samples, train_labels) for m in models_2]
        println("Average train accuracy 2: ", mean(train_accuracies_2))
        train_accuracies_2_1 = [accuracy(m, train_samples, train_labels) for m in models_2_1]
        println("Average train accuracy 2.1: ", mean(train_accuracies_2_1))
        train_accuracies_2_2 = [accuracy(m, train_samples, train_labels) for m in models_2_2]
        println("Average train accuracy 2.2: ", mean(train_accuracies_2_2))
        train_accuracies_3 = [accuracy(m, train_samples, train_labels) for m in models_3]
        println("Average train accuracy 3: ", mean(train_accuracies_3))
        train_accuracies_3_1 = [accuracy(m, train_samples, train_labels) for m in models_3_1]
        println("Average train accuracy 3.1: ", mean(train_accuracies_3_1))
        train_accuracies_3_2 = [accuracy(m, train_samples, train_labels) for m in models_3_2]
        println("Average train accuracy 3.2: ", mean(train_accuracies_3_2))

        println("Calculating testing accuracies")
        test_accuracies_1 = [accuracy(m, test_samples, test_labels) for m in models_1]
        println("Average test accuracy 1: ", mean(test_accuracies_1))
        test_accuracies_1_1 = [accuracy(m, test_samples, test_labels) for m in models_1_1]
        println("Average test accuracy 1.1: ", mean(test_accuracies_1_1))
        test_accuracies_1_2 = [accuracy(m, test_samples, test_labels) for m in models_1_2]
        println("Average test accuracy 1.2: ", mean(test_accuracies_1_2))
        test_accuracies_2 = [accuracy(m, test_samples, test_labels) for m in models_2]
        println("Average test accuracy 2: ", mean(test_accuracies_2))
        test_accuracies_2_1 = [accuracy(m, test_samples, test_labels) for m in models_2_1]
        println("Average test accuracy 2.1: ", mean(test_accuracies_2_1))
        test_accuracies_2_2 = [accuracy(m, test_samples, test_labels) for m in models_2_2]
        println("Average test accuracy 2.2: ", mean(test_accuracies_2_2))
        test_accuracies_3 = [accuracy(m, test_samples, test_labels) for m in models_3]
        println("Average test accuracy 3: ", mean(test_accuracies_3))
        test_accuracies_3_1 = [accuracy(m, test_samples, test_labels) for m in models_3_1]
        println("Average test accuracy 3.1: ", mean(test_accuracies_3_1))
        test_accuracies_3_2 = [accuracy(m, test_samples, test_labels) for m in models_3_2]
        println("Average test accuracy 3.2: ", mean(test_accuracies_3_2))
        
        println("Calculating validation accuracies")
        validation_accuracies_1 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_1]
        println("Average validation accuracy 1: ", mean(validation_accuracies_1))
        validation_accuracies_1_1 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_1_1]
        println("Average validation accuracy 1.1: ", mean(validation_accuracies_1_1))
        validation_accuracies_1_2 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_1_2]
        println("Average validation accuracy 1.2: ", mean(validation_accuracies_1_2))
        validation_accuracies_2 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_2]
        println("Average validation accuracy 2: ", mean(validation_accuracies_2))
        validation_accuracies_2_1 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_2_1]
        println("Average validation accuracy 2.1: ", mean(validation_accuracies_2_1))
        validation_accuracies_2_2 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_2_2]
        println("Average validation accuracy 2.2: ", mean(validation_accuracies_2_2))
        validation_accuracies_3 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_3]
        println("Average validation accuracy 3: ", mean(validation_accuracies_3))
        validation_accuracies_3_1 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_3_1]
        println("Average validation accuracy 3.1: ", mean(validation_accuracies_3_1))
        validation_accuracies_3_2 = [accuracy(m, dataset.validation_samples, dataset.validation_labels) for m in models_3_2]
        println("Average validation accuracy 3.2: ", mean(validation_accuracies_3_2))

        #3.2: get model size metrics
        unweighted_size(c) = size_metric(c, 6)
        population_sizes(p) = unweighted_size.(p)
        sizes_1 = population_sizes(population_1)
        sizes_1_1 = sizes_1
        sizes_1_2 = sizes_1
        sizes_2 = population_sizes(population_2)
        sizes_2_1 = sizes_2
        sizes_2_2 = sizes_2
        sizes_3 = population_sizes(population_3)
        sizes_3_1 = sizes_3
        sizes_3_2 = sizes_3

        #4: draw pareto front with all approaches listed
        println("Drawing population metrics")
        function add_pareto_plot(accuracies, sizes, colour, label)
            Plots.plot!(sizes, accuracies,
                        seriestype=:scatter,
                        colour=colour,
                        seriesalpha=0.4,
                        label=label)
        end
        # make new empty plot
        fig = Plots.plot(title="$(dataset_printing_name) metrics for each approach's final population", legend=true,
                        xlabel="Size metric", ylabel="Validation set accuracy",
                        xlim=(0, 6), ylim=(0.35, 1.4),
                        labels=["1" "1.1" "1.2" "2" "2.1" "2.2" "3" "3.1" "3.2"])
        add_pareto_plot(validation_accuracies_1, sizes_1, colour_1, "1")
        add_pareto_plot(validation_accuracies_1_1, sizes_1_1, colour_1_1, "1.1")
        add_pareto_plot(validation_accuracies_1_2, sizes_1_2, colour_1_2, "1.2")
        add_pareto_plot(validation_accuracies_2, sizes_2, colour_2, "2")
        add_pareto_plot(validation_accuracies_2_1, sizes_2_1, colour_2_1, "2.1")
        add_pareto_plot(validation_accuracies_2_2, sizes_2_2, colour_2_2, "2.2")
        add_pareto_plot(validation_accuracies_3, sizes_3, colour_3, "3")
        add_pareto_plot(validation_accuracies_3_1, sizes_3_1, colour_3_1, "3.1")
        add_pareto_plot(validation_accuracies_3_2, sizes_3_2, colour_3_2, "3.2")
        
        savefig(fig, "./diagrams/$(dataset_name)_validation_population_metrics.pdf")

        # make new empty plot
        fig = Plots.plot(title="$(dataset_printing_name) metrics for each approach's final population", legend=true,
                        xlabel="Size metric", ylabel="Test set accuracy",
                        xlim=(0, 6), ylim=(0.35, 1.4),
                        labels=["1" "1.1" "1.2" "2" "2.1" "2.2" "3" "3.1" "3.2"])
        add_pareto_plot(test_accuracies_1, sizes_1, colour_1, "1")
        add_pareto_plot(test_accuracies_1_1, sizes_1_1, colour_1_1, "1.1")
        add_pareto_plot(test_accuracies_1_2, sizes_1_2, colour_1_2, "1.2")
        add_pareto_plot(test_accuracies_2, sizes_2, colour_2, "2")
        add_pareto_plot(test_accuracies_2_1, sizes_2_1, colour_2_1, "2.1")
        add_pareto_plot(test_accuracies_2_2, sizes_2_2, colour_2_2, "2.2")
        add_pareto_plot(test_accuracies_3, sizes_3, colour_3, "3")
        add_pareto_plot(test_accuracies_3_1, sizes_3_1, colour_3_1, "3.1")
        add_pareto_plot(test_accuracies_3_2, sizes_3_2, colour_3_2, "3.2")
        savefig(fig, "./diagrams/$(dataset_name)_test_population_metrics.pdf")

        #5.1: find best individuals by validation set accuracy
        function best_individual_index(accuracies, sizes)
            # get the highest-accuracy individual
            highest_accuracy = accuracies[1]
            smallest_size = sizes[1]
            result_index = 1
            for i in 2:length(sizes)
                next_accuracy = accuracies[i]
                next_size = sizes[i]
                # Conditional check passes if accuracy is better or if accuracy is the same but size is better
                if next_accuracy > highest_accuracy || (next_accuracy == highest_accuracy && next_size < smallest_size)
                    highest_accuracy = next_accuracy
                    smallest_size = next_size
                    result_index = i
                end
            end
            return result_index
        end
        best_index_1 = best_individual_index(validation_accuracies_1, sizes_1)
        best_index_1_1 = best_individual_index(validation_accuracies_1_1, sizes_1_1)
        best_index_1_2 = best_individual_index(validation_accuracies_1_2, sizes_1_2)
        best_index_2 = best_individual_index(validation_accuracies_2, sizes_2)
        best_index_2_1 = best_individual_index(validation_accuracies_2_1, sizes_2_1)
        best_index_2_2 = best_individual_index(validation_accuracies_2_2, sizes_2_2)
        best_index_3 = best_individual_index(validation_accuracies_3, sizes_3)
        best_index_3_1 = best_individual_index(validation_accuracies_3_1, sizes_3_1)
        best_index_3_2 = best_individual_index(validation_accuracies_3_2, sizes_3_2)
        

        # calculate accuracies of best individuals
        test_set_accuracy(model) = accuracy(model, test_samples, test_labels)
        train_set_accuracy(model) = accuracy(model, train_samples, train_labels)
        
        groups = ["Training", "Testing", "Validation"]
        models = [
            models_1[best_index_1]
            models_1_1[best_index_1_1]
            models_1_2[best_index_1_2]
            models_2[best_index_2]
            models_2_1[best_index_2_1]
            models_2_2[best_index_2_2]
            models_3[best_index_3]
            models_3_1[best_index_3_1]
            models_3_2[best_index_3_2]
         ]
        train_set_accuracies = train_set_accuracy.(models)
        test_set_accuracies = test_set_accuracy.(models)
        validation_accuracies = [
                                 validation_accuracies_1[best_index_1]
                                 validation_accuracies_1_1[best_index_1_1]
                                 validation_accuracies_1_2[best_index_1_2]
                                 validation_accuracies_2[best_index_2]
                                 validation_accuracies_2_1[best_index_2_1]
                                 validation_accuracies_2_2[best_index_2_2]
                                 validation_accuracies_3[best_index_3]
                                 validation_accuracies_3_1[best_index_3_1]
                                 validation_accuracies_3_2[best_index_3_2]
                                 ]

        improvements_1_1 = validation_accuracies_1_1 .- validation_accuracies_1
        improvements_1_2 = validation_accuracies_1_2 .- validation_accuracies_1
        println("Average validation improvement over approach 1 (approach 1.1):", sum(improvements_1_1) / length(improvements_1_1))
        println("Average validation improvement over approach 1 (approach 1.2):", sum(improvements_1_2) / length(improvements_1_2))
        improvements_2_1 = validation_accuracies_2_1 .- validation_accuracies_2
        improvements_2_2 = validation_accuracies_2_2 .- validation_accuracies_2
        println("Average validation improvement over approach 2 (approach 2.1):", sum(improvements_2_1) / length(improvements_2_1))
        println("Average validation improvement over approach 2 (approach 2.2):", sum(improvements_2_2) / length(improvements_2_2))
        improvements_3_1 = validation_accuracies_3_1 .- validation_accuracies_3
        improvements_3_2 = validation_accuracies_3_2 .- validation_accuracies_3
        println("Average validation improvement over approach 3 (approach 3.1):", sum(improvements_3_1) / length(improvements_3_1))
        println("Average validation improvement over approach 3 (approach 3.2):", sum(improvements_3_2) / length(improvements_3_2))

        best_solution_size_metrics = [
                                      sizes_1[best_index_1],
                                      sizes_1_1[best_index_1_1],
                                      sizes_1_2[best_index_1_2],
                                      sizes_2[best_index_2],
                                      sizes_2_1[best_index_2_1],
                                      sizes_2_2[best_index_2_2],
                                      sizes_3[best_index_3],
                                      sizes_3_1[best_index_3_1],
                                      sizes_3_2[best_index_3_2],
                                     ]
        # print details on best models for document table
        println("Information about models that achieved the highest validation accuracy (train, test, validation, size metric):")
        data_string(approach_number, approach_index) = "$dataset_printing_name approach $approach_number: ($(train_set_accuracies[approach_index]), $(test_set_accuracies[approach_index]), $(validation_accuracies[approach_index]), $(best_solution_size_metrics[approach_index]))"
        for (i, anr) in enumerate([1, 1.1, 1.2, 2, 2.1, 2.2, 3, 3.1, 3.2])
            println(data_string(anr, i))
        end

        #5.2: draw and save best circuits for each approach
        println("Drawing best circuits")
        save_text(draw_chromosome_latex(population_1[best_index_1], dataset.feature_count, 6, 6),
                    "./diagrams/$(dataset_name)_best_circuit_1.tex")
        save_text(draw_chromosome_latex(population_1[best_index_1_1], dataset.feature_count, 6, 6; parameters=parameters_1_1[best_index_1_1]),
                    "./diagrams/$(dataset_name)_best_circuit_1_1.tex")
        save_text(draw_chromosome_latex(population_1[best_index_1_2], dataset.feature_count, 6, 6; parameters=parameters_1_2[best_index_1_2]),
                    "./diagrams/$(dataset_name)_best_circuit_1_2.tex")
        
        save_text(draw_chromosome_latex(population_2[best_index_2], dataset.feature_count, 6, 6),
                    "./diagrams/$(dataset_name)_best_circuit_2.tex")
        save_text(draw_chromosome_latex(population_2[best_index_2_1], dataset.feature_count, 6, 6; parameters=parameters_2_1[best_index_2_1]),
                    "./diagrams/$(dataset_name)_best_circuit_2_1.tex")
        save_text(draw_chromosome_latex(population_2[best_index_2_2], dataset.feature_count, 6, 6; parameters=parameters_2_2[best_index_2_2]),
                    "./diagrams/$(dataset_name)_best_circuit_2_2.tex")
        
        save_text(draw_chromosome_latex(population_3[best_index_3], dataset.feature_count, 6, 6),
                    "./diagrams/$(dataset_name)_best_circuit_3.tex")
        save_text(draw_chromosome_latex(population_3[best_index_3_1], dataset.feature_count, 6, 6; parameters=parameters_3_1[best_index_3_1]),
                    "./diagrams/$(dataset_name)_best_circuit_3_1.tex")
        save_text(draw_chromosome_latex(population_3[best_index_3_2], dataset.feature_count, 6, 6; parameters=parameters_3_2[best_index_3_2]),
                    "./diagrams/$(dataset_name)_best_circuit_3_2.tex")
        
        #6: draw roc curves of the approaches on the same graph
        println("Drawing ROC curves")
        function add_roc_curve(model, colour, label)
            kernel_outputs = compute_kernel_matrix(model.kernel, model.training_set, dataset.validation_samples)
            df_outputs = decision_function(model.classifier, kernel_outputs)
            true_positive_rates, false_positive_rates = [], []
            width=0.001
            for boundary in -3:width:3
                # label samples using the current boundary
                predictions = [o < boundary ? -1 : 1 for o in df_outputs]
                # count true and false positives
                true_positive_p = ((true_label, predicted_label),) -> predicted_label == 1 && true_label == 1
                false_positive_p = ((true_label, predicted_label),) -> predicted_label == 1 && true_label == -1
                tp = count(true_positive_p, zip(dataset.validation_labels, predictions))
                fp = count(false_positive_p, zip(dataset.validation_labels, predictions))
                tpr = tp / dataset.num_positive_validation_instances
                fpr = fp / dataset.num_negative_validation_instances
                push!(true_positive_rates, tpr)
                push!(false_positive_rates, fpr)
            end
            Plots.plot!(false_positive_rates, true_positive_rates, colour=colour, label=label)
        end
        # make axis for roc curve plots
        fig = Plots.plot(title="$(dataset_printing_name) ROC curves for the best produced models", legend=true,
                        xlabel="False positive rate", ylabel="True positive rate",
                        xlim=(0, 1), ylim=(0, 1),
                        labels=["1" "1.1" "1.2" "2" "2.1" "2.2" "3" "3.1" "3.2"])

        add_roc_curve(models_1[best_index_1], colour_1, "1")
        add_roc_curve(models_1_1[best_index_1_1], colour_1_1, "1.1")
        add_roc_curve(models_1_2[best_index_1_2], colour_1_2, "1.2")
        add_roc_curve(models_2[best_index_2], colour_2, "2")
        add_roc_curve(models_2_1[best_index_2_1], colour_2_1, "2.1")
        add_roc_curve(models_2_2[best_index_2_2], colour_2_2, "2.2")
        add_roc_curve(models_3[best_index_3], colour_3, "3")
        add_roc_curve(models_3_1[best_index_3_1], colour_3_1, "3.1")
        add_roc_curve(models_3_2[best_index_3_2], colour_3_2, "3.2")
        savefig(fig, "./diagrams/$(dataset_name)_roc_curves.pdf")

        #7: draw confusion matrices for the best individuals
        println("Drawing confusion matrices")
        confusion_matrix(models_1[best_index_1], dataset.validation_samples, dataset.validation_labels)
        py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_1.pdf")
        confusion_matrix(models_1_1[best_index_1_1], dataset.validation_samples, dataset.validation_labels)
        py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_1_1.pdf")
        confusion_matrix(models_1_2[best_index_1_2], dataset.validation_samples, dataset.validation_labels)
        py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_1_2.pdf")
        confusion_matrix(models_2[best_index_2], dataset.validation_samples, dataset.validation_labels)
        py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_2.pdf")
        confusion_matrix(models_2_1[best_index_2_1], dataset.validation_samples, dataset.validation_labels)
        py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_2_1.pdf")
        confusion_matrix(models_2_2[best_index_2_2], dataset.validation_samples, dataset.validation_labels)
        py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_2_2.pdf")
        confusion_matrix(models_3[best_index_3], dataset.validation_samples, dataset.validation_labels)
        py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_3.pdf")
        confusion_matrix(models_3_1[best_index_3_1], dataset.validation_samples, dataset.validation_labels)
        py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_3_1.pdf")
        confusion_matrix(models_3_2[best_index_3_2], dataset.validation_samples, dataset.validation_labels)
        py"plt.savefig"("./diagrams/$(dataset_name)_confusion_matrix_3_2.pdf")

        #9: graph train, test and validation accuracies for each approach
        println("Plotting classification accuracies graph")
        
        classical_model = SVC(kernel="rbf", class_weight="balanced")
        fit!(classical_model, train_samples, train_labels) #use same training data as genetically produced QSVM kernels
        classical_accuracies = [score(classical_model,  train_samples, train_labels) score(classical_model, test_samples, test_labels) score(classical_model, dataset.validation_samples, dataset.validation_labels)]
        println("Classical accuracies: ", classical_accuracies)

        accuracies(i) = [train_set_accuracies[i] test_set_accuracies[i] validation_accuracies[i]]
        y1s = accuracies(1)
        y1_1s = accuracies(2)
        y1_2s = accuracies(3)
        y2s = accuracies(4)
        y2_1s = accuracies(5)
        y2_2s = accuracies(6)
        y3s = accuracies(7)
        y3_1s = accuracies(8)
        y3_2s = accuracies(9)
        ys = [
            classical_accuracies;
            y1s;
            y1_1s;
            y1_2s;
            y2s;
            y2_1s;
            y2_2s;
            y3s;
            y3_1s;
            y3_2s
             ]'
        fig = Plots.plot(title="$(dataset_printing_name) accuracies for each approach and data subset",
                         legend=true,
                         ylim=(0, 2))
        groupedbar!(groups, ys, labels=["SVM RBF kernel" "1 - Original" "1.1" "1.2" "2 - Alignment" "2.1" "2.2" "3 - Approximation" "3.1" "3.2"],
                                color=[classical_colour colour_1 colour_1_1 colour_1_2 colour_2 colour_2_1 colour_2_2 colour_3 colour_3_1 colour_3_2],
                                ylabel="Accuracy on subset")
        """
        xs = ["1"
              "2.1"
              "2.2"
              "3.1"
              "3.2"
              ]
        models = [
                    models_1[best_index_1]
                    models_2_1[best_index_2_1]
                    models_2_2[best_index_2_2]
                    models_3_1[best_index_3_1]
                    models_3_2[best_index_3_2]
                 ]
        y0s = train_set_accuracy.(models)
        y1s = test_set_accuracy.(models)
        y2s = [
                validation_accuracies_1[best_index_1]
                validation_accuracies_2_1[best_index_2_1]
                validation_accuracies_2_2[best_index_2_2]
                validation_accuracies_3_1[best_index_3_1]
                validation_accuracies_3_2[best_index_3_2]
              ]
        # create plot
        fig = Plots.plot(title="$(dataset_printing_name) accuracies for each approach", legend=true, ylim=(0, 1.2)) # set ylim past 100 to make space for the legend
        #annotation_text(x) = Plots.text(x, pointsize=1)
        ys = [y0s y1s y2s]
        groupedbar!(xs, ys, labels=["Train" "Test" "Validation"])#, series_annotations=annotation_text.([string.(y0s) string.(y1s) string.(y2s)]))
        """
        """
        # add train, test and validation accuracies
        #Plots.bar!(xs, y0s, colour="green", label="Train")
        #Plots.bar!(xs, y1s, colour="blue", label="Test")
        #Plots.bar!(xs, y2s, colour="orange", label="Validation")
        """
        savefig(fig, "./diagrams/$(dataset_name)_accuracies.pdf")

        #10: parameter training history (for convergence analysis)
        println("Plotting parameter training convergence")
        function plot_parameter_history(y_axis_label, approach_number, history)
            fig = Plots.plot(1:100, history, xlabel="$y_axis_label evaluations", ylabel=y_axis_label, title="$dataset_printing_name, approach $approach_number parameter training", legend=false)
            savefig(fig, "./diagrams/$(dataset_name)_optimization_history_$(approach_number).pdf")
        end
        best_history_1_1 = parameter_history_1_1[best_index_1_1]
        plot_parameter_history("RMSE", 1.1, best_history_1_1)
        best_history_1_2 = parameter_history_1_2[best_index_1_2]
        plot_parameter_history("Kernel-target alignment", 1.2, best_history_1_2)
        best_history_2_1 = parameter_history_2_1[best_index_2_1]
        plot_parameter_history("RMSE", 2.1, best_history_2_1)
        best_history_2_2 = parameter_history_2_2[best_index_2_2]
        plot_parameter_history("Kernel-target alignment", 2.2, best_history_2_2)
        best_history_3_1 = parameter_history_3_1[best_index_3_1]
        plot_parameter_history("RMSE", 3.1, best_history_3_1)
        best_history_3_2 = parameter_history_3_2[best_index_3_2]
        plot_parameter_history("Kernel-target alignment", 3.2, best_history_3_2)


        println("Plotting margin measures")
        #NOTE: the margin measure doesn't need to be weighted to correct for class imbalance
        # since the training set should be approximately balanced as it is taken randomly from
        # a balanced subset of the dataset
        function margin_measures_bar_graph()
            "Return mean and standard deviation of margin values for the model on the training data."
            function margin_measures(model)
                train_margins = decision_function(model.classifier, model.gram_matrix)
                train_margins_sign_corrected = train_margins .* model.training_labels
                average = mean(train_margins_sign_corrected)
                error_value = std(train_margins_sign_corrected)
                return average, error_value
            end
            approach_names = ["1", "1.1", "1.2", "2", "2.1", "2.2", "3", "3.1", "3.2"]
            margin_means_and_deviations = margin_measures.(models)
            ys = []
            errors = []
            for (y, err) in margin_means_and_deviations
                push!(ys, y)
                push!(errors, err)
            end
            fig = Plots.plot(ylabel="Mean margin on training points", title="Margins of best models with std. deviations", legend=false)
            bar!(approach_names, ys, yerr=errors,
                colour=[colour_1, colour_1_1, colour_1_2, colour_2, colour_2_1, colour_2_2, colour_3, colour_3_1, colour_3_2])
            savefig(fig, "./diagrams/$(dataset_name)_margins.pdf")
        end
        function margin_measures_boxplot()
            function margins(model)
                train_margins = decision_function(model.classifier, model.gram_matrix)
                train_margins_sign_corrected = train_margins .* model.training_labels
                return train_margins_sign_corrected
            end
            approach_names = ["1", "1.1", "1.2", "2", "2.1", "2.2", "3", "3.1", "3.2"]
            #TODO: fix ys calculation, it seems that it shouldn't simply take all the margin values as input
            ys = margins.(models)
            fig = Plots.plot(ylabel="Mean margin on training points", title="Margins of best models with std. deviations", legend=false)
            boxplot!(approach_names, ys)
                #colour=[colour_1, colour_1_1, colour_1_2, colour_2, colour_2_1, colour_2_2, colour_3, colour_3_1, colour_3_2])
            savefig(fig, "./diagrams/$(dataset_name)_margins.pdf")
        end
        margin_measures_bar_graph()
        #margin_measures_boxplot()
       
        #8: draw decision boundaries for the best individuals
        if dataset.feature_count == 2
            println("Drawing decision boundaries")
            # compute classification outputs for the contour graph
            axis_interval = -1.2:.01:1.2
            point_to_index(x) = trunc(Int64, (x+1.2)*100)+1
            grid_samples = []
            for x in axis_interval
                for y in axis_interval
                    push!(grid_samples, [x,y])
                end
            end

            "Creates 2 decision boundaries: 1 for training and testing set points, and 1 for validation set points"
            function graph_decision_boundaries(model, approach_number, validation_samples, validation_labels)
                # calculate contour coordinate outputs
                kernel_output_matrix = compute_kernel_matrix(model.kernel, model.training_set, grid_samples)
                outputs = predict(model.classifier, kernel_output_matrix)
                # create contour output matrix and fill in the values
                z_matrix = Matrix(undef, length(axis_interval), length(axis_interval))
                index = 1
                for x in 1:length(axis_interval)
                    for y in 1:length(axis_interval)
                        output = outputs[index]
                        z_matrix[x, y] = (output == -1 ? 1 : 2)
                        index += 1
                    end
                end
                ccol = cgrad([RGB(1,.2,.2), RGB(.2,.2,1)])
                # create first decision boundary plot.
                fig = Plots.plot(leg=:none, title="$(dataset_printing_name) decision boundary, approach $approach_number",
                                lims=(-1.2, 1.2), xlabel="Feature 1", ylabel="Feature 2")
                # draw contour
                contour!(axis_interval, axis_interval, (x,y)->z_matrix[point_to_index(x),point_to_index(y)],
                        f=true, nlev=2, c=ccol)
                # draw training set points
                scatter!([s[1] for s in train_samples],
                        [s[2] for s in train_samples],
                        m=[:rect :circle],
                        color=[label == -1 ? "red" : "blue" for label in train_labels])
                # draw test set points
                scatter!([s[1] for s in test_samples],
                        [s[2] for s in test_samples],
                        m=[:rect :circle],
                        color=[label == -1 ? "pink" : "cyan" for label in test_labels])
                # save to disk
                savefig(fig, "./diagrams/$(dataset_name)_decision_boundary_$(approach_number).pdf")

                # second, plot decision boundary for validation points
                fig2 = Plots.plot(leg=:none, title="$(dataset_printing_name) decision boundary, approach $approach_number, validation data",
                                lims=(-1.2, 1.2), xlabel="Feature 1", ylabel="Feature 2")
                # draw contour again
                contour!(axis_interval, axis_interval, (x,y)->z_matrix[point_to_index(x),point_to_index(y)],
                        f=true, nlev=2, c=ccol)
                # draw validation set points
                scatter!([s[1] for s in validation_samples],
                        [s[2] for s in validation_samples],
                        m=[:rect :circle],
                        color=[label == -1 ? "red" : "blue" for label in validation_labels])
                # save to disk
                savefig(fig2, "./diagrams/$(dataset_name)_decision_boundary_$(approach_number)_validation.pdf")
            end
            graph_decision_boundaries(models_1[best_index_1], 1, dataset.validation_samples, dataset.validation_labels)
            graph_decision_boundaries(models_1_1[best_index_1_1], 1.1, dataset.validation_samples, dataset.validation_labels)
            graph_decision_boundaries(models_1_2[best_index_1_2], 1.2, dataset.validation_samples, dataset.validation_labels)
            graph_decision_boundaries(models_2[best_index_2], 2, dataset.validation_samples, dataset.validation_labels)
            graph_decision_boundaries(models_2_1[best_index_2_1], 2.1, dataset.validation_samples, dataset.validation_labels)
            graph_decision_boundaries(models_2_2[best_index_2_2], 2.2, dataset.validation_samples, dataset.validation_labels)
            graph_decision_boundaries(models_3[best_index_3], 3, dataset.validation_samples, dataset.validation_labels)
            graph_decision_boundaries(models_3_1[best_index_3_1], 3.1, dataset.validation_samples, dataset.validation_labels)
            graph_decision_boundaries(models_3_2[best_index_3_2], 3.2, dataset.validation_samples, dataset.validation_labels)
            #=
            # graph decision boundaries in parallel
            tasks = []
            push!(tasks, Dagger.@spawn graph_decision_boundaries(models_1[best_index_1], 1, dataset.validation_samples, dataset.validation_labels))
            push!(tasks, Dagger.@spawn graph_decision_boundaries(models_1_1[best_index_1_1], 1.1, dataset.validation_samples, dataset.validation_labels))
            push!(tasks, Dagger.@spawn graph_decision_boundaries(models_1_2[best_index_1_2], 1.2, dataset.validation_samples, dataset.validation_labels))
            push!(tasks, Dagger.@spawn graph_decision_boundaries(models_2[best_index_2], 2, dataset.validation_samples, dataset.validation_labels))
            push!(tasks, Dagger.@spawn graph_decision_boundaries(models_2_1[best_index_2_1], 2.1, dataset.validation_samples, dataset.validation_labels))
            push!(tasks, Dagger.@spawn graph_decision_boundaries(models_2_2[best_index_2_2], 2.2, dataset.validation_samples, dataset.validation_labels))
            push!(tasks, Dagger.@spawn graph_decision_boundaries(models_3[best_index_3], 3, dataset.validation_samples, dataset.validation_labels))
            push!(tasks, Dagger.@spawn graph_decision_boundaries(models_3_1[best_index_3_1], 3.1, dataset.validation_samples, dataset.validation_labels))
            push!(tasks, Dagger.@spawn graph_decision_boundaries(models_3_2[best_index_3_2], 3.2, dataset.validation_samples, dataset.validation_labels))
            fetch.(tasks)
            =#
        end
    end
end
