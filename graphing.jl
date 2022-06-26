# TODO: move graphing functions from tests.jl into this file.
# also add other graphing functions for displaying results like
# confusion matrices, ROC curves, decision boundaries, etc.

# TODO: create animation of decision boundary graph changing with
# threshold choice

using Plots, ScikitLearn
using Compose

py"""
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
"""

function roc_curve(model_struct::TrainedModel, dataset::Dataset)
    # compute kernel values for model's training data and the dataset's validation samples
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
    # for each unique point on the curve, sum the height at the point times width
    # to the next point
    #auc = 0.0
    #for pair in zip(true_positive_rates, false_positive_rates)
            #auc += pair[1] * width / length
    #end
    #auc = round(auc, digits=4)
    dataset_name = dataset.name
    title = "Roc curve, $dataset_name dataset"#, AUC=$auc"
    Plots.plot(false_positive_rates, true_positive_rates, plot_title=title, xlabel="FP rate", ylabel="TP rate")
end

# dispatch version that takes a chromosome
function roc_curve(chromosome::AbstractVector{Bool}, dataset::Dataset, n_qubits, depth)
    n_features = dataset.feature_count
    kernel = decode_chromosome_yao(chromosome, n_features, n_qubits, depth)
    roc_curve(kernel, dataset)
end

# dispatch version that takes a kernel and first creates the model
function roc_curve(kernel, dataset::Dataset)
    problem_data = py"train_test_split"(dataset.training_samples, dataset.training_labels, train_size=0.7)
    model = train_model(problem_data, kernel)
    roc_curve(model, dataset)
end

function decision_boundary(model_struct, dataset)
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

    ccol = cgrad([RGB(1,.1,.1), RGB(.1,.1,1)])
    mcol = [RGB(1,.1,.1) RGB(.1,.1,1)]

    name = dataset.name
    contour(axis_interval, axis_interval, point_to_output,
            f=true, nlev=2, c=ccol, leg=:none, title="$name decision boundary",
            xlabel="Feature 1", ylabel="Feature 2")
    scatter!([s[1] for s in dataset.training_samples],
             [s[2] for s in dataset.training_samples],
             m=[:rect :circle],
             c=mcol,
             lims=(-1.2, 1.2))
end

#TODO: write a method for this function that loads results for the data set and
# draws the decision boundary for the highest accuracy classifier in the results
# by calling the other methods for this function
function decision_boundary(dataset::Dataset)
end

function plot_final_fitness_pareto_front(final_fitnesses; dataset_name="undef", training_type="undef")
    xs = [final_fitnesses[i][2] for i in 1:length(final_fitnesses)] #size values
    ys = [-final_fitnesses[i][1] for i in 1:length(final_fitnesses)] #accuracy values
    # plot pareto front with 
    Plots.plot(xs, ys,
               seriestype=:scatter,
               title="Final generation pareto front ($dataset_name, $training_type)",
               xlim=(0, 6), ylim=(0.35, 1),
               xlabel="Size metric", ylabel="Accuracy metric",
               legend=false)
end

#TODO: compute xlim bounds by checking the max and min of xs over all generations
"Animates the history of multi-objective fitness values changing
over genetic optimization generations. This can be used to visualize
the convergence of the genetic optimization."
function animate_genetic_fitness_history(fitness_history; dataset_name="undef", training_type="undef")
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
                         xlim=(0, 6), ylim=(0.35, 1),
                         xlabel="Size metric", ylabel="Accuracy metric",
                         legend=false)
        # annotate with generation number
        annotate!([(10, 10, Plots.text("Generation: $gen", :black, :right, 3))])
        plt
    end every step
    gif(animation, "./diagrams/pareto_front_change $dataset_name $training_type.gif", fps=frame_rate)
end

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
    py"sn.set(font_scale=2)" # set label size
    result = py"sn.heatmap"(dataframe, annot=true, annot_kws=Dict("size"=>20)) # set font size
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

#POPULATION_TYPE
#SINGLE_CLASSIFIER_TYPE
"Load results, then create and save graphs for each data set and approach being compared (original, parameter refinement, trained accuracy metric)."
function generate_graphs(seed=22)
    # for each data set
    for dataset_name in ["moons", "digits", "cancer", "iris"]
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
        # 2. load results from genetic optimization that use trained accuracy
        #    in genetic fitness function. if the final parameters weren't saved,
        #    re-train them to retrieve the original models. the saved fitness values will
        #    still be correct
        #    2.1 draw graphs for this approach like for steps 1.1 and 1.2

        # GRAPHS TO DRAW IN EACH STEP:
        # draw pareto fronts for each dataset and configuration
        # draw accuracy over pareto fronts change with successive parameter training
        # draw decision boundaries
        # draw roc curves

        println("Dataset: $dataset_name")
        println("Step 1: Approaches 1 and 2")
        # step 1
        println("Step 1.1: Approach 1")
        population, fitnesses, history = load_results(dataset_name, "accuracy")
        # step 1.1
        println("Step 1.1.1: Final pareto front")
        # 1.1.1 Pareto front
        fig = plot_final_fitness_pareto_front(fitnesses; dataset_name=dataset_name, training_type="accuracy")
        savefig(fig, "./diagrams/$dataset_name accuracy final_fitness_pareto_front.pdf")
        # 1.1.2 Pareto front animation
        println("Step 1.1.2: Pareto front animation")
        animate_genetic_fitness_history(history; dataset_name=dataset_name, training_type="accuracy")
        # 1.1.3 Best individual circuit
        println("Step 1.1.3: Best individual circuit")
        best_chromosome_index = best_individual_index(population, fitnesses)
        best_chromosome = population[best_chromosome_index]
        figure = draw_chromosome(best_chromosome, dataset.feature_count, 6, 6) # the draw_chromosome function is defined in tests.jl, maybe move the definition to this file        
        figure |> PDF("./diagrams/$dataset_name accuracy best_individual_circuit.pdf")
        #savefig(figure, ./diagrams/$dataset_name accuracy best_individual_circuit.tex") # save circuits in tex output format so that the symbolic figures can be corrected for parameters to fit in the gate blocks
        
        # for the further stages of step 1, the best individual model must be instantiated from the chromosome and training dataset
        kernel = decode_chromosome_yao(best_chromosome, dataset.feature_count, 6, 6)
        
        train_percent = 0.7     # fraction of data to use for training, check nsga2.jl for the correct value to use (maybe change for flexibility or to read a variable instead of using a constant literal)
        problem_data = py"train_test_split"(dataset.training_samples,
                                            dataset.training_labels,
                                            train_size=train_percent,
                                            random_state=seed,
                                            shuffle=true)
        model_struct = train_model(problem_data, kernel, seed)
        
        # 1.1.4 Best individual ROC curve
        println("Step 1.1.4: Best individual ROC curve")
        figure = roc_curve(model_struct, dataset)
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
            figure = decision_boundary(model_struct, dataset)
            savefig(figure, "./diagrams/$dataset_name accuracy best_individual_decision_boundary.pdf")
        end

        # Step 1.2: perform parameter training to get 2nd approach results (maybe separate training into experiments.jl and just load the results here?)
        println("Step 1.2: Approach 2")
        population_final_parameters, parameter_training_fitness_history = population_parameterised_training(population, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=300, seed=seed, metric_type="accuracy")
        # This below variable is the recreation of the fitnesses variable for the second approach (parameter refinement approach).
        # it holds the multi-objective fitness values for the final population after parameter training. It uses the last recorded
        # accuracy from parameter training as the accuracy and the size metric is copied from the first approach fitnesses.
        # Note that the parameterised training doesn't return negated accuracy, so it must be manually negated here to be consistent
        # with the form of the accuracies of the first approach.
        fitnesses_2 = [[-parameter_training_fitness_history[i][end], fitnesses[i][2]] for i in 1:length(population)]
        # 1.2.1-1.2.6 follow similarly to 1.1.1-1.1.6, so some repeated comments are left out
        # 1.2.1 Pareto front
        println("Step 1.2.1: Final pareto front")
        figure = plot_final_fitness_pareto_front(fitnesses_2; dataset_name=dataset_name, training_type="parameters accuracy trained")
        savefig(figure, "./diagrams/$dataset_name accuracy trained final_fitness_pareto_front.pdf")
        # 1.2.2 Pareto front animation
        println("Step 1.2.2: Pareto front animation")
        visualize_genetic_and_parameter_training(history, parameter_training_fitness_history, dataset, "accuracy", "accuracy", seed)
        # 1.2.3 Best individual circuit
        println("Step 1.2.3: Best individual circuit")
        best_chromosome_index_2 = best_individual_index(population, fitnesses_2)
        best_chromosome_2 = population[best_chromosome_index_2]
        best_chromosome_trained_parameters = population_final_parameters[best_chromosome_index_2]
        best_parameterised_kernel, best_initial_parameters = decode_chromosome_parameterised_yao(best_chromosome_2, dataset.feature_count, 6, 6)
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
        figure = draw_kernel(best_kernel, dataset.feature_count)
        figure |> PDF("./diagrams/$dataset_name accuracy trained best_individual_circuit.pdf")
        #savefig(figure, "./diagrams/$dataset_name accuracy trained best_individual_circuit.tex")

        # problem_data variable is defined when graphing first approach
        model_struct_2 = train_model(problem_data, best_kernel, seed)
        
        # 1.2.4 Best individual ROC curve
        println("Step 1.2.4: Best individual ROC curve")
        figure = roc_curve(model_struct_2, dataset)
        savefig(figure, "./diagrams/$dataset_name accuracy trained best_individual_roc_curve.pdf")
        # 1.2.5 Best individual confusion matrix
        println("Step 1.2.5: Best individual confusion matrix")
        cm = confusion_matrix(model_struct_2, dataset)
        py"plt.savefig"("./diagrams/$dataset_name accuracy trained best_individual_confusion_matrix.pdf")
        # 1.2.6 Best individual decision boundary (for 2D datasets)
        if dataset.feature_count == 2
            println("Step 1.2.6: Best individual decision boundary")
            figure = decision_boundary(model_struct_2, dataset)
            savefig(figure, "./diagrams/$dataset_name accuracy trained best_individual_decision_boundary.pdf")
        end

        # step 2
        println("Step 2: Approach 3")
        population_3, fitnesses_3, history_3 = load_results(dataset_name, "accuracy_parameter_training")
        population_final_parameters_3, parameter_training_fitness_history_3 = population_parameterised_training(population_3, problem_data, dataset.feature_count; qubit_count=6, depth=6, max_evaluations=300, seed=seed, metric_type="accuracy")
        # 2.1
        println("Step 2.1: Draw graphs from Step 1.1")
        # 2.1.1 Pareto front
        println("Step 2.1.1: Final pareto front")
        figure = plot_final_fitness_pareto_front(fitnesses_3; dataset_name=dataset_name, training_type="accuracy training in fitness")
        savefig(figure, "./diagrams/$dataset_name accuracy training in fitness final_fitness_pareto_front.pdf")
        # 2.1.2 Pareto front animation
        println("Step 2.1.2: Pareto front animation")
        animate_genetic_fitness_history(history_3; dataset_name=dataset_name, training_type="accuracy training in fitness")
        # 2.1.3 Best individual circuit
        println("Step 2.1.3: Best individual circuit")
        best_chromosome_index_3 = best_individual_index(population_3, fitnesses_3)
        best_chromosome_3 = population_3[best_chromosome_index_3]
        best_chromosome_trained_parameters_3 = population_final_parameters_3[best_chromosome_index_3]
        best_parameterised_kernel_3, best_initial_parameters_3 = decode_chromosome_parameterised_yao(best_chromosome_3, dataset.feature_count, 6, 6)
        best_kernel_3 = best_parameterised_kernel_3(best_chromosome_trained_parameters_3)
        figure = draw_kernel(best_kernel_3, dataset.feature_count)
        figure |> PDF("./diagrams/$dataset_name accuracy training in fitness best_individual_circuit.pdf")
        #savefig(figure, "./diagrams/$dataset_name accuracy trained best_individual_circuit.tex")

        # problem_data variable is defined when graphing first approach
        model_struct_3 = train_model(problem_data, best_kernel_3, seed)
        
        # 2.1.4 Best individual ROC curve
        println("Step 2.1.4: Best individual ROC curve")
        figure = roc_curve(model_struct_3, dataset)
        savefig(figure, "./diagrams/$dataset_name accuracy training in fitness best_individual_roc_curve.pdf")
        # 2.1.5 Best individual confusion matrix
        println("Step 2.1.5: Best individual confusion matrix")
        cm = confusion_matrix(model_struct_3, dataset)
        py"plt.savefig"("./diagrams/$dataset_name accuracy training in fitness best_individual_confusion_matrix.pdf")
        # 2.1.6 Best individual decision boundary (for 2D datasets)
        if dataset.feature_count == 2
            println("Step 2.1.6: Best individual decision boundary")
            figure = decision_boundary(model_struct_3, dataset)
            savefig(figure, "./diagrams/$dataset_name accuracy training in fitness best_individual_decision_boundary.pdf")
        end
    end
end
