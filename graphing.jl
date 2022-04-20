# TODO: move graphing functions from tests.jl into this file.
# also add other graphing functions for displaying results like
# confusion matrices, ROC curves, decision boundaries, etc.

# TODO: create animation of decision boundary graph changing with
# threshold choice

using Plots, ScikitLearn

function roc_curve(model_struct::TrainedModel, dataset)
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

# dispatch version that also creates the trained model
function roc_curve(chromosome::AbstractVector{Bool}, dataset, n_qubits, depth, n_features=dataset.feature_count)
    problem_data = py"train_test_split"(dataset.training_samples, dataset.training_labels, train_size=0.7)
    kernel = decode_chromosome_yao(chromosome, n_features, n_qubits, depth)
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
        annotate!([(10, 10, text("Generation: $gen", :black, :right, 3))])
        plt
    end every step
    gif(animation, "./diagrams/pareto_front_change $dataset_name $training_type.gif", fps=frame_rate)
end

function visualize_genetic_and_parameter_training(genetic_fitness_history, parameter_fitness_history, dataset_name, genetic_training_type, parameter_training_type)
    target_seconds = 10
    frame_rate = 30
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
            annotate!([(10, 10, text("Generation: $iteration", :black, :right, 3))])
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
            annotate!([(10, 10, text("Fitness evaluations: $parameter_training_evaluations", :black, :right, 3))])
            plt
        else
            # case of inserting waiting frames to extend animation length
            #(do nothing)
        end
    end every step
    gif(animation, "./diagrams/pareto_front_change_with_parameter_training $dataset_name $genetic_training_type $parameter_training_type.gif", fps=frame_rate)
end
