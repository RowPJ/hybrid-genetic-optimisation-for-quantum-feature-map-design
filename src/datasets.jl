# This file is responsible for loading data sets and preprocessing them by selecting two classes,
# scaling the features, and splitting them into training and validation sets, with +-75 samples
# from each class being used for training and the remaining samples being used for validation of
# the genetic method.

#TODO: perform data set shuffling before doing the training / validation split.
# figure out how to set the shuffling seed for reproducible results

using PyCall
using JLD2 # for saving and loading adhoc dataset

# for loading datasets
using DataFrames
using CSV

# ensure python dependencies (modules and user defined functions) are loaded
py"""
from sklearn.datasets import make_moons, make_blobs, make_circles, load_iris, load_breast_cancer, load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def extract_binary_classes(feature_array, label_array, class_override):
    #Takes a numpy array of feature vectors and a numpy array of labels
    #and returns transformed numpy arrays with the number of classes reduced
    #to 2. Picks two random classes.
    # get 2 random classes based on the seed, and output what they are
    seed = 22 #use a fixed class-selection seed so classes don't change across different seed tests
    all_classes = list(set(label_array))
    if class_override is not None:
        classes = class_override
    else:
        classes = np.random.default_rng(seed).choice(all_classes, size=2, replace=False) # must have replace=False to guarantee different classes
    print(f"Choosing classes {classes}.")
    class_map = {classes[0]:0, classes[1]:1} # convert labels to 0 and 1
    # construct a feature and label description with information from only the first 2 classes
    features = []
    labels = []
    for (feature, label) in zip(feature_array, label_array):
        if label in classes:
            features.append(feature)
            labels.append(label)
    class_split = np.unique(labels, return_counts=True)
    print(f"Class split: {class_split}")
    # also return selected class indices
    return (np.array(features), np.array(labels), classes)

def process_dataset(feature_vectors, labels, binary_classification=True, class_override=None):
    # maybe extract classes for binary classification
    if binary_classification:
        feature_vectors, labels, classes = extract_binary_classes(feature_vectors, labels, class_override)
    else:
        classes = list(set(labels))

    # Now we standardize for gaussian around 0 with unit variance
    scaler = StandardScaler()
    scaler.fit(feature_vectors)
    feature_vectors = scaler.transform(feature_vectors)

    # Scale to the range (-1,+1)
    minmax_scaler = MinMaxScaler((-1, 1)).fit(feature_vectors)
    feature_vectors = minmax_scaler.transform(feature_vectors)

    # replace labels with +1 and -1 for kernel
    # target alignment computations to be valid.
    if not binary_classification:
        raise Exception()
    label_types = list(set(labels))
    label_map = {label_types[0]:-1, label_types[1]:1}
    labels = [label_map[l] for l in labels]

    # return samples, labels, and the class selection
    return feature_vectors, labels, classes
"""

"This struct keeps track data set attributes for
convenient access."
struct Dataset
    training_samples
    training_labels
    validation_samples
    validation_labels
    class_indices
    class_names
    feature_count
    training_sample_count
    validation_sample_count
    num_positive_training_instances
    num_negative_training_instances
    num_positive_validation_instances
    num_negative_validation_instances
    name
end

"Splits the data set into disjoint training and validation subsets.
training_size determines the number of samples in the training set,
split evenly between the two classes. The remaining samples go into the
validation set. NOTE: This function must only be called after scaling the
data set and replacing the labels with -1 and 1."
function separate_training_and_validation_sets(samples, labels, training_size)
    # arrays to hold the split data
    training_samples::Vector{Vector{Float64}} = []
    training_labels::Vector{Real} = []
    remaining_samples::Vector{Vector{Float64}} = []
    remaining_labels::Vector{Real} = []

    # number of samples of each count in the training data
    training_count_minus = 0
    training_count_plus = 0

    # each class should make up half the training data
    samples_per_class = training_size ÷ 2

    for (index, (sample, label)) in enumerate(zip(samples, labels))
        # if sample is from positive class
        if label == 1
            # if there aren't enough positive samples in training set
            if training_count_plus < samples_per_class
                # include the sample in training set
                training_count_plus += 1
                push!(training_samples, sample)
                push!(training_labels, label)
            else
                # otherwise include the sample in validation set
                push!(remaining_samples, sample)
                push!(remaining_labels, label)
            end
        # else if sample is from negative class
        elseif label == -1
            # if there aren't enough negative samples in training set
            if training_count_minus < samples_per_class
                # include sample in training set
                training_count_minus += 1
                push!(training_samples, sample)
                push!(training_labels, label)
            else
                # else include sample in validation set
                push!(remaining_samples, sample)
                push!(remaining_labels, label)
            end
        else
            # if label is not recognized, error
            error("Found a label $label that was not equal to 1 or -1. Ensure that the labels been replaced before calling this function.")
        end
    end

    #trim validation data to be balanced by class
    remaining_count_plus = count(==(1), remaining_labels)
    remaining_count_minus = length(remaining_labels) - remaining_count_plus
    validation_target = min(remaining_count_plus, remaining_count_minus)
    validation_plus_counter = 0
    validation_minus_counter = 0
    validation_samples::Vector{Vector{Float64}} = []
    validation_labels::Vector{Real} = []
    for (sample, label) in zip(remaining_samples, remaining_labels)
        if label == 1
            if validation_plus_counter < validation_target
                push!(validation_samples, sample)
                push!(validation_labels, label)
                validation_plus_counter += 1
            end
        elseif label == -1
            if validation_minus_counter < validation_target
                push!(validation_samples, sample)
                push!(validation_labels, label)
                validation_minus_counter += 1
            end
        end
    end

    # ensure that there are the desired number of samples in each class.
    # there should be an equal number of samples in each class to make a
    # balanced learning problem, and the number of samples should sum
    # to approximately the argument training size. it's fine to have
    # one less sample then requested since that will be inevitable if
    # training_size is odd.
    if training_count_minus + training_count_plus < training_size - 1
        error("Desired train:test ratio of $samples_per_class:$samples_per_class, the achieved ratio was $training_count_minus:$training_count_plus")
    end
    return ((training_samples, training_labels), (validation_samples, validation_labels))
end


# initialize variables if they don't already have a value
if !isdefined(Main, :cancer_dataset)
    cancer_dataset = nothing
end
"Loads processed cancer data set training and validation sets."
function load_cancer(;num_train_samples=300, target_dimensionality=10)
    # load data
    dataset_dict = py"load_breast_cancer()"

    # get samples and labels
    samples, labels = dataset_dict["data"], dataset_dict["target"]

    # convert samples to a list of rows instead of a matrix
    row(m, i) = @view m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    samples = to_rows(samples)

    # dimensionality reduction
    pca = py"PCA"(n_components=target_dimensionality)
    pca.fit(samples)
    samples = to_rows(pca.transform(samples))

    # scaling, and replacing labels with -1 and 1
    samples, labels, chosen_classes = py"process_dataset"(samples, labels)
    samples = to_rows(samples)

    # record names of selected classes
    all_class_names = dataset_dict["target_names"]
    class_names = [all_class_names[chosen_classes[1]+1], all_class_names[chosen_classes[2]+1]]

    # ensure training/validation split size is valid
    if num_train_samples > length(samples)
        error("num_train_samples can be at most $length(samples), but should be lower to leave validation data.")
    end

    # split data into training and validation pairs
    training_pair, validation_pair = separate_training_and_validation_sets(samples, labels, num_train_samples)

    num_positive(l) = count(==(1), l)
    num_negative(l) = count(==(-1), l)

    # save loaded data to variables
    global cancer_dataset
    cancer_dataset = Dataset(training_pair[1],
                             training_pair[2],
                             validation_pair[1],
                             validation_pair[2],
                             chosen_classes,
                             class_names,
                             target_dimensionality,
                             length(training_pair[1]),
                             length(validation_pair[1]),
                             num_positive(training_pair[2]),
                             num_negative(training_pair[2]),
                             num_positive(validation_pair[2]),
                             num_negative(validation_pair[2]),
                             "cancer")
    
    nothing
end


# similar process for moons data set, except simpler since data is generated
if !isdefined(Main, :moons_dataset)
    moons_dataset = nothing
end
"Generates and processes moons training and validation sets."
function load_moons(;num_train_samples=300, seed=22, num_validation_samples=500)
    # generate data
    samples, labels = py"make_moons"(n_samples=num_train_samples+num_validation_samples, noise=0.15, random_state=seed)

    # convert to a list of rows instead of a matrix
    row(m, i) = @view m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    samples = to_rows(samples)

    # scale features and replace labels
    samples, labels = py"process_dataset"(samples, labels)
    samples = to_rows(samples)

    # separate training and validation data points
    training_pair, validation_pair = separate_training_and_validation_sets(samples, labels, num_train_samples)

    num_positive(l) = count(==(1), l)
    num_negative(l) = count(==(-1), l)

    # save loaded data to variables
    global moons_dataset
    moons_dataset = Dataset(training_pair[1],
                            training_pair[2],
                            validation_pair[1],
                            validation_pair[2],
                            [0, 1],
                            ["Moon 1", "Moon 2"],
                            2,
                            length(training_pair[1]),
                            length(validation_pair[1]),
                            num_positive(training_pair[2]),
                            num_negative(training_pair[2]),
                            num_positive(validation_pair[2]),
                            num_negative(validation_pair[2]),
                            "moons")
    
    nothing
end


# same for iris data
if !isdefined(Main, :iris_dataset)
    iris_dataset = nothing
end
"Loads and processes iris data set."
function load_iris(;num_train_samples=60, target_dimensionality=4)
    # load data
    dataset_dict = py"load_iris()"

    # get samples and labels
    samples, labels = dataset_dict["data"], dataset_dict["target"]

    # convert to a list of rows instead of a matrix
    row(m, i) = @view m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    samples = to_rows(samples)

    # dimensionality reduction
    #@assert target_dimensionality <= 4 # 4 is the max target dimensionality, since the original dataset has 4 features
    #pca = py"PCA"(n_components=target_dimensionality)
    #pca.fit(samples)
    #samples = to_rows(pca.transform(samples))

    # scale features, replace labels
    samples, labels, chosen_classes = py"process_dataset"(samples, labels)
    # convert python output to rows again
    samples = to_rows(samples)

    # record names of selected classes
    all_class_names = dataset_dict["target_names"]
    class_names = [all_class_names[chosen_classes[1]+1], all_class_names[chosen_classes[2]+1]]
 
    # ensure training/validation split size is valid
    if num_train_samples > length(samples)
        error("num_train_samples can be at most $length(samples), but should be lower to leave validation data.")
    end

    # separate training and validation points
    training_pair, validation_pair = separate_training_and_validation_sets(samples, labels, num_train_samples)

    num_positive(l) = count(==(1), l)
    num_negative(l) = count(==(-1), l)

    # save data to variables
    global iris_dataset
    iris_dataset = Dataset(training_pair[1],
                             training_pair[2],
                             validation_pair[1],
                             validation_pair[2],
                             chosen_classes,
                             class_names,
                             target_dimensionality,#length(samples[1]),#
                             length(training_pair[1]),
                             length(validation_pair[1]),
                             num_positive(training_pair[2]),
                             num_negative(training_pair[2]),
                             num_positive(validation_pair[2]),
                             num_negative(validation_pair[2]),
                             "iris")

    nothing
end


# same for digits data
if !isdefined(Main, :digits_dataset)
    digits_dataset = nothing
end
"Loads and processes iris data set."
function load_digits(;num_train_samples=200, target_dimensionality=10)
    # load data
    dataset_dict = py"load_digits()"

    # get samples and labels
    samples, labels = dataset_dict["data"], dataset_dict["target"]

    # convert to a list of rows instead of a matrix
    row(m, i) = @view m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    samples = to_rows(samples)

    # dimensionality reduction
    pca = py"PCA"(n_components=target_dimensionality)
    pca.fit(samples)
    samples = to_rows(pca.transform(samples))

    # scale features, replace labels
    samples, labels, chosen_classes = py"process_dataset"(samples, labels, class_override=[8,9]) #maybe try [5,6] instead
    samples = to_rows(samples)

     # record names of selected classes
    all_class_names = dataset_dict["target_names"]
    class_names = [all_class_names[chosen_classes[1]+1], all_class_names[chosen_classes[2]+1]]

    # ensure training/validation split size is valid
    if num_train_samples > length(samples)
        error("num_train_samples can be at most $length(samples), but should be lower to leave validation data.")
    end

    # separate training and validation points
    training_pair, validation_pair = separate_training_and_validation_sets(samples, labels, num_train_samples)

    num_positive(l) = count(==(1), l)
    num_negative(l) = count(==(-1), l)

    # save loaded data to variables
    global digits_dataset
    digits_dataset = Dataset(training_pair[1],
                             training_pair[2],
                             validation_pair[1],
                             validation_pair[2],
                             chosen_classes,
                             class_names,
                             target_dimensionality,
                             length(training_pair[1]),
                             length(validation_pair[1]),
                             num_positive(training_pair[2]),
                             num_negative(training_pair[2]),
                             num_positive(validation_pair[2]),
                             num_negative(validation_pair[2]),
                             "digits")

    nothing
end

# blobs, like moons
if !isdefined(Main, :blobs_dataset)
    blobs_dataset = nothing
end
"Generates and processes blobs training and validation sets."
function load_blobs(;num_train_samples=300, seed=22, num_validation_samples=500)
    # generate data
    samples, labels = py"make_blobs"(n_samples=num_train_samples+num_validation_samples,
                                     random_state=seed,
                                     centers=2, # 2 classes
                                     n_features=2) # 2 features

    # convert to a list of rows instead of a matrix
    row(m, i) = @view m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    samples = to_rows(samples)

    # scale features and replace labels
    samples, labels = py"process_dataset"(samples, labels)
    samples = to_rows(samples)

    # separate training and validation data points
    training_pair, validation_pair = separate_training_and_validation_sets(samples, labels, num_train_samples)

    num_positive(l) = count(==(1), l)
    num_negative(l) = count(==(-1), l)

    # save loaded data to variables
    global blobs_dataset
    blobs_dataset = Dataset(training_pair[1],
                            training_pair[2],
                            validation_pair[1],
                            validation_pair[2],
                            [0, 1],
                            ["Blob 1", "Blob 2"],
                            2,
                            length(training_pair[1]),
                            length(validation_pair[1]),
                            num_positive(training_pair[2]),
                            num_negative(training_pair[2]),
                            num_positive(validation_pair[2]),
                            num_negative(validation_pair[2]),
                            "blobs")
    
    nothing
end

# circles, like moons
if !isdefined(Main, :circles_dataset)
    circles_dataset = nothing
end
"Generates and processes circles training and validation sets."
function load_circles(;num_train_samples=300, seed=22, num_validation_samples=500)
    # generate data
    samples, labels = py"make_circles"(n_samples=num_train_samples+num_validation_samples,
                                       random_state=seed,
                                       noise=0.1,
                                       factor=0.6)

    # convert to a list of rows instead of a matrix
    row(m, i) = @view m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    samples = to_rows(samples)

    # scale features and replace labels
    samples, labels = py"process_dataset"(samples, labels)
    samples = to_rows(samples)

    # separate training and validation data points
    training_pair, validation_pair = separate_training_and_validation_sets(samples, labels, num_train_samples)

    num_positive(l) = count(==(1), l)
    num_negative(l) = count(==(-1), l)

    # save loaded data to variables
    global circles_dataset
    circles_dataset = Dataset(training_pair[1],
                            training_pair[2],
                            validation_pair[1],
                            validation_pair[2],
                            [0, 1],
                            ["Circle 1", "Circle 2"],
                            2,
                            length(training_pair[1]),
                            length(validation_pair[1]),
                            num_positive(training_pair[2]),
                            num_negative(training_pair[2]),
                            num_positive(validation_pair[2]),
                            num_negative(validation_pair[2]),
                            "circles")
    
    nothing
end

function shuffle(vector)
    len = length(vector)
    @inbounds for i in 1:(len-1)
        j = floor(Int64, rand()*(len-(i-1)))+1
        temp = vector[i]
        vector[i] = vector[j]
        vector[j] = temp
    end
    return vector
end

"Generates and process adhoc training data set, then saves it to a file."
function generate_adhoc(;num_train_samples=300, seed=22)
    # generate data
    n_samples = num_train_samples #don't add num_validation_samples since the validation data will reuse the training data
    n_positive = n_samples ÷ 2
    n_negative = n_samples - n_positive
    samples = [8 .* rand(2) .- 4 for i in 1:n_samples] # random floats from -4 to 4
    labels = shuffle(vcat([1 for i in 1:n_positive], [0 for i in 1:n_negative])) # a balanced number of samples per class

    # convert to a list of rows instead of a matrix
    row(m, i) = @view m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    #samples = to_rows(samples)

    # scale features and replace labels
    samples, labels = py"process_dataset"(samples, labels)
    samples = to_rows(samples)

    samples = convert(Vector{Vector{Float64}}, samples)

    # create training data and validation data pair (in adhoc dataset case, they are the same data)
    training_pair = (samples, labels)
    # save dataset to file
    jldsave("data/adhoc_dataset.jld2", compress=true; training_pair=training_pair)
end

# random adhoc dataset
if !isdefined(Main, :adhoc_dataset)
    adhoc_dataset = nothing
end
"Loads pre-generated adhoc dataset from a file."
function load_adhoc()
    training_pair = JLD2.load("data/adhoc_dataset.jld2")["training_pair"]
    validation_pair = training_pair
    #training_pair, validation_pair = separate_training_and_validation_sets(samples, labels, num_train_samples)

    num_positive(l) = count(==(1), l)
    num_negative(l) = count(==(-1), l)

    # save loaded data to variables
    global adhoc_dataset
    adhoc_dataset = Dataset(training_pair[1],
                            training_pair[2],
                            validation_pair[1],
                            validation_pair[2],
                            [0, 1],
                            ["Class 1", "Class 2"],
                            2,
                            length(training_pair[1]),
                            length(validation_pair[1]),
                            num_positive(training_pair[2]),
                            num_negative(training_pair[2]),
                            num_positive(validation_pair[2]),
                            num_negative(validation_pair[2]),
                            "adhoc")
    
    nothing
end

if !isdefined(Main, :susy_dataset)
    susy_dataset = nothing
    susy_hard_dataset = nothing
end
"Loads both susy and susy_hard datasets."
function load_susy_and_susy_hard(;num_train_samples=300, target_dimensionality=10)
    # converts a matrix to a list of rows without copying the rows
    row(m, i) = @view m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    
    # for parsing dataframe entries to ints
    dfparse(x::Int) = Int64(x)
    dfparse(x::Float64) = x
    dfparse(x::AbstractString) = parse(Float64, String([c==',' ? '.' : c for c in x]))
    dfparse(x::Missing) = 0

    df = CSV.read("./datasets/SUSY.csv", DataFrame, footerskip=4999499) #skip all but the first 500 rows
    unparsed_matrix = Matrix(df)
    unparsed = to_rows(unparsed_matrix) # only load the first 500 samples to save time (150 will be used in training, 350 in validation)
    data_rows = [[dfparse(x) for x in row] for row in unparsed]

    # extract labels and 18-feature samples
    samples = [row[2:end] for row in data_rows]
    labels = [row[1] for row in data_rows]

    # extract 8-feature samples from 18-feature samples, the labels are the same as the original though
    samples_reduced = [sample[1:8] for sample in samples]

    # dimensionality reduction of 18-feature dataset
    pca = py"PCA"(n_components=target_dimensionality)
    pca.fit(samples)
    samples = to_rows(pca.transform(samples))

    # dimensionality reduction of 8-feature dataset
    #pca = py"PCA"(n_components=target_dimensionality)
    #pca.fit(samples_reduced)
    #samples_reduced = to_rows(pca.transform(samples_reduced))

    # processing 18-feature dataset
    samples, labels, classes = py"process_dataset"(samples, labels)
    samples = to_rows(samples)
    training_pair, validation_pair = separate_training_and_validation_sets(samples, labels, num_train_samples)

    # processing 8-feature dataset
    samples_reduced, labels_reduced, classes_reduced = py"process_dataset"(samples_reduced, labels) #note: it is correct to use original labels in this function argument
    samples_reduced = to_rows(samples_reduced)
    training_pair_reduced, validation_pair_reduced = separate_training_and_validation_sets(samples_reduced, labels_reduced, num_train_samples)

    num_positive(l) = count(==(1), l)
    num_negative(l) = count(==(-1), l)

    # creating 18-feature dataset instance
    global susy_dataset
    susy_dataset = Dataset(training_pair[1],
                            training_pair[2],
                            validation_pair[1],
                            validation_pair[2],
                            [0, 1],
                            ["Background", "Signal"],
                            target_dimensionality,
                            length(training_pair[1]),
                            length(validation_pair[1]),
                            num_positive(training_pair[2]),
                            num_negative(training_pair[2]),
                            num_positive(validation_pair[2]),
                            num_negative(validation_pair[2]),
                            "susy")

    # creating 8-feature dataset instance
    global susy_hard_dataset
    susy_hard_dataset = Dataset(training_pair_reduced[1],
                                training_pair_reduced[2],
                                validation_pair_reduced[1],
                                validation_pair_reduced[2],
                                [0, 1],
                                ["Background", "Signal"],
                                8,#target_dimensionality,
                                length(training_pair_reduced[1]),
                                length(validation_pair_reduced[1]),
                                num_positive(training_pair_reduced[2]),
                                num_negative(training_pair_reduced[2]),
                                num_positive(validation_pair_reduced[2]),
                                num_negative(validation_pair_reduced[2]),
                                "susy_hard")
end

# random adhoc dataset
if !isdefined(Main, :voice_dataset)
    voice_dataset = nothing
end
function load_voice(;num_train_samples=40, target_dimensionality=10)
    
    # converts a matrix to a list of rows without copying the rows
    row(m, i) = @view m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    
    # for parsing dataframe entries to ints
    dfparse(x::Int) = Int64(x)
    dfparse(x::AbstractString) = parse(Float64, String([c==',' ? '.' : c for c in x]))
    dfparse(x::Missing) = 0

    features_df = CSV.read("./datasets/LSVT_voice_rehabilitation_features.csv", DataFrame)
    samples_unparsed_matrix = Matrix(features_df)
    samples_unparsed = to_rows(samples_unparsed_matrix)
    samples = [[dfparse(x) for x in row] for row in samples_unparsed]
    
    labels_df = CSV.read("./datasets/LSVT_voice_rehabilitation_labels.csv", DataFrame)
    labels_unparsed_matrix = Matrix(labels_df)
    labels_unparsed = to_rows(labels_unparsed_matrix)
    labels = [dfparse(row[1]) == 1 ? -1 : 1 for row in labels_unparsed]

    # dimensionality reduction
    pca = py"PCA"(n_components=target_dimensionality)
    pca.fit(samples)
    samples = to_rows(pca.transform(samples))

    samples, labels, classes = py"process_dataset"(samples, labels)
    samples = to_rows(samples)
    training_pair, validation_pair = separate_training_and_validation_sets(samples, labels, num_train_samples)

    num_positive(l) = count(==(1), l)
    num_negative(l) = count(==(-1), l)

    global voice_dataset
    voice_dataset = Dataset(training_pair[1],
                            training_pair[2],
                            validation_pair[1],
                            validation_pair[2],
                            [0, 1],
                            ["Acceptable", "Unacceptable"],
                            target_dimensionality,#length(samples[1]),#
                            length(training_pair[1]),
                            length(validation_pair[1]),
                            num_positive(training_pair[2]),
                            num_negative(training_pair[2]),
                            num_positive(validation_pair[2]),
                            num_negative(validation_pair[2]),
                            "voice")
end

function load_all_datasets()
    println("Loading moons")
    load_moons()
    println("Loading iris")
    load_iris()
    println("Loading cancer")
    load_cancer()
    println("Loading digits")
    load_digits()
    println("Loading blobs")
    load_blobs()
    println("Loading circles")
    load_circles()
    println("Loading adhoc")
    load_adhoc()
    #println("Loading voice")
    #load_voice()
    #println("Loading susy_and_susy_hard")
    #load_susy_and_susy_hard()
end

load_all_datasets()

#allows retrieving a dataset directly from the name
dataset_map = Dict("moons"=>moons_dataset,
                   "digits"=>digits_dataset,
                   "cancer"=>cancer_dataset,
                   "iris"=>iris_dataset,
                   "blobs"=>blobs_dataset,
                   "circles"=>circles_dataset,
                   "adhoc"=>adhoc_dataset,
                   #"voice"=>voice_dataset,
                   #"susy"=>susy_dataset,
                   #"susy_hard"=>susy_hard_dataset
                   )
