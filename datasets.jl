# This file is responsible for loading data sets and preprocessing them by selecting two classes,
# scaling the features, and splitting them into training and validation sets, with +-75 samples
# from each class being used for training and the remaining samples being used for validation of
# the genetic method.

#TODO: perform data set shuffling before doing the training / validation split.
# figure out how to set the shuffling seed for reproducible results

using PyCall

# ensure python dependencies (modules and user defined functions) are loaded
py"""
from sklearn.datasets import make_moons, load_iris, load_breast_cancer, load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def extract_binary_classes(feature_array, label_array):
    #Takes a numpy array of feature vectors and a numpy array of labels
    #and returns transformed numpy arrays with the number of classes reduced
    #to 2. Picks two random classes.
    # get 2 random classes based on the seed, and output what they are
    seed = 22 #use a fixed class-selection seed so classes don't change across different seed tests
    all_classes = list(set(label_array))
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

def process_dataset(feature_vectors, labels, binary_classification=True):
    # maybe extract classes for binary classification
    if binary_classification:
        feature_vectors, labels, classes = extract_binary_classes(feature_vectors, labels)
    else:
        classes = list(set(labels))

    # Now we standardize for gaussian around 0 with unit variance
    scaler = StandardScaler()
    scaler.fit(feature_vectors)
    feature_vectors = scaler.transform(feature_vectors)

    # Scale to the range (-1,+1)
    minmax_scaler = MinMaxScaler((-1, 1)).fit(feature_vectors)
    feature_vectors = minmax_scaler.transform(feature_vectors)

    # labels are either 1 or 0.
    # replace labels with +1 and -1 for kernel
    # target alignment computations to be valid.
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
    validation_samples::Vector{Vector{Float64}} = []
    validation_labels::Vector{Real} = []

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
                push!(validation_samples, sample)
                push!(validation_labels, label)
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
                push!(validation_samples, sample)
                push!(validation_labels, label)
            end
        else
            # if label is not recognized, error
            error("Found a label $label that was not equal to 1 or -1. Ensure that the labels been replaced before calling this function.")
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
function load_cancer(;num_train_samples=150, target_dimensionality=8)
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
function load_moons(;num_train_samples=150, seed=22, num_validation_samples=500)
    # generate data
    samples, labels = py"make_moons"(n_samples=num_train_samples+num_validation_samples, random_state=seed)

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
function load_iris(;num_train_samples=60)
    # load data
    dataset_dict = py"load_iris()"

    # get samples and labels
    samples, labels = dataset_dict["data"], dataset_dict["target"]

    # convert to a list of rows instead of a matrix
    row(m, i) = @view m[i, :]
    to_rows(matrix) = [row(matrix, i) for i in 1:size(matrix)[1]]
    samples = to_rows(samples)

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
                             4,
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
function load_digits(;num_train_samples=150, target_dimensionality=8)
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
    samples, labels, chosen_classes = py"process_dataset"(samples, labels)
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

function load_all_datasets()
    load_moons()
    load_iris()
    load_cancer()
    load_digits()
end

load_all_datasets()
