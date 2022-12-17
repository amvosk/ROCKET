import numpy as np

def load_dataset(path_dataset, dataset_name):

    training_data = np.loadtxt(f"{path_dataset}/{dataset_name}/{dataset_name}_TRAIN.txt")
    Y_training_, X_training = training_data[:, 0].astype(np.int32), training_data[:, 1:]

    test_data = np.loadtxt(f"{path_dataset}/{dataset_name}/{dataset_name}_TEST.txt")
    Y_test_, X_test = test_data[:, 0].astype(np.int32), test_data[:, 1:]

    classes_old = np.unique(Y_training_)
    n_classes = classes_old.shape[0]
    
    Y_training = np.zeros(Y_training_.shape[0]).astype(np.int32)
    Y_test = np.zeros(Y_test_.shape[0]).astype(np.int32)
    for i, c in enumerate(classes_old):
        Y_training[Y_training_ == c] = i
        Y_test[Y_test_ == c] = i

    return X_training, Y_training, X_test, Y_test, n_classes, classes_old