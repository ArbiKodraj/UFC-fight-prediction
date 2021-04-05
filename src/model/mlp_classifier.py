import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def standardization(data):
    """Standadizes a dataset.

    Args:
        data (pd.DataFrame): Data that shall be standarized.

    Returns:
        pd.DataFrame: Standarized data.
    """
    scaler = StandardScaler()
    scaler.fit(data)
    data_stand = scaler.transform(data)
    return data_stand


def splitting_data(X_data, y_data, testing_size=0.2, random_state=1):
    """Returns testing and training data.

    Args:
        X_data (pd.DataFrame): Stats of UFC Fighters.
        y_data (pd.DataFrame): Results.
        testing_size (float, optional): [description]. Defaults to 0.2.
        random_state (int, optional): [description]. Defaults to 1.

    Returns:
        np.array: training and testing data
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, stratify=y_data, random_state=1, test_size=0.2
    )
    return X_train, X_test, y_train, y_test


def prepare_model(
    X_train,
    y_train,
    hidden_layer_sizes=(28, 24, 18),
    activation="logistic",
    solver="adam",
    max_iter=10000,
):
    """Prepares neural network.

    Args:
        X_train (np.array): Training X data.
        y_train (np.array): Training y data.
        hidden_layer_sizes (tuple, optional): Layers and number of neurons.
            Defaults to (40, 34, 28).
        activation (str, optional): Activation function. Defaults to "logistic".
        solver (str, optional): Used Optimizer. Defaults to "adam".
        max_iter (int, optional): Number of iterations. Defaults to 5000.

    Returns:
        method: Neural Network Model.
    """
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        max_iter=max_iter,
    ).fit(X_train, y_train.to_numpy().ravel())
    return clf


def return_prediction(X_train, y_train, X_test, **kwargs):
    """Returns prediction of Neural Network.

    Args:
        X_train (np.array): Training data X - Stats of fighters.
        y_train (np.array): Training data y - Results of fights.
        X_test (np.array): Testing data X - Stats of fighters.

    Returns:
        np.array: Predicted results.
    """
    model = prepare_model(X_train, y_train, **kwargs)
    return model.predict(X_test)
