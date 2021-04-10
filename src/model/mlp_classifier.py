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


def splitting_data(X_data, y_data, testing_size=0.2, random_state=42):
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
        X_data, y_data, stratify=y_data, random_state=random_state, test_size=testing_size
    )
    return X_train, X_test, y_train, y_test


def prepare_model(
    X_train,
    y_train,
    hidden_layer_sizes=(24, 20, 12),
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


def who_wins_the_upcoming_fight(
    favorite_name, underdog_name, stats_data, outcome, names_data
):
    """Predicts an upcoming UFC fight.

    Args:
        favorite_name (str): Name of the Favorite-Fighter.
        underdog_name (str): Name of the Underdog-Fighter.
        stats_data (pd.DataFrame): Standardized data of stats.
        outcome (pd.Series): Historical outcomes.
        names_data (pd.DataFrame): Whole data that includes names.
    """
    clf_all = prepare_model(stats_data, outcome)

    fighter_names = names_data.iloc[:, :2].reset_index(drop=True)
    df_standardized = pd.DataFrame(stats_data)

    pred_df = pd.concat([fighter_names, df_standardized], axis=1)

    if favorite_name not in pred_df["Fighter"].to_list():
        raise IndexError(f"{favorite_name} was dropped because of nan values!")
    if underdog_name not in pred_df["Opponent"].to_list():
        raise IndexError(f"{underdog_name} was dropped because of nan values!")

    favorite = pred_df[pred_df["Fighter"] == favorite_name]
    underdog = pred_df[pred_df["Opponent"] == underdog_name]
    assert len(favorite.columns) == len(underdog.columns)

    stats_bound = int(len(favorite.iloc[:, 2:].columns) / 2) + 2
    favorite_stats = favorite.iloc[0, 2:stats_bound]
    underdog_stats = underdog.iloc[0, stats_bound:]
    assert len(favorite_stats) == len(underdog_stats) == 16

    upcoming_fight = (
        pd.concat([favorite_stats, underdog_stats]).to_numpy().reshape(1, -1)
    )
    outcome = clf_all.predict(upcoming_fight)[0]

    if outcome == 1:
        print(f"The favorite '{favorite_name}' will win the fight!")
    elif outcome == 0:
        print(f"The underdog '{underdog_name}' will win the fight!")
    else:
        print("Draw decision!")
