import numpy as np
import pandas as pd


# Load dataset
data = pd.read_csv("data/test_dataset.csv")

X = data.drop("target", axis=1)
y = data["target"].values


# Mutual Information
def mutual_information(feature, target, bins=10):

    feature_bins = pd.cut(
        feature,
        bins=bins,
        labels=False,
        duplicates="drop"
    )

    mi = 0.0

    for feature_value in np.unique(feature_bins):

        for target_value in np.unique(target):

            joint = np.sum(
                (feature_bins == feature_value) &
                (target == target_value)
            )

            if joint == 0:
                continue

            p_xy = joint / len(feature)
            p_x = np.sum(feature_bins == feature_value) / len(feature)
            p_y = np.sum(target == target_value) / len(target)

            mi += p_xy * np.log2(p_xy / (p_x * p_y))

    return mi


# Calculate MI scores
scores = {}

for column in X.columns:
    scores[column] = mutual_information(X[column], y)


# Rank features
ranking = sorted(
    scores.items(),
    key=lambda item: item[1],
    reverse=True
)


# Train logistic regression using NumPy
def train_model(X, y):

    np.random.seed(42)

    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    X_train = np.c_[np.ones(len(X_train)), X_train]
    X_test = np.c_[np.ones(len(X_test)), X_test]

    weights = np.zeros(X_train.shape[1])

    learning_rate = 0.01
    epochs = 1000

    for epoch in range(epochs):

        z = np.clip(X_train @ weights, -500, 500)
        predictions = 1 / (1 + np.exp(-z))

        gradient = (
            X_train.T @ (predictions - y_train)
        ) / len(X_train)

        weights = weights - learning_rate * gradient

    z = np.clip(X_test @ weights, -500, 500)

    probabilities = 1 / (1 + np.exp(-z))

    predictions = (probabilities >= 0.5).astype(int)

    accuracy = np.mean(predictions == y_test)

    return accuracy


# Test different numbers of selected features
print("Mutual Information Experiment")
print("-----------------------------")

for number in [5, 10, 15]:

    selected_features = [
        feature for feature, score in ranking[:number]
    ]

    X_selected = X[selected_features].values

    accuracy = train_model(X_selected, y)

    print(
        "Top",
        number,
        "features:",
        round(accuracy, 4)
    )