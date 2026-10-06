import numpy as np
import pandas as pd


# Load dataset
data = pd.read_csv("data/test_dataset.csv")

X = data.drop("target", axis=1)
y = data["target"].values


def mutual_information(x, y, bins=10):

    x_bins = pd.cut(
        x,
        bins=bins,
        labels=False,
        duplicates="drop"
    )

    y_bins = pd.cut(
        y,
        bins=2,
        labels=False
    )

    mi = 0.0

    for x_value in np.unique(x_bins):
        for y_value in np.unique(y_bins):

            joint = np.sum(
                (x_bins == x_value) &
                (y_bins == y_value)
            )

            if joint == 0:
                continue

            p_xy = joint / len(x)
            p_x = np.sum(x_bins == x_value) / len(x)
            p_y = np.sum(y_bins == y_value) / len(y)

            mi += p_xy * np.log2(
                p_xy / (p_x * p_y)
            )

    return mi


# Relevance scores
relevance = {}

for feature in X.columns:
    relevance[feature] = mutual_information(
        X[feature],
        y
    )


# Redundancy scores
redundancy = {}

columns = list(X.columns)

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):

        feature1 = columns[i]
        feature2 = columns[j]

        score = mutual_information(
            X[feature1],
            X[feature2]
        )

        redundancy[(feature1, feature2)] = score
        redundancy[(feature2, feature1)] = score


# Select 5 features using mRMR
selected = []
remaining = columns.copy()

while len(selected) < 5:

    best_feature = None
    best_score = -np.inf

    for feature in remaining:

        if len(selected) == 0:

            score = relevance[feature]

        else:

            redundancy_score = np.mean([
                redundancy[(feature, selected_feature)]
                for selected_feature in selected
            ])

            score = relevance[feature] - redundancy_score

        if score > best_score:
            best_score = score
            best_feature = feature

    selected.append(best_feature)
    remaining.remove(best_feature)


# Select data
X_selected = X[selected].values


# Train-test split
np.random.seed(42)

indices = np.random.permutation(len(X_selected))

train_size = int(0.8 * len(X_selected))

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X_selected[train_indices]
y_train = y[train_indices]

X_test = X_selected[test_indices]
y_test = y[test_indices]


# Standardize
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / (std + 1e-8)
X_test = (X_test - mean) / (std + 1e-8)


# Add bias
X_train = np.c_[np.ones(len(X_train)), X_train]
X_test = np.c_[np.ones(len(X_test)), X_test]


# Logistic regression
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


# Test
z = np.clip(X_test @ weights, -500, 500)

probabilities = 1 / (1 + np.exp(-z))

predictions = (probabilities >= 0.5).astype(int)

accuracy = np.mean(predictions == y_test)


print("mRMR Experiment")
print("----------------")

print("Selected features:")

for feature in selected:
    print(feature)

print()
print("Number of features:", len(selected))
print("Test accuracy:", accuracy)