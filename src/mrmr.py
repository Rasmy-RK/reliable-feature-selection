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


# Relevance: feature -> target
relevance = {}

for feature in X.columns:
    relevance[feature] = mutual_information(
        X[feature],
        y
    )


# Redundancy: feature -> feature
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


# mRMR-style selection
selected = []
remaining = columns.copy()

number_of_features = 5

while len(selected) < number_of_features:

    best_feature = None
    best_score = -np.inf

    for feature in remaining:

        relevance_score = relevance[feature]

        if len(selected) == 0:

            score = relevance_score

        else:

            redundancy_score = np.mean([
                redundancy[(feature, selected_feature)]
                for selected_feature in selected
            ])

            score = relevance_score - redundancy_score

        if score > best_score:
            best_score = score
            best_feature = feature

    selected.append(best_feature)
    remaining.remove(best_feature)


print("mRMR-Style Feature Selection")
print("----------------------------")

print("Selected features:")

for feature in selected:
    print(feature)

print()
print("Number of selected features:", len(selected))