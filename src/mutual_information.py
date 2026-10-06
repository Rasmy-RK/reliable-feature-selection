import numpy as np
import pandas as pd


# Load dataset
data = pd.read_csv("data/test_dataset.csv")

X = data.drop("target", axis=1)
y = data["target"].values


# Calculate Mutual Information
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


# Calculate MI for every feature
scores = {}

for column in X.columns:
    scores[column] = mutual_information(X[column], y)


# Sort features by MI score
ranking = sorted(
    scores.items(),
    key=lambda item: item[1],
    reverse=True
)


print("Mutual Information Feature Ranking")
print("-----------------------------------")

for feature, score in ranking:
    print(feature, ":", round(score, 4))