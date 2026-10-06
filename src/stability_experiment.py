import numpy as np
import pandas as pd


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

    for xv in np.unique(x_bins):
        for yv in np.unique(y_bins):

            joint = np.sum(
                (x_bins == xv) & (y_bins == yv)
            )

            if joint == 0:
                continue

            p_xy = joint / len(x)
            p_x = np.sum(x_bins == xv) / len(x)
            p_y = np.sum(y_bins == yv) / len(y)

            mi += p_xy * np.log2(
                p_xy / (p_x * p_y)
            )

    return mi


def generate_data(seed):

    np.random.seed(seed)

    samples = 1000

    informative = np.random.normal(
        0, 1, (samples, 5)
    )

    score = (
        2 * informative[:, 0]
        + 1.5 * informative[:, 1]
        - 1.2 * informative[:, 2]
        + informative[:, 3]
        - 0.8 * informative[:, 4]
    )

    probability = 1 / (1 + np.exp(-score))

    target = (probability > 0.5).astype(int)

    noise = np.random.normal(
        0, 1, (samples, 10)
    )

    redundant = informative + np.random.normal(
        0, 0.2, (samples, 5)
    )

    X = np.hstack([
        informative,
        noise,
        redundant
    ])

    columns = []

    for i in range(5):
        columns.append("informative_" + str(i + 1))

    for i in range(10):
        columns.append("noise_" + str(i + 1))

    for i in range(5):
        columns.append("redundant_" + str(i + 1))

    return pd.DataFrame(X, columns=columns), target


def select_mi(X, y, k=5):

    scores = {}

    for feature in X.columns:
        scores[feature] = mutual_information(
            X[feature],
            y
        )

    ranking = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return ranking[:k]


def select_mrmr(X, y, k=5):

    relevance = {}

    for feature in X.columns:
        relevance[feature] = mutual_information(
            X[feature],
            y
        )

    redundancy = {}

    columns = list(X.columns)

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):

            f1 = columns[i]
            f2 = columns[j]

            score = mutual_information(
                X[f1],
                X[f2]
            )

            redundancy[(f1, f2)] = score
            redundancy[(f2, f1)] = score

    selected = []
    remaining = columns.copy()

    while len(selected) < k:

        best_feature = None
        best_score = -np.inf

        for feature in remaining:

            if len(selected) == 0:

                score = relevance[feature]

            else:

                redundancy_score = np.mean([
                    redundancy[
                        (feature, selected_feature)
                    ]
                    for selected_feature in selected
                ])

                score = (
                    relevance[feature]
                    - redundancy_score
                )

            if score > best_score:
                best_score = score
                best_feature = feature

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected


# True informative features
true_features = {
    "informative_1",
    "informative_2",
    "informative_3",
    "informative_4",
    "informative_5"
}


mi_recovery = []
mrmr_recovery = []

seeds = [1, 2, 3, 4, 5]

print("Feature Selection Stability Experiment")
print("--------------------------------------")

for seed in seeds:

    X, y = generate_data(seed)

    mi_features = select_mi(X, y)
    mrmr_features = select_mrmr(X, y)

    mi_correct = len(
        set(mi_features) & true_features
    )

    mrmr_correct = len(
        set(mrmr_features) & true_features
    )

    mi_recovery.append(mi_correct)
    mrmr_recovery.append(mrmr_correct)

    print()
    print("Seed:", seed)
    print("MI:", mi_features)
    print("MI correct:", mi_correct, "/ 5")

    print("mRMR:", mrmr_features)
    print("mRMR correct:", mrmr_correct, "/ 5")


print()
print("Average Feature Recovery")
print("-------------------------")

print(
    "MI:",
    round(np.mean(mi_recovery), 2),
    "/ 5"
)

print(
    "mRMR:",
    round(np.mean(mrmr_recovery), 2),
    "/ 5"
)