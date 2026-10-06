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


def generate_data(seed, total_features):

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

    redundant = informative + np.random.normal(
        0, 0.2, (samples, 5)
    )

    noise_count = total_features - 10

    noise = np.random.normal(
        0, 1, (samples, noise_count)
    )

    X = np.hstack([
        informative,
        redundant,
        noise
    ])

    columns = []

    for i in range(5):
        columns.append("informative_" + str(i + 1))

    for i in range(5):
        columns.append("redundant_" + str(i + 1))

    for i in range(noise_count):
        columns.append("noise_" + str(i + 1))

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

    # Relevance
    relevance = {}

    for feature in X.columns:
        relevance[feature] = mutual_information(
            X[feature],
            y
        )

    selected = []
    remaining = list(X.columns)

    while len(selected) < k:

        best_feature = None
        best_score = -np.inf

        for feature in remaining:

            relevance_score = relevance[feature]

            if len(selected) == 0:

                score = relevance_score

            else:

                redundancy_scores = []

                for selected_feature in selected:

                    correlation = abs(
                        np.corrcoef(
                            X[feature],
                            X[selected_feature]
                        )[0, 1]
                    )

                    redundancy_scores.append(
                        correlation
                    )

                redundancy_score = np.mean(
                    redundancy_scores
                )

                score = (
                    relevance_score
                    - redundancy_score
                )

            if score > best_score:

                best_score = score
                best_feature = feature

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected


true_features = {
    "informative_1",
    "informative_2",
    "informative_3",
    "informative_4",
    "informative_5"
}


dimensions = [20, 50, 100, 200, 500]

print("Dimensionality Experiment")
print("-------------------------")

for total_features in dimensions:

    mi_results = []
    mrmr_results = []

    for seed in [1, 2, 3]:

        X, y = generate_data(
            seed,
            total_features
        )

        mi_features = select_mi(X, y)

        mrmr_features = select_mrmr(
            X,
            y
        )

        mi_correct = len(
            set(mi_features) & true_features
        )

        mrmr_correct = len(
            set(mrmr_features) & true_features
        )

        mi_results.append(mi_correct)
        mrmr_results.append(mrmr_correct)

    print()
    print("Total features:", total_features)

    print(
        "MI average recovery:",
        round(np.mean(mi_results), 2),
        "/ 5"
    )

    print(
        "mRMR-style average recovery:",
        round(np.mean(mrmr_results), 2),
        "/ 5"
    )