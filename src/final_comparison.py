import numpy as np
import pandas as pd


# ============================================================
# DATA GENERATION
# ============================================================

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

    probability = 1 / (
        1 + np.exp(-score)
    )

    target = (
        probability > 0.5
    ).astype(int)

    redundant_count = max(
        5,
        int(total_features * 0.15)
    )

    redundant = []

    feature_groups = {}

    for i in range(5):

        feature_groups[
            "informative_" + str(i + 1)
        ] = i

    for i in range(redundant_count):

        source_index = i % 5

        redundant_feature = (
            informative[:, source_index]
            + np.random.normal(
                0, 0.2, samples
            )
        )

        redundant.append(
            redundant_feature
        )

        feature_groups[
            "redundant_" + str(i + 1)
        ] = source_index

    redundant = np.column_stack(
        redundant
    )

    noise_count = (
        total_features
        - 5
        - redundant_count
    )

    noise = np.random.normal(
        0,
        1,
        (samples, noise_count)
    )

    X = np.hstack([
        informative,
        redundant,
        noise
    ])

    columns = []

    for i in range(5):
        columns.append(
            "informative_" + str(i + 1)
        )

    for i in range(redundant_count):
        columns.append(
            "redundant_" + str(i + 1)
        )

    for i in range(noise_count):
        columns.append(
            "noise_" + str(i + 1)
        )

    data = pd.DataFrame(
        X,
        columns=columns
    )

    return data, target, feature_groups


# ============================================================
# MUTUAL INFORMATION
# ============================================================

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
                (x_bins == xv)
                & (y_bins == yv)
            )

            if joint == 0:
                continue

            p_xy = joint / len(x)

            p_x = np.sum(
                x_bins == xv
            ) / len(x)

            p_y = np.sum(
                y_bins == yv
            ) / len(y)

            mi += p_xy * np.log2(
                p_xy / (p_x * p_y)
            )

    return mi


def select_mi(X, y, k=5):

    scores = {}

    for feature in X.columns:

        scores[feature] = (
            mutual_information(
                X[feature],
                y
            )
        )

    ranking = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return ranking[:k]


# ============================================================
# mRMR-STYLE
# ============================================================

def select_mrmr(X, y, k=5):

    relevance = {}

    for feature in X.columns:

        relevance[feature] = (
            mutual_information(
                X[feature],
                y
            )
        )

    selected = []

    remaining = list(X.columns)

    while len(selected) < k:

        best_feature = None
        best_score = -np.inf

        for feature in remaining:

            relevance_score = (
                relevance[feature]
            )

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

                redundancy = np.mean(
                    redundancy_scores
                )

                score = (
                    relevance_score
                    - 0.1 * redundancy
                )

            if score > best_score:

                best_score = score
                best_feature = feature

        selected.append(
            best_feature
        )

        remaining.remove(
            best_feature
        )

    return selected


# ============================================================
# L1 LOGISTIC REGRESSION
# ============================================================

def sigmoid(z):

    z = np.clip(
        z,
        -500,
        500
    )

    return 1 / (
        1 + np.exp(-z)
    )


def train_l1(
    X,
    y,
    regularization=0.05,
    learning_rate=0.01,
    iterations=1000
):

    n_samples, n_features = X.shape

    weights = np.zeros(
        n_features
    )

    bias = 0.0

    for _ in range(iterations):

        predictions = sigmoid(
            np.dot(X, weights)
            + bias
        )

        error = predictions - y

        gradient = (
            np.dot(X.T, error)
            / n_samples
        )

        gradient += (
            regularization
            * np.sign(weights)
        )

        weights -= (
            learning_rate
            * gradient
        )

        bias -= (
            learning_rate
            * np.mean(error)
        )

    return weights, bias


def select_l1(
    X,
    y,
    feature_names
):

    weights, bias = train_l1(
        X,
        y
    )

    selected = []

    threshold = 0.01

    for i in range(
        len(feature_names)
    ):

        if abs(
            weights[i]
        ) > threshold:

            selected.append(
                feature_names[i]
            )

    return selected


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    selected,
    feature_groups
):

    selected = set(selected)

    true_features = {
        "informative_1",
        "informative_2",
        "informative_3",
        "informative_4",
        "informative_5"
    }

    exact_recovery = len(
        selected & true_features
    )

    recovered_groups = set()

    for feature in selected:

        if feature in feature_groups:

            recovered_groups.add(
                feature_groups[feature]
            )

    group_recovery = len(
        recovered_groups
    )

    noise_selections = sum(
        feature.startswith("noise_")
        for feature in selected
    )

    return (
        exact_recovery,
        group_recovery,
        noise_selections
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

dimensions = [
    50,
    100,
    200,
    500
]

seeds = [
    1,
    2,
    3
]

results = []


print(
    "Final Method Comparison"
)

print(
    "-----------------------"
)


for total_features in dimensions:

    print()
    print(
        "Total features:",
        total_features
    )

    for method in [
        "MI",
        "mRMR-style",
        "L1"
    ]:

        group_values = []
        noise_values = []
        exact_values = []
        selected_values = []

        for seed in seeds:

            data, y, feature_groups = (
                generate_data(
                    seed,
                    total_features
                )
            )

            if method == "MI":

                selected = select_mi(
                    data,
                    y
                )

            elif method == "mRMR-style":

                selected = select_mrmr(
                    data,
                    y
                )

            else:

                X = data.values

                means = np.mean(
                    X,
                    axis=0
                )

                stds = np.std(
                    X,
                    axis=0
                )

                stds[stds == 0] = 1

                X = (
                    X - means
                ) / stds

                selected = select_l1(
                    X,
                    y,
                    list(data.columns)
                )

            (
                exact,
                groups,
                noise
            ) = calculate_metrics(
                selected,
                feature_groups
            )

            exact_values.append(
                exact
            )

            group_values.append(
                groups
            )

            noise_values.append(
                noise
            )

            selected_values.append(
                len(selected)
            )

        average_exact = np.mean(
            exact_values
        )

        average_groups = np.mean(
            group_values
        )

        average_noise = np.mean(
            noise_values
        )

        average_selected = np.mean(
            selected_values
        )

        results.append({
            "dimensions": total_features,
            "method": method,
            "exact_recovery": round(
                average_exact,
                2
            ),
            "group_recovery": round(
                average_groups,
                2
            ),
            "noise_selections": round(
                average_noise,
                2
            ),
            "selected_features": round(
                average_selected,
                2
            )
        })

        print(
            method,
            "| groups:",
            round(average_groups, 2),
            "| noise:",
            round(average_noise, 2),
            "| selected:",
            round(average_selected, 2)
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "results/tables/final_comparison.csv",
    index=False
)

print()
print(
    "Results saved to:"
)

print(
    "results/tables/final_comparison.csv"
)