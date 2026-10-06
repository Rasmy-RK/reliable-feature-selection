import numpy as np
import pandas as pd


# --------------------------------------------------
# Generate controlled dataset
# --------------------------------------------------

def generate_data(seed, total_features):

    np.random.seed(seed)

    samples = 1000

    # 5 informative features
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

    target = (
        probability > 0.5
    ).astype(int)

    # Redundant features
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

    # Noise
    noise_count = (
        total_features
        - 5
        - redundant_count
    )

    noise = np.random.normal(
        0, 1,
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


# --------------------------------------------------
# Sigmoid
# --------------------------------------------------

def sigmoid(z):

    z = np.clip(
        z,
        -500,
        500
    )

    return 1 / (1 + np.exp(-z))


# --------------------------------------------------
# L1 Logistic Regression
# --------------------------------------------------

def l1_logistic_regression(
    X,
    y,
    learning_rate=0.01,
    regularization=0.01,
    iterations=1000
):

    X = np.asarray(X)

    y = np.asarray(y)

    n_samples, n_features = X.shape

    weights = np.zeros(
        n_features
    )

    bias = 0.0

    for _ in range(iterations):

        predictions = sigmoid(
            np.dot(X, weights) + bias
        )

        error = predictions - y

        gradient = (
            np.dot(X.T, error)
            / n_samples
        )

        # L1 penalty
        l1_gradient = (
            regularization
            * np.sign(weights)
        )

        gradient += l1_gradient

        weights -= (
            learning_rate
            * gradient
        )

        bias -= (
            learning_rate
            * np.mean(error)
        )

    return weights, bias


# --------------------------------------------------
# Feature selection
# --------------------------------------------------

def select_features(
    X,
    y,
    feature_names
):

    weights, bias = (
        l1_logistic_regression(
            X,
            y
        )
    )

    # Select non-zero / sufficiently large
    # coefficients
    threshold = 0.01

    selected = []

    for i in range(
        len(feature_names)
    ):

        if abs(weights[i]) > threshold:

            selected.append(
                feature_names[i]
            )

    return selected, weights


# --------------------------------------------------
# Metrics
# --------------------------------------------------

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

    # Exact recovery
    exact_recovery = len(
        selected & true_features
    )

    # Information-group recovery
    recovered_groups = set()

    for feature in selected:

        if feature in feature_groups:

            recovered_groups.add(
                feature_groups[feature]
            )

    group_recovery = len(
        recovered_groups
    )

    # Noise selections
    noise_selections = 0

    for feature in selected:

        if feature.startswith(
            "noise_"
        ):

            noise_selections += 1

    if len(selected) > 0:

        noise_rate = (
            noise_selections
            / len(selected)
        )

    else:

        noise_rate = 0

    return (
        exact_recovery,
        group_recovery,
        noise_selections,
        noise_rate
    )


# --------------------------------------------------
# Experiment
# --------------------------------------------------

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


print(
    "L1 Feature Selection Experiment"
)

print(
    "-------------------------------"
)


for total_features in dimensions:

    exact_results = []

    group_results = []

    noise_results = []

    noise_rate_results = []

    for seed in seeds:

        data, y, feature_groups = (
            generate_data(
                seed,
                total_features
            )
        )

        X = data.values

        feature_names = (
            list(data.columns)
        )

        # Standardize features
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

        selected, weights = (
            select_features(
                X,
                y,
                feature_names
            )
        )

        (
            exact,
            groups,
            noise,
            noise_rate
        ) = calculate_metrics(
            selected,
            feature_groups
        )

        exact_results.append(
            exact
        )

        group_results.append(
            groups
        )

        noise_results.append(
            noise
        )

        noise_rate_results.append(
            noise_rate
        )

    print()

    print(
        "Total features:",
        total_features
    )

    print(
        "Average number of selected features:",
        round(
            np.mean([
                len(
                    select_features(
                        (
                            generate_data(
                                seed,
                                total_features
                            )[0]
                        ).values,
                        generate_data(
                            seed,
                            total_features
                        )[1],
                        list(
                            generate_data(
                                seed,
                                total_features
                            )[0].columns
                        )
                    )[0]
                )
                for seed in seeds
            ]),
            2
        )
    )

    print(
        "Exact feature recovery:",
        round(
            np.mean(exact_results),
            2
        ),
        "/ 5"
    )

    print(
        "Information-group recovery:",
        round(
            np.mean(group_results),
            2
        ),
        "/ 5"
    )

    print(
        "Average noise selections:",
        round(
            np.mean(noise_results),
            2
        )
    )

    print(
        "Noise selection rate:",
        round(
            np.mean(noise_rate_results),
            2
        )
    )