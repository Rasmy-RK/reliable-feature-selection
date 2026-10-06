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
                (x_bins == xv) &
                (y_bins == yv)
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


def generate_data(seed, total_features):

    np.random.seed(seed)

    samples = 1000

    # 5 truly informative features
    informative = np.random.normal(
        0,
        1,
        (samples, 5)
    )

    # Create target
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

    # Create redundant features
    redundant_count = max(
        5,
        int(total_features * 0.15)
    )

    redundant = []

    # Group information
    feature_groups = {}

    for i in range(5):

        feature_groups[
            "informative_" + str(i + 1)
        ] = i

    for i in range(redundant_count):

        source_index = i % 5

        source = informative[
            :,
            source_index
        ]

        redundant_feature = (
            source
            + np.random.normal(
                0,
                0.2,
                samples
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

    # Noise features
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

    # Combine
    X = np.hstack([
        informative,
        redundant,
        noise
    ])

    # Column names
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

    return (
        data,
        target,
        feature_groups
    )


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

                redundancy_score = np.mean(
                    redundancy_scores
                )

                score = (
                    relevance_score
                    - 0.1 * redundancy_score
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


def calculate_metrics(
    selected,
    feature_groups,
    total_features
):

    selected = set(selected)

    # ------------------------------------------------
    # 1. Exact informative feature recovery
    # ------------------------------------------------

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

    # ------------------------------------------------
    # 2. Information-group recovery
    # ------------------------------------------------

    recovered_groups = set()

    for feature in selected:

        if feature in feature_groups:

            recovered_groups.add(
                feature_groups[feature]
            )

    group_recovery = len(
        recovered_groups
    )

    # ------------------------------------------------
    # 3. Noise selections
    # ------------------------------------------------

    noise_selections = 0

    for feature in selected:

        if feature.startswith("noise_"):

            noise_selections += 1

    noise_rate = (
        noise_selections / len(selected)
    )

    return (
        exact_recovery,
        group_recovery,
        noise_selections,
        noise_rate
    )


# Dimensions
dimensions = [
    50,
    100,
    200,
    500
]


print(
    "High-Dimensional Feature Selection"
)

print(
    "----------------------------------"
)


for total_features in dimensions:

    mi_exact = []
    mi_groups = []
    mi_noise = []
    mi_noise_rate = []

    mrmr_exact = []
    mrmr_groups = []
    mrmr_noise = []
    mrmr_noise_rate = []

    redundant_count = max(
        5,
        int(total_features * 0.15)
    )

    for seed in [1, 2, 3]:

        X, y, feature_groups = (
            generate_data(
                seed,
                total_features
            )
        )

        # MI
        mi_features = select_mi(
            X,
            y
        )

        # mRMR-style
        mrmr_features = select_mrmr(
            X,
            y
        )

        # MI metrics
        (
            exact,
            groups,
            noise,
            noise_rate
        ) = calculate_metrics(
            mi_features,
            feature_groups,
            total_features
        )

        mi_exact.append(exact)
        mi_groups.append(groups)
        mi_noise.append(noise)
        mi_noise_rate.append(noise_rate)

        # mRMR metrics
        (
            exact,
            groups,
            noise,
            noise_rate
        ) = calculate_metrics(
            mrmr_features,
            feature_groups,
            total_features
        )

        mrmr_exact.append(exact)
        mrmr_groups.append(groups)
        mrmr_noise.append(noise)
        mrmr_noise_rate.append(noise_rate)

    print()

    print(
        "Total features:",
        total_features
    )

    print(
        "Redundant features:",
        redundant_count
    )

    print()

    print("MI:")

    print(
        "  Exact feature recovery:",
        round(
            np.mean(mi_exact),
            2
        ),
        "/ 5"
    )

    print(
        "  Information-group recovery:",
        round(
            np.mean(mi_groups),
            2
        ),
        "/ 5"
    )

    print(
        "  Average noise selections:",
        round(
            np.mean(mi_noise),
            2
        )
    )

    print(
        "  Noise selection rate:",
        round(
            np.mean(mi_noise_rate),
            2
        )
    )

    print()

    print("mRMR-style:")

    print(
        "  Exact feature recovery:",
        round(
            np.mean(mrmr_exact),
            2
        ),
        "/ 5"
    )

    print(
        "  Information-group recovery:",
        round(
            np.mean(mrmr_groups),
            2
        ),
        "/ 5"
    )

    print(
        "  Average noise selections:",
        round(
            np.mean(mrmr_noise),
            2
        )
    )

    print(
        "  Noise selection rate:",
        round(
            np.mean(mrmr_noise_rate),
            2
        )
    )