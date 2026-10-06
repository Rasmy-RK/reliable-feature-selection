# Results and Discussion

## 1. Experimental Setup

The feature-selection methods were evaluated using controlled synthetic datasets
with 5 informative features, redundant features, and noise features.

The total dimensionality was varied across:

- 50 features
- 100 features
- 200 features
- 500 features

For each dimensionality, experiments were repeated using three random seeds.

Three feature-selection approaches were compared:

1. Mutual Information (MI)
2. mRMR-style feature selection
3. L1 regularization

The evaluation metrics were:

- Exact feature recovery
- Information-group recovery
- Noise feature selections
- Number of selected features

An information group consists of an original informative feature and its
redundant copies. Therefore, group recovery measures whether the method
recovers the underlying information even when it does not select the exact
original feature.

---

## 2. Exact Feature Recovery

Exact feature recovery measures how many of the five original informative
features were selected.

The results show that L1 regularization consistently recovered all five
informative features across all tested dimensionalities.

MI recovered fewer exact informative features as dimensionality increased.
Its average exact recovery decreased from 2.0 out of 5 at 50 features to
1.0 out of 5 at 200 and 500 features.

The mRMR-style method showed a similar decline. Its exact recovery decreased
from 2.33 out of 5 at 50 features to 0.67 out of 5 at 500 features.

These results indicate that increasing dimensionality makes exact recovery
more difficult for the relevance-based methods.

---

## 3. Information-Group Recovery

Information-group recovery provides a more robust measure because redundant
features may contain almost the same information as the original informative
feature.

The L1 method achieved complete recovery of all five information groups at
every tested dimensionality.

MI achieved:

- 2.00/5 at 50 features
- 2.00/5 at 100 features
- 1.00/5 at 200 features
- 1.00/5 at 500 features

The mRMR-style method achieved:

- 3.00/5 at 50 features
- 2.33/5 at 100 features
- 2.00/5 at 200 features
- 1.33/5 at 500 features

The mRMR-style method generally recovered more information groups than MI,
suggesting that considering redundancy can improve feature selection compared
with relevance alone.

However, both methods experienced a reduction in recovery as dimensionality
increased.

In contrast, L1 maintained 5/5 group recovery throughout the tested range.

---

## 4. Noise Feature Selection

All three methods selected zero noise features in the final comparison.

MI and mRMR-style methods selected exactly five features at every
dimensionality, while L1 selected a larger subset.

Therefore, the absence of noise selections should be interpreted together
with the number of selected features and recovery metrics rather than as
evidence that all methods perform equally well.

---

## 5. Selected Feature Count

MI and mRMR-style selection were constrained to select exactly five features.

L1 regularization selected:

| Total Features | Average Selected Features |
|---:|---:|
| 50 | 12 |
| 100 | 20 |
| 200 | 35 |
| 500 | 74 |

The number of selected features increased as dimensionality increased.

This indicates that although L1 maintained complete recovery of the
informative groups, it required increasingly larger subsets to maintain that
recovery in the tested high-dimensional settings.

---

## 6. Discussion

The experiments demonstrate a clear difference between relevance-based
selection and sparse regularization in the controlled synthetic setting.

Mutual Information evaluates the relationship between individual features
and the target. However, it does not explicitly account for redundancy
between selected features. This can make it difficult to identify a
representative set of informative features when many correlated features
are present.

The mRMR-style approach incorporates a redundancy penalty in addition to
feature relevance. Its higher group recovery than MI at most tested
dimensionalities suggests that explicitly considering redundancy can help
recover diverse information.

However, its performance also decreases as the number of features increases.

L1 regularization produced the most stable recovery in these experiments.
It recovered all five informative features and all five information groups
at every tested dimensionality, while selecting no noise features.

At the same time, the number of selected features increased substantially
from 12 at 50 dimensions to 74 at 500 dimensions. Therefore, its strong
recovery came with a larger selected subset.

Overall, the results suggest that feature-selection reliability depends not
only on whether a method selects relevant variables, but also on how it
handles redundancy and increasing dimensionality.

---

## 7. Limitations

These experiments were conducted using controlled synthetic datasets.
Therefore, the results should not be interpreted as evidence that L1
regularization will always outperform MI or mRMR on real-world datasets.

The experiments also used three random seeds for each dimensionality.
Additional repetitions would provide a stronger estimate of variability.

The Mutual Information estimator used in this project is a simple
histogram-based estimator. More sophisticated estimators could produce
different results.

The mRMR implementation used in this project is described as mRMR-style
because it combines mutual-information relevance with a Pearson-correlation
redundancy penalty rather than implementing every detail of a canonical
mRMR formulation.

Future work could evaluate the methods on real datasets and investigate
how parameter choices affect recovery, stability, and predictive performance.