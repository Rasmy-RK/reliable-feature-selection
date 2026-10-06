import numpy as np
import pandas as pd

np.random.seed(42)

samples = 1000

# 5 informative features
informative = np.random.normal(0, 1, (samples, 5))

# Create target from the informative features
score = (
    2 * informative[:, 0]
    + 1.5 * informative[:, 1]
    - 1.2 * informative[:, 2]
    + informative[:, 3]
    - 0.8 * informative[:, 4]
)

probability = 1 / (1 + np.exp(-score))

target = (probability > 0.5).astype(int)

# 10 irrelevant features
noise = np.random.normal(0, 1, (samples, 10))

# 5 redundant features
redundant = informative[:, :5] + np.random.normal(
    0, 0.2, (samples, 5)
)

# Combine everything
X = np.hstack([
    informative,
    noise,
    redundant
])

# Create column names
columns = []

for i in range(5):
    columns.append("informative_" + str(i + 1))

for i in range(10):
    columns.append("noise_" + str(i + 1))

for i in range(5):
    columns.append("redundant_" + str(i + 1))

data = pd.DataFrame(X, columns=columns)

data["target"] = target

# Save dataset
data.to_csv("data/test_dataset.csv", index=False)

print("Dataset created successfully!")
print("Shape:", data.shape)
print(data.head())