import numpy as np
import pandas as pd


# Load dataset
data = pd.read_csv("data/test_dataset.csv")

# Separate features and target
X = data.drop("target", axis=1).values
y = data["target"].values

# Train-test split
np.random.seed(42)

indices = np.random.permutation(len(X))

train_size = int(0.8 * len(X))

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]


# Standardize features
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / (std + 1e-8)
X_test = (X_test - mean) / (std + 1e-8)


# Add bias column
X_train = np.c_[np.ones(len(X_train)), X_train]
X_test = np.c_[np.ones(len(X_test)), X_test]


# Logistic regression
weights = np.zeros(X_train.shape[1])

learning_rate = 0.01
epochs = 1000


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


for epoch in range(epochs):

    predictions = sigmoid(X_train @ weights)

    gradient = (X_train.T @ (predictions - y_train)) / len(X_train)

    weights = weights - learning_rate * gradient


# Test prediction
probabilities = sigmoid(X_test @ weights)

predictions = (probabilities >= 0.5).astype(int)


# Accuracy
accuracy = np.mean(predictions == y_test)


print("Baseline Model")
print("----------------")
print("Number of features:", X.shape[1])
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Test accuracy:", accuracy)