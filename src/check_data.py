import pandas as pd

data = pd.read_csv("data/test_dataset.csv")

print("Dataset shape:", data.shape)
print()

print("Correlation with target:")
print(data.corr(numeric_only=True)["target"].sort_values(ascending=False))

print()
print("Correlation between informative and redundant features:")

for i in range(1, 6):
    informative = "informative_" + str(i)
    redundant = "redundant_" + str(i)

    correlation = data[informative].corr(data[redundant])

    print(informative, "<->", redundant, ":", correlation)