import pandas as pd
import matplotlib.pyplot as plt

# Load final comparison results
data = pd.read_csv("results/tables/final_comparison.csv")

# -----------------------------
# Plot 1: Information-group recovery
# -----------------------------

plt.figure(figsize=(8, 5))

for method in data["method"].unique():
    method_data = data[data["method"] == method]

    plt.plot(
        method_data["dimensions"],
        method_data["group_recovery"],
        marker="o",
        label=method
    )

plt.xlabel("Number of Features")
plt.ylabel("Information-Group Recovery")
plt.title("Feature Selection Reliability vs Dimensionality")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/figures/group_recovery_vs_dimensions.png",
    dpi=300
)

plt.close()


# -----------------------------
# Plot 2: Number of selected features
# -----------------------------

plt.figure(figsize=(8, 5))

for method in data["method"].unique():
    method_data = data[data["method"] == method]

    plt.plot(
        method_data["dimensions"],
        method_data["selected_features"],
        marker="o",
        label=method
    )

plt.xlabel("Number of Features")
plt.ylabel("Average Selected Features")
plt.title("Selected Feature Count vs Dimensionality")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/figures/selected_features_vs_dimensions.png",
    dpi=300
)

plt.close()


# -----------------------------
# Plot 3: Noise selections
# -----------------------------

plt.figure(figsize=(8, 5))

for method in data["method"].unique():
    method_data = data[data["method"] == method]

    plt.plot(
        method_data["dimensions"],
        method_data["noise_selections"],
        marker="o",
        label=method
    )

plt.xlabel("Number of Features")
plt.ylabel("Average Noise Selections")
plt.title("Noise Feature Selection vs Dimensionality")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/figures/noise_selections_vs_dimensions.png",
    dpi=300
)

plt.close()


print("Plots created successfully!")
print()
print("Saved to:")
print("results/figures/group_recovery_vs_dimensions.png")
print("results/figures/selected_features_vs_dimensions.png")
print("results/figures/noise_selections_vs_dimensions.png")\
# -----------------------------
# Plot 4: Exact feature recovery
# -----------------------------

plt.figure(figsize=(8, 5))

for method in data["method"].unique():
    method_data = data[data["method"] == method]

    plt.plot(
        method_data["dimensions"],
        method_data["exact_recovery"],
        marker="o",
        label=method
    )

plt.xlabel("Number of Features")
plt.ylabel("Exact Feature Recovery")
plt.title("Exact Informative Feature Recovery vs Dimensionality")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/figures/exact_recovery_vs_dimensions.png",
    dpi=300
)

plt.close()

print("results/figures/exact_recovery_vs_dimensions.png")