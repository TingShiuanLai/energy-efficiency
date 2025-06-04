import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("energy_efficiency_data.csv")
features = ['Relative_Compactness', 'Wall_Area', 'Glazing_Area', 'Glazing_Area_Distribution', 'Overall_Height']


# Step 1: Scatter plot of Heating vs Cooling
plt.figure(figsize=(8, 6))
plt.scatter(df['Heating_Load'], df['Cooling_Load'], alpha=0.6)
plt.xlabel("Heating Load")
plt.ylabel("Cooling Load")
plt.title("Trade-off Between Heating and Cooling Load")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/tradeoff_heating_vs_cooling.png")
plt.show()

# Step 2: Identify Pareto-efficient points
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True  # Keep self
    return is_efficient

costs = df[['Heating_Load', 'Cooling_Load']].values
pareto_mask = is_pareto_efficient(costs)

# Plot Pareto front
plt.figure(figsize=(8, 6))
plt.scatter(df['Heating_Load'], df['Cooling_Load'], alpha=0.3, label='All Designs')
plt.scatter(df['Heating_Load'][pareto_mask], df['Cooling_Load'][pareto_mask],
            color='red', label='Pareto-efficient', s=60)
plt.xlabel("Heating Load")
plt.ylabel("Cooling Load")
plt.title("Pareto Front - Energy Load Trade-offs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/pareto_front_heating_cooling.png")
plt.show()

# Add Pareto mask to dataframe
df['Pareto'] = pareto_mask

# Plot comparison for each feature
import seaborn as sns
import matplotlib.pyplot as plt

# Determine layout (2 columns, adjust rows automatically)
n_features = len(features)
n_cols = 2
n_rows = int(np.ceil(n_features / n_cols))

# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
axes = axes.flatten()  # Flatten in case it's 2D array

# Plot each feature
for i, feature in enumerate(features):
    sns.boxplot(x='Pareto', y=feature, data=df, ax=axes[i])
    axes[i].set_title(f"{feature} Distribution")
    axes[i].set_xlabel("Pareto Efficient (True = Red Dot)")
    axes[i].set_ylabel(feature)

# Hide any empty subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("plots/pareto_boxplot_all_features.png")
plt.show()

# Step 7: Summarize Best Feature Value Ranges for Trade-off
pareto_df = df[df['Pareto'] == True]
summary = pareto_df[features].describe().T[['min', '25%', '50%', '75%', 'max']]
summary = summary.rename(columns={
    'min': 'Min',
    '25%': 'Q1 (25%)',
    '50%': 'Median',
    '75%': 'Q3 (75%)',
    'max': 'Max'
})

# Round for clarity
summary = summary.round(3)

# Print summary to console
print("\nBest Feature Value Ranges (Pareto-efficient designs):\n")
print(summary)

# Save to CSV for documentation
summary.to_csv("pareto_feature_ranges.csv")

