
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats

# Ensure output directory exists
os.makedirs("plots", exist_ok=True)

# Load dataset
df = pd.read_csv('energy_efficiency_data.csv')

# Basic info
print("Dataset Info:")
print(df.info())
print("\nDescriptive Stats:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

# Pairplot
sns.pairplot(df)
plt.suptitle("Pairplot of Features", y=1.02)
plt.savefig("plots/pairplot.png")
plt.close()

# Histogram of Heating Load
plt.hist(df['Heating_Load'], bins=30, color='skyblue', edgecolor='black')
plt.title('Heating Load Distribution')
plt.xlabel('Heating Load')
plt.ylabel('Frequency')
plt.savefig("plots/heating_load_histogram.png")
plt.close()

# Histogram of Cooling Load
plt.hist(df['Cooling_Load'], bins=30, color='lightcoral', edgecolor='black')
plt.title('Cooling Load Distribution')
plt.xlabel('Cooling Load')
plt.ylabel('Frequency')
plt.savefig("plots/cooling_load_histogram.png")
plt.close()
