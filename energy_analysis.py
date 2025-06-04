import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Load dataset
df = pd.read_csv("energy_efficiency_data.csv")

# Feature selection
features = ['Relative_Compactness', 'Wall_Area', 'Glazing_Area', 'Glazing_Area_Distribution', 'Overall_Height']
# features = ['Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
            # 'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution']
X = df[features]
y_heat = df['Heating_Load']
y_cool = df['Cooling_Load']

# ==== Standardized Coefficients ====
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
y_heat_scaled = StandardScaler().fit_transform(y_heat.values.reshape(-1, 1)).flatten()
y_cool_scaled = StandardScaler().fit_transform(y_cool.values.reshape(-1, 1)).flatten()

X_scaled_const = sm.add_constant(X_scaled)
model_heat = sm.OLS(y_heat_scaled, X_scaled_const).fit()
model_cool = sm.OLS(y_cool_scaled, X_scaled_const).fit()

print("🔹 Standardized Coefficients - Heating Load:\n", model_heat.params)
print("\n🔹 Standardized Coefficients - Cooling Load:\n", model_cool.params)

# ==== Permutation Importance ====
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X, y_heat, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cool, test_size=0.2, random_state=42)

rf_heat = RandomForestRegressor(n_estimators=100, random_state=42)
rf_cool = RandomForestRegressor(n_estimators=100, random_state=42)
rf_heat.fit(X_train_h, y_train_h)
rf_cool.fit(X_train_c, y_train_c)

perm_heat = permutation_importance(rf_heat, X_test_h, y_test_h, n_repeats=30, random_state=42)
perm_cool = permutation_importance(rf_cool, X_test_c, y_test_c, n_repeats=30, random_state=42)

# Plot permutation importance
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x=perm_heat.importances_mean, y=X.columns, ax=axes[0])
axes[0].set_title("Permutation Importance - Heating Load")
sns.barplot(x=perm_cool.importances_mean, y=X.columns, ax=axes[1])
axes[1].set_title("Permutation Importance - Cooling Load")
plt.tight_layout()
plt.savefig("plots/feature_importance_heating_cooling.png")
plt.close()

# ==== Partial Dependence Plots ====
PartialDependenceDisplay.from_estimator(rf_heat, X, features)
plt.suptitle("Partial Dependence - Heating Load")
plt.tight_layout()
plt.savefig("plots/pdp_heating.png")
plt.close()

PartialDependenceDisplay.from_estimator(rf_cool, X, features)
plt.suptitle("Partial Dependence - Cooling Load")
plt.tight_layout()
plt.savefig("plots/pdp_cooling.png")
plt.close()

# ==== Residual Plot for Heating Load ====
X_const = sm.add_constant(X)
model = sm.OLS(y_heat, X_const).fit()
residuals = model.resid
fitted = model.fittedvalues

sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted - Heating Load")
plt.tight_layout()
plt.savefig("plots/residuals_vs_fitted.png")
plt.close()

# ==== Histogram of Heating Load ====
plt.hist(y_heat, bins=30, color='skyblue', edgecolor='black')
plt.title("Heating Load Distribution")
plt.xlabel("Heating Load")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/heating_load_histogram.png")
plt.close()

# ==== Bootstrap Coefficient Density ====
def bootstrap_coefficients(X, y, n_iter=1000):
    np.random.seed(42)
    coefs = []
    for _ in tqdm(range(n_iter), desc="Bootstrapping"):
        idx = np.random.choice(len(y), size=len(y), replace=True)
        X_sample = X.iloc[idx]
        y_sample = y.iloc[idx]
        m = sm.OLS(y_sample, X_sample).fit()
        coefs.append(m.params.values)
    return pd.DataFrame(coefs, columns=X.columns)

coef_df = bootstrap_coefficients(X_const, y_heat)

coef_df.plot(kind='density', subplots=True,
             layout=(int(np.ceil(len(coef_df.columns) / 2)), 2),
             figsize=(12, 8), sharex=False)
plt.suptitle("Bootstrap Coefficient Distributions - Heating Load")
plt.tight_layout()
plt.savefig("plots/bootstrap_coeff_densities.png")
plt.close()
