
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load dataset
df = pd.read_csv('energy_efficiency_data.csv')

# Target and predictors
X = df.drop(columns=['Heating_Load', 'Cooling_Load'])
y = df['Heating_Load']  # Choose 'Heating_Load' or 'Cooling_Load' as needed

# Scale predictors
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Add intercept
X_scaled = sm.add_constant(X_scaled)

# VIF filtering
def calculate_vif(X):
    return pd.DataFrame({
        'Feature': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })

while True:
    vif_df = calculate_vif(X_scaled)
    max_vif = vif_df.loc[vif_df['Feature'] != 'const', 'VIF'].max()
    if max_vif > 10:
        drop_feature = vif_df.sort_values('VIF', ascending=False)['Feature'].values[0]
        print(f"Dropping '{drop_feature}' due to high VIF: {max_vif:.2f}")
        X_scaled = X_scaled.drop(columns=[drop_feature])
    else:
        break

# Backward stepwise AIC
def backward_stepwise_aic(X, y, verbose=True):
    initial_vars = X.columns.tolist()
    best_aic = float('inf')
    current_vars = initial_vars.copy()
    best_model = None

    while True:
        changed = False
        aic_with_vars = []

        for var in current_vars:
            if var == 'const':
                continue
            vars_subset = [v for v in current_vars if v != var]
            model = sm.OLS(y, X[vars_subset]).fit()
            aic_with_vars.append((model.aic, var, model))

        aic_with_vars.sort()
        best_new_aic, removed_var, candidate_model = aic_with_vars[0]

        if best_new_aic < best_aic:
            best_aic = best_new_aic
            current_vars.remove(removed_var)
            best_model = candidate_model
            changed = True
            if verbose:
                print(f"Removed: {removed_var}, AIC: {best_new_aic:.4f}")

        if not changed:
            break

    return best_model, current_vars

final_model, selected_vars = backward_stepwise_aic(X_scaled, y)

print("\nFinal model variables:")
print(selected_vars)

print("\nFinal Model Summary:")
print(final_model.summary())
