
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import statsmodels.api as sm

# Ensure output directory exists
os.makedirs("plots", exist_ok=True)

# Load dataset
df = pd.read_csv('energy_efficiency_data.csv')

# Fit a model
X = df.drop(columns=['Heating_Load', 'Cooling_Load'])
y = df['Heating_Load']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Residuals vs Fitted Plot
residuals = model.resid
fitted = model.fittedvalues
sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.tight_layout()
plt.savefig("plots/residuals_vs_fitted.png")
plt.close()

# Histogram of Heating Load
plt.hist(df['Heating_Load'], bins=30, color='skyblue', edgecolor='black')
plt.title('Heating Load Distribution')
plt.xlabel('Heating Load')
plt.ylabel('Frequency')
plt.savefig("plots/heating_load_histogram.png")
plt.close()

# Bootstrap coefficient density plots
from tqdm import tqdm

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

coef_df = bootstrap_coefficients(X, y)

# Plotting density
coef_df.plot(kind='density', subplots=True, layout=(int(np.ceil(len(coef_df.columns)/2)), 2), figsize=(12, 8), sharex=False)
plt.suptitle("Bootstrap Coefficient Distributions")
plt.tight_layout()
plt.savefig("plots/bootstrap_coeff_densities.png")
plt.close()
