
# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from sklearn.preprocessing import StandardScaler

# # Load dataset
# df = pd.read_csv('energy_efficiency_data.csv')

# # Step 1: Binarize Heating_Load using the 75th percentile
# threshold = df['Heating_Load'].median() #.quantile(0.75)
# df['High_Heating'] = (df['Heating_Load'] > threshold).astype(int)

# # Step 2: Select predictors and standardize them
# predictors = ['Relative_Compactness', 'Wall_Area', 'Overall_Height', 'Glazing_Area', 'Glazing_Area_Distribution']
# scaler = StandardScaler()
# X_scaled = pd.DataFrame(scaler.fit_transform(df[predictors]), columns=predictors)

# # Add constant
# X_scaled = sm.add_constant(X_scaled)
# y = df['High_Heating']

# # Step 3: Fit regularized logistic regression
# logit_model = sm.Logit(y, X_scaled).fit_regularized(method='l2', maxiter=100, disp=True)
# print("\nRegularized Logistic Regression Summary:")
# print(logit_model.summary())

# # Step 4: Compute odds ratios and confidence intervals (approximate)
# params = logit_model.params
# odds_ratios = pd.DataFrame({
#     'Feature': X_scaled.columns,
#     'Odds Ratio': np.exp(params)
# })
# print("\nOdds Ratios (approximate):")
# print(odds_ratios)

# # Save to file
# odds_ratios.to_csv('logistic_regularized_odds_ratios.csv', index=False)
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("energy_efficiency_data.csv")

# Binarize Heating_Load using 75th percentile
threshold = df['Heating_Load'].quantile(0.75)
df['High_Heating'] = (df['Heating_Load'] > threshold).astype(int)

# Select predictors and scale them
predictors = ['Relative_Compactness', 'Wall_Area', 'Glazing_Area', 'Glazing_Area_Distribution']  # drop unstable ones
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(df[predictors]), columns=predictors)
X_scaled = sm.add_constant(X_scaled)
y = df['High_Heating']

# Fit model with statsmodels (MLE gives SEs and p-values)
model = sm.Logit(y, X_scaled).fit()
print(model.summary())

# Odds ratios and CI
params = model.params
conf = model.conf_int()
conf['OR'] = np.exp(params)
conf.columns = ['2.5%', '97.5%', 'Odds Ratio']
print("\nOdds Ratios and 95% CI:")
print(np.exp(conf))
