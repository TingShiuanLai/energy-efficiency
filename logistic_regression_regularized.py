from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load and preprocess
df = pd.read_csv("energy_efficiency_data.csv")
threshold = df['Heating_Load'].quantile(0.25)
df['Low_Heating'] = (df['Heating_Load'] < threshold).astype(int)

predictors = ['Relative_Compactness', 'Wall_Area', 'Glazing_Area', 'Glazing_Area_Distribution', 'Overall_Height']
X = df[predictors]
y = df['Low_Heating']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit regularized logistic regression
clf = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
clf.fit(X_scaled, y)

# Coefficients and odds ratios
import numpy as np
odds_ratios = np.exp(clf.coef_[0])

for feature, coef, or_ in zip(predictors, clf.coef_[0], odds_ratios):
    print(f"{feature:30s} Coef: {coef: .4f}  |  Odds Ratio: {or_: .4f}")

