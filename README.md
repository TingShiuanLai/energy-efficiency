# 🔍 Energy Efficiency: Statistical Modeling and Feature Analysis

This project explores statistical and machine learning techniques to analyze the energy efficiency of residential buildings. Using the UCI Energy Efficiency dataset, we examine the impact of various architectural features on **Heating Load** and **Cooling Load**, identify trade-offs, and apply feature selection and model interpretation strategies.

---

## 📁 Project Structure

```
.
├── energy_efficiency_data.csv                # Dataset (UCI repository)
├── trade-off.py                              # Pareto analysis for heating vs cooling
├── energy_analysis.py                        # Feature importance, model interpretation
├── linear_regression_model.py                # VIF, AIC, residual checks, bootstrap SE
├── logistic_regression_regularized.py        # Logistic model to classify low heating load
├── plots/                                    # Output visualizations
└── pareto_feature_ranges.csv                 # Summary of top feature ranges
```

---

## 🔬 Key Analyses

### 1. `trade-off.py`
- Visualizes the trade-off between heating and cooling loads.
- Identifies **Pareto-efficient** designs using bi-objective analysis.
- Highlights feature distributions that contribute to optimal energy trade-offs.
- Saves boxplots and Pareto front visualizations.

### 2. `energy_analysis.py`
- Performs standardized linear regression.
- Calculates permutation importance with **Random Forest**.
- Generates **partial dependence plots (PDP)**.
- Applies **bootstrap resampling** to examine coefficient uncertainty.
- Saves residual plots, histograms, and coefficient density plots.

### 3. `linear_regression_model.py`
- Applies **VIF filtering** to remove multicollinearity.
- Conducts **backward stepwise AIC** selection.
- Evaluates heteroscedasticity using the **Breusch-Pagan test**.
- Uses **bootstrap** to estimate standard errors for regression coefficients.

### 4. `logistic_regression_regularized.py`
- Converts the regression target into a binary classification: **Low Heating Load**.
- Trains a **regularized logistic regression (L2)** model.
- Reports feature coefficients and **odds ratios**.

---

## 📊 Dataset

- Source: [UCI Energy Efficiency Dataset](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency)
- Features:
  - `Relative_Compactness`, `Wall_Area`, `Glazing_Area`, `Glazing_Area_Distribution`, `Overall_Height`, etc.
- Targets:
  - `Heating_Load`, `Cooling_Load`

---

## 📦 Requirements

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn tqdm
```

---

## 📈 Outputs

All plots are saved to the `plots/` directory, including:
- Trade-off scatterplots
- Pareto boxplots
- Feature importance barplots
- PDPs
- Residual analysis
- Bootstrap coefficient distributions

---

## 👥 Authors

This work was completed through collaborative effort and discussion. All scripts were developed and written by the project contributors.

---

## 📄 License

This project is for academic and educational purposes only.
