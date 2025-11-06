#  Heart Disease Predictive & Causal Analysis

This project combines **Machine Learning** and **Causal Inference** to analyze and predict heart disease risk.
It estimates the **causal impact of cholesterol levels** on heart disease and builds predictive models to classify patients based on medical attributes.
The app also includes **interactive 3D visualization**, **feature importance insights**, and **robustness checks** for causal validation.

---

##  Objectives

* Understand the **causal relationship** between cholesterol and heart disease using the `DoWhy` library.
* Estimate the **Average Treatment Effect (ATE)** and validate it through **refutation (robustness) tests**.
* Build and evaluate predictive models (`Logistic Regression`, `Random Forest`) to classify heart disease.
* Visualize **correlations**, **causal graphs**, **feature importance**, and **3D risk surfaces**.

---

##  Live Demo

👉 Try the app here: [**Heart Disease Causal Analysis App**](https://causecare-fdgclkwrefbep4dnwneguf.streamlit.app/)

You can enter your own numeric data or use the examples below 👇

---

###  Example 1 — Lower Risk

| Feature              | Value |
| -------------------- | ----- |
| Age                  | 45    |
| Sex                  | 0     |
| ChestPainType        | 1     |
| RestingBP            | 120   |
| Cholesterol          | 180   |
| Cholesterol_unscaled | 180   |
| FastingBS            | 0     |
| RestingECG           | 0     |
| MaxHR                | 160   |
| ExerciseAngina       | 0     |
| Oldpeak              | 0.5   |
| ST_Slope             | 2     |

**Predicted Risk:** ~0.39
**ATE (Cholesterol Effect):** -0.0424
**p-value:** 0.98

---

###  Example 2 — Higher Risk

| Feature              | Value |
| -------------------- | ----- |
| Age                  | 58    |
| Sex                  | 1     |
| ChestPainType        | 3     |
| RestingBP            | 140   |
| Cholesterol          | 300   |
| Cholesterol_unscaled | 300   |
| FastingBS            | 1     |
| RestingECG           | 2     |
| MaxHR                | 110   |
| ExerciseAngina       | 1     |
| Oldpeak              | 2.3   |
| ST_Slope             | 1     |

**Predicted Risk:** ~0.60
**ATE (Cholesterol Effect):** -0.0424
**p-value:** 0.98

---

## Project Structure

```
project/
│
├── data/
│   ├── heart.csv                # Raw dataset
│   └── heart_cleaned.csv        # Cleaned dataset
│
├── notebooks/
│   ├── data_analysis.ipynb      # Initial EDA
│   └── eda.ipynb                # Visual exploration
│
├── src/
│   ├── data_processing.py       # Cleaning, encoding, scaling
│   ├── causal_analysis.py       # DoWhy causal model + ATE
│   ├── train_predictive.py      # Logistic Regression + Random Forest
│   ├── quickfix.py              # Debug/refactor fixes
│   ├── summarize_results.py     # Causal result summaries
│   ├── visualize_ates.py        # ATE comparison and plots
│   ├── models/                  # Trained model files
│   └── outputs/                 # CSV/JSON results, visuals
│
├── app.py                       # Streamlit app
│                                # (prediction + causal results + 3D visualization)
│
├── .gitignore                   # Ignore unnecessary files
└── README.md                    # Documentation
```

---

##  Visualizations in App

* **Feature Importance Chart:** Shows which medical features most influence predictions.
* **3D Risk Surface:** Displays how heart disease risk changes with **Cholesterol** and **Age** — dynamically updates with user inputs.
* **ATE & Robustness Tests:** Demonstrates the causal strength and validity of cholesterol’s effect on heart disease.

---

##  Results Summary

* **Causal Finding:** Cholesterol showed a weak and statistically insignificant causal effect on heart disease (ATE ≈ -0.0424, p ≈ 0.98).
* **Predictive Finding:** Random Forest performed best for heart disease classification.
* **Interpretation:** While cholesterol alone doesn’t *cause* heart disease strongly, it remains an important *predictive factor* when combined with other health indicators.

---

##  Tech Stack

**Languages:** Python
**Libraries:** pandas, numpy, scikit-learn, dowhy, matplotlib, plotly, streamlit
**Machine Learning Models:** Logistic Regression, Random Forest
**Causal Inference:** DoWhy (ATE estimation + refutations)
**Visualization:** Matplotlib, Plotly (3D interactive surfaces)

