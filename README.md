# ❤️ Heart Disease Predictive & Causal Analysis

This project combines **Machine Learning** and **Causal Inference** to analyze and predict heart disease risk.
It estimates the **causal impact of cholesterol levels** on heart disease and builds predictive models to classify patients based on medical attributes.

## 🎯 Objectives

- Understand the causal relationship between **cholesterol** and **heart disease** using the `DoWhy` library.
- Estimate the **Average Treatment Effect (ATE)** and validate it through refutation tests.
- Build and evaluate predictive models (`Logistic Regression`, `Random Forest`) to classify heart disease.
- Visualize correlations, causal graphs, and model results.

## 🚀 Live Demo  
Try the app here 👉 [Heart Disease Causal Analysis App](https://causecare-fdgclkwrefbep4dnwneguf.streamlit.app/)

Below are sample numeric inputs you can use to test the model:

### 🩺 Example 1 (Lower Risk)
| Feature | Value |
|----------|--------|
| Age | 45 |
| Sex | 0 |
| ChestPainType | 1 |
| RestingBP | 120 |
| Cholesterol | 180 |
| FastingBS | 0 |
| RestingECG | 0 |
| MaxHR | 160 |
| ExerciseAngina | 0 |
| Oldpeak | 0.5 |
| ST_Slope | 2 |

**Predicted Risk:** ~0.39  
**ATE (Cholesterol Effect):** -0.0424  
**p-value:** 0.98  

---

### ❤️ Example 2 (Higher Risk)
| Feature | Value |
|----------|--------|
| Age | 58 |
| Sex | 1 |
| ChestPainType | 3 |
| RestingBP | 140 |
| Cholesterol | 300 |
| FastingBS | 1 |
| RestingECG | 2 |
| MaxHR | 110 |
| ExerciseAngina | 1 |
| Oldpeak | 2.3 |
| ST_Slope | 1 |

**Predicted Risk:** ~0.6
**ATE (Cholesterol Effect):** -0.0424  
**p-value:** 0.98  


## 📁 Project Structure
project/
│
├── data/
│ ├── heart.csv # Raw heart disease dataset
│ └── heart_cleaned.csv # Preprocessed & cleaned dataset
│
├── notebooks/
│ ├── data_analysis.ipynb # Early data exploration and EDA
│ └── eda.ipynb # Visual analysis & feature understanding
│
├── src/
│ ├── data_processing.py # Cleans and encodes data, creates heart_cleaned.csv
│ ├── causal_analysis.py # Builds DoWhy causal model & estimates ATE
│ ├── train_predictive.py # Trains ML models (LogReg, RandomForest)
│ ├── quickfix.py # Debug/refactor script for handling summary issues
│ ├── summarize_results.py # Generates readable summaries from causal outputs
│ ├── visualize_ates.py # Plots ATE comparisons and confidence intervals
│ │
│ ├── models/ # Trained models (.joblib files)
│ └── outputs/ # Causal results, JSON, CSV, and visualizations
│
├── app.py # Streamlit app (for risk prediction + explanations)
│
├── .gitignore # Files/folders to ignore on GitHub (like heart-env/)
└── README.md # Project documentation

## 📊 Results Summary
- **Causal Finding:** Cholesterol showed a weak and statistically insignificant causal effect on heart disease (ATE ≈ -0.0424, p ≈ 0.98).  
- **Best Predictive Model:** Random Forest (Accuracy ≈ 0.89, ROC-AUC ≈ 0.94).  
- **Interpretation:** While cholesterol alone isn’t a strong causal driver, models indicate it still helps improve predictive accuracy when combined with other features.

## 🧠 Tech Stack
- **Languages:** Python  
- **Libraries:** pandas, numpy, scikit-learn, dowhy, matplotlib, streamlit  
- **ML Models:** Logistic Regression, Random Forest  
- **Causal Inference:** DoWhy (ATE estimation + refutations)