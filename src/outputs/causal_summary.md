# 🧠 Causal Analysis Summary
Generated: 2025-11-03T03:30:17.637032

## 📊 Average Treatment Effect (ATE) Summary

- **Estimator:** linear_regression
  - ATE Value: -0.0318791726457154
  - Confidence Interval: N/A

- **Estimator:** psm
  - ATE Value: -0.0424836601307189
  - Confidence Interval: N/A

- **Estimator:** ipw
  - ATE Value: -0.0363041158268877
  - Confidence Interval: N/A

## 🔁 Refutation Test Results

### prefix
- binary

### timestamp
- 20251103_032757

### identified_estimand
- Estimand type: EstimandType.NONPARAMETRIC_ATE

### Estimand : 1
Estimand name: backdoor
Estimand expression:
         d                                                                                 ↪
───────────────────(E[HeartDisease|Sex,RestingECG,FastingBS,MaxHR,ST_Slope,RestingBP,Exerc ↪
d[high_cholesterol]                                                                        ↪

↪                                      
↪ iseAngina,ChestPainType,Age,Oldpeak])
↪                                      
Estimand assumption 1, Unconfoundedness: If U→{high_cholesterol} and U→HeartDisease then P(HeartDisease|high_cholesterol,Sex,RestingECG,FastingBS,MaxHR,ST_Slope,RestingBP,ExerciseAngina,ChestPainType,Age,Oldpeak,U) = P(HeartDisease|high_cholesterol,Sex,RestingECG,FastingBS,MaxHR,ST_Slope,RestingBP,ExerciseAngina,ChestPainType,Age,Oldpeak)

### Estimand : 2
Estimand name: iv
No such variable(s) found!

### Estimand : 3
Estimand name: frontdoor
No such variable(s) found!

### Estimand : 4
Estimand name: general_adjustment
Estimand expression:
         d                                                                                 ↪
───────────────────(E[HeartDisease|RestingECG,Sex,FastingBS,MaxHR,ST_Slope,RestingBP,Exerc ↪
d[high_cholesterol]                                                                        ↪

↪                                      
↪ iseAngina,ChestPainType,Age,Oldpeak])
↪                                      
Estimand assumption 1, Unconfoundedness: If U→{high_cholesterol} and U→HeartDisease then P(HeartDisease|high_cholesterol,RestingECG,Sex,FastingBS,MaxHR,ST_Slope,RestingBP,ExerciseAngina,ChestPainType,Age,Oldpeak,U) = P(HeartDisease|high_cholesterol,RestingECG,Sex,FastingBS,MaxHR,ST_Slope,RestingBP,ExerciseAngina,ChestPainType,Age,Oldpeak)


### results
- **linear_regression:** {"method_name": "backdoor.linear_regression", "ate": -0.03187917264571549, "estimator_object_str": "*** Causal Estimate ***\n\n## Identified estimand\nEstimand type: EstimandType.NONPARAMETRIC_ATE\n\n### Estimand : 1\nEstimand name: backdoor\nEstimand expression:\n         d                                                                                 ↪\n───────────────────(E[HeartDisease|Sex,RestingECG,FastingBS,MaxHR,ST_Slope,RestingBP,Exerc ↪\nd[high_cholesterol]                                                                        ↪\n\n↪                                      \n↪ iseAngina,ChestPainType,Age,Oldpeak])\n↪                                      \nEstimand assumption 1, Unconfoundedness: If U→{high_cholesterol} and U→HeartDisease then P(HeartDisease|high_cholesterol,Sex,RestingECG,FastingBS,MaxHR,ST_Slope,RestingBP,ExerciseAngina,ChestPainType,Age,Oldpeak,U) = P(HeartDisease|high_cholesterol,Sex,RestingECG,FastingBS,MaxHR,ST_Slope,RestingBP,ExerciseAngina,ChestPainType,Age,Oldpeak)\n\n## Realized estimand\nb: HeartDisease~high_cholesterol+Sex+RestingECG+FastingBS+MaxHR+ST_Slope+RestingBP+ExerciseAngina+ChestPainType+Age+Oldpeak\nTarget units: ate\n\n## Estimate\nMean value: -0.03187917264571549\n", "ci": [-0.07482095371848987, 0.013802333908441248], "ate_boot_mean": -0.027698847926538678, "refutations": {"random_common_cause": {"refuter": "random_common_cause", "refutation_string": "Refute: Add a random common cause\nEstimated effect:-0.03187917264571549\nNew effect:-0.03182919527470388\np value:0.94\n", "new_effect": -0.03182919527470388, "p_value": 0.94}, "placebo_treatment_refuter": {"refuter": "placebo_treatment_refuter", "refutation_string": "Refute: Use a Placebo Treatment\nEstimated effect:-0.03187917264571549\nNew effect:-0.00016870490894906108\np value:0.92\n", "new_effect": -0.00016870490894906108, "p_value": 0.92}, "data_subset_refuter": {"refuter": "data_subset_refuter", "refutation_string": "Refute: Use a subset of data\nEstimated effect:-0.03187917264571549\nNew effect:-0.03289453941024348\np value:0.9199999999999999\n", "new_effect": -0.03289453941024348, "p_value": 0.9199999999999999}}}
- **psm:** {"method_name": "backdoor.propensity_score_matching", "ate": -0.042483660130718956, "estimator_object_str": "*** Causal Estimate ***\n\n## Identified estimand\nEstimand type: EstimandType.NONPARAMETRIC_ATE\n\n### Estimand : 1\nEstimand name: backdoor\nEstimand expression:\n         d                                                                                 ↪\n───────────────────(E[HeartDisease|Sex,RestingECG,FastingBS,MaxHR,ST_Slope,RestingBP,Exerc ↪\nd[high_cholesterol]                                                                        ↪\n\n↪                                      \n↪ iseAngina,ChestPainType,Age,Oldpeak])\n↪                                      \nEstimand assumption 1, Unconfoundedness: If U→{high_cholesterol} and U→HeartDisease then P(HeartDisease|high_cholesterol,Sex,RestingECG,FastingBS,MaxHR,ST_Slope,RestingBP,ExerciseAngina,ChestPainType,Age,Oldpeak,U) = P(HeartDisease|high_cholesterol,Sex,RestingECG,FastingBS,MaxHR,ST_Slope,RestingBP,ExerciseAngina,ChestPainType,Age,Oldpeak)\n\n## Realized estimand\nb: HeartDisease~high_cholesterol+Sex+RestingECG+FastingBS+MaxHR+ST_Slope+RestingBP+ExerciseAngina+ChestPainType+Age+Oldpeak\nTarget units: ate\n\n## Estimate\nMean value: -0.042483660130718956\n", "refutations": {"random_common_cause": {"refuter": "random_common_cause", "refutation_string": "Refute: Add a random common cause\nEstimated effect:-0.042483660130718956\nNew effect:-0.042483660130718956\np value:1.0\n", "new_effect": -0.042483660130718956, "p_value": 1.0}, "placebo_treatment_refuter": {"refuter": "placebo_treatment_refuter", "refutation_string": "Refute: Use a Placebo Treatment\nEstimated effect:-0.042483660130718956\nNew effect:-0.004379084967320262\np value:0.98\n", "new_effect": -0.004379084967320262, "p_value": 0.98}, "data_subset_refuter": {"refuter": "data_subset_refuter", "refutation_string": "Refute: Use a subset of data\nEstimated effect:-0.042483660130718956\nNew effect:-0.04674386920980926\np value:0.88\n", "new_effect": -0.04674386920980926, "p_value": 0.88}}}
- **ipw:** {"method_name": "backdoor.propensity_score_weighting", "ate": -0.03630411582688775, "estimator_object_str": "*** Causal Estimate ***\n\n## Identified estimand\nEstimand type: EstimandType.NONPARAMETRIC_ATE\n\n### Estimand : 1\nEstimand name: backdoor\nEstimand expression:\n         d                                                                                 ↪\n───────────────────(E[HeartDisease|Sex,RestingECG,FastingBS,MaxHR,ST_Slope,RestingBP,Exerc ↪\nd[high_cholesterol]                                                                        ↪\n\n↪                                      \n↪ iseAngina,ChestPainType,Age,Oldpeak])\n↪                                      \nEstimand assumption 1, Unconfoundedness: If U→{high_cholesterol} and U→HeartDisease then P(HeartDisease|high_cholesterol,Sex,RestingECG,FastingBS,MaxHR,ST_Slope,RestingBP,ExerciseAngina,ChestPainType,Age,Oldpeak,U) = P(HeartDisease|high_cholesterol,Sex,RestingECG,FastingBS,MaxHR,ST_Slope,RestingBP,ExerciseAngina,ChestPainType,Age,Oldpeak)\n\n## Realized estimand\nb: HeartDisease~high_cholesterol+Sex+RestingECG+FastingBS+MaxHR+ST_Slope+RestingBP+ExerciseAngina+ChestPainType+Age+Oldpeak\nTarget units: ate\n\n## Estimate\nMean value: -0.03630411582688775\n", "refutations": {"random_common_cause": {"refuter": "random_common_cause", "refutation_string": "Refute: Add a random common cause\nEstimated effect:-0.03630411582688775\nNew effect:-0.03630411582688776\np value:1.0\n", "new_effect": -0.03630411582688776, "p_value": 1.0}, "placebo_treatment_refuter": {"refuter": "placebo_treatment_refuter", "refutation_string": "Refute: Use a Placebo Treatment\nEstimated effect:-0.03630411582688775\nNew effect:0.056806766704262344\np value:0.1200000000000001\n", "new_effect": 0.056806766704262344, "p_value": 0.1200000000000001}, "data_subset_refuter": {"refuter": "data_subset_refuter", "refutation_string": "Refute: Use a subset of data\nEstimated effect:-0.03630411582688775\nNew effect:-0.03670583917792362\np value:0.98\n", "new_effect": -0.03670583917792362, "p_value": 0.98}}}

### encoded_info

---
✅ Summary generated by summarize_results.py