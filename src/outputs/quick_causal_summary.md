# Quick Causal Run Summary
Generated: 2025-11-03T03:27:58.238297

## Predictive check (logistic on Cholesterol_unscaled)
{'coef': -0.5131603380754474, 'auc': 0.5627449912891986}

## Results for continuous

- Estimator: linear_regression
  - method: backdoor.linear_regression
  - ATE: -0.05528520597099512
  - CI: [-0.08133385963483689, -0.02905194740883847]
    - Refuter random_common_cause: new_effect=-0.05532347431167991, p=0.9
    - Refuter placebo_treatment_refuter: new_effect=0.0, p=1.0
    - Refuter data_subset_refuter: new_effect=-0.05415363349422111, p=0.94


## Results for binary

- Estimator: linear_regression
  - method: backdoor.linear_regression
  - ATE: -0.03187917264571549
  - CI: [-0.07482095371848987, 0.013802333908441248]
    - Refuter random_common_cause: new_effect=-0.03182919527470388, p=0.94
    - Refuter placebo_treatment_refuter: new_effect=-0.00016870490894906108, p=0.92
    - Refuter data_subset_refuter: new_effect=-0.03289453941024348, p=0.9199999999999999

- Estimator: psm
  - method: backdoor.propensity_score_matching
  - ATE: -0.042483660130718956
    - Refuter random_common_cause: new_effect=-0.042483660130718956, p=1.0
    - Refuter placebo_treatment_refuter: new_effect=-0.004379084967320262, p=0.98
    - Refuter data_subset_refuter: new_effect=-0.04674386920980926, p=0.88

- Estimator: ipw
  - method: backdoor.propensity_score_weighting
  - ATE: -0.03630411582688775
    - Refuter random_common_cause: new_effect=-0.03630411582688776, p=1.0
    - Refuter placebo_treatment_refuter: new_effect=0.056806766704262344, p=0.1200000000000001
    - Refuter data_subset_refuter: new_effect=-0.03670583917792362, p=0.98
