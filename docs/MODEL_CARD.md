# Model Card

## Model and intended use

Tuned Random Forest regression pipeline trained on a log-transformed annual salary target. It supports exploratory salary benchmarking and demonstrates an end-to-end regression workflow. It is not intended to automate compensation, hiring, promotion, or other employment decisions.

## Inputs

Country, job level, remote status, required experience, experience tier, skill count, and flags for Python, SQL, Spark, AWS, Excel, machine learning, Power BI, and Tableau.

## Performance

| Split | MAE | RMSE | R² |
|---|---:|---:|---:|
| Untouched 20% test set | $23,374.50 | $32,385.28 | 0.610 |

The result does not beat the stated $18,500 legacy MAE benchmark.

## Limitations

- Historical training salaries cover only five countries.
- Fixed 2024 exchange rates do not capture purchasing power or later currency movement.
- Required experience is mostly missing and may contain extraction or unit errors.
- Extreme salaries and underrepresented role-country combinations can produce large errors.
- Country is a dominant predictor and may encode structural differences requiring fairness review.
- Predictions are point estimates without calibrated uncertainty intervals.

Use current market evidence, subgroup validation, legal requirements, and role-specific context before acting. Load the joblib artifact only from a trusted source because deserialization can execute code.
