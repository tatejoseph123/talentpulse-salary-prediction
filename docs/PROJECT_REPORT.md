# Project Report

## Objective

Develop an explainable regression workflow that predicts average annual technology-job salary in USD and compare it with the legacy MAE benchmark of $18,500.

## Data and preparation

The source contains 4,653 job postings across five countries and 17 original fields. The workflow standardizes text, audits missingness and duplicates, validates salary bounds and experience, parses skills, converts local salaries to USD, and separates labeled training candidates from unlabeled scoring records and questionable salaries.

Missing targets are not imputed. Salary minimum, maximum, range, `has_salary`, and identifiers are excluded from model inputs because they reveal, proxy, or do not meaningfully explain the target.

## Modeling and result

The notebook compares Linear Regression, Random Forest, and Gradient Boosting using five-fold cross-validation. It evaluates engineered features, permutation importance, VIF, residual behavior, target skewness, and a log-transformed target before tuning the selected Random Forest.

On the untouched 20% test set, the model achieved MAE of $23,374.50, RMSE of $32,385.28, and R² of 0.610. It improved on simple baselines but did not beat the $18,500 legacy MAE benchmark.

## Recommendation

Use the model as a benchmarking aid and review trigger. Continue human review for unusual predictions, country-specific discrepancies, and poorly represented roles. Before production use, validate on fresher salary data, add reliable role and company features, and monitor error by country and seniority.

The preserved notebook is the authoritative analysis record; narrative values here come from its executed outputs.
