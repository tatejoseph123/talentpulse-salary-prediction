# TalentPulse Salary Prediction

An end-to-end machine-learning capstone for predicting annual technology-job salaries in USD. The project validates and cleans 4,653 job postings, engineers leakage-safe features, compares three regression algorithms, tunes the strongest model, and exposes the final pipeline through Streamlit.

## 🚀 Live Application

[Lanuch the Tech Job Salary Predictor ] (https://tatejoseph123-talentpulse-salary-p-salary-prediction-app-i68iod.streamlit.app/)

## Results

| Metric | Untouched 20% test set |
|---|---:|
| MAE | $23,374.50 |
| RMSE | $32,385.28 |
| R² | 0.610 |

The selected tuned Random Forest explains about 61% of salary variation, but its MAE remains above the $18,500 legacy benchmark. It should support salary review, not replace human judgment.

## Repository contents

```text
.
├── My Capstone Project_Rearranged.ipynb  # Preserved submitted notebook
├── Salary_prediction_app.py             # Streamlit interface
├── salary_prediction_pipeline.pkl       # Validated fitted pipeline
├── feature_columns.pkl                   # Model input feature list
├── jobs_dataset.csv                     # Notebook-compatible source data
├── data/                                # Raw and processed assets
├── notebooks/                           # Supporting staged notebooks
├── reports/figures/                     # Analytical figures
├── scripts/                             # Reproducibility scripts
├── references/                          # Original case-study brief
└── docs/                                # Project documentation
```

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab "My Capstone Project_Rearranged.ipynb"
```

Run the app:

```powershell
streamlit run Salary_prediction_app.py
```

## Reproduce the analysis

Run from the repository root because notebook paths use the current working directory:

```powershell
jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1200 "My Capstone Project_Rearranged.ipynb"
```

The notebook writes the fitted model, feature list, Streamlit app, figures, and a deployment-focused `requirements.txt`. Restore the repository's complete `requirements.txt` afterward if Git reports it changed.

## Method summary

- Preserve unlabeled salary records as a scoring pool rather than inventing targets.
- Standardize salaries to USD using fixed 2024 official exchange-rate values documented in the notebook.
- Exclude identifiers and direct salary components from model features to limit leakage.
- Compare Linear Regression, Random Forest, and Gradient Boosting with five-fold cross-validation.
- Assess engineered features, permutation importance, VIF, residuals, abnormal salaries, and target transformation before tuning.

## Documentation

- [Project report](docs/PROJECT_REPORT.md)
- [Data dictionary](docs/DATA_DICTIONARY.md)
- [Model card](docs/MODEL_CARD.md)
- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## License

No open-source license has been specified. See [LICENSE](LICENSE) before reuse or redistribution.
