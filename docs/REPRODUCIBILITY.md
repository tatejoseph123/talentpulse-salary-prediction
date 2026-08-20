# Reproducibility Guide

Install `requirements.txt` and run commands from the repository root.

```powershell
jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=1200 "My Capstone Project_Rearranged.ipynb"
```

Expected outputs include `salary_prediction_pipeline.pkl`, `feature_columns.pkl`, `Salary_prediction_app.py`, and figures under `reports/figures/`.

The notebook's deployment cell rewrites `requirements.txt` with a minimal Streamlit list. The committed file intentionally contains the complete analysis and deployment environment; restore it after a full run if needed.

## Validation record

On 20 August 2026, a temporary byte-for-byte copy of the supplied notebook executed top-to-bottom against the matching dataset. All 207 cells completed with no captured notebook errors. The model, feature list, and app in this repository came from that run.

- Notebook SHA-256: `5FAEC7EF0927986E41E5662999A124C4E0631D2ECCD75D1130789D97B88E249E`
- Dataset SHA-256: `B3AA6A9B2037FD982C19049BEEBCE47A8B0E65F0A8FE8B235398486A1F27C014`
