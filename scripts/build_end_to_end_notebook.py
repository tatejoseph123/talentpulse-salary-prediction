"""Build the self-contained end-to-end TalentPulse modeling notebook."""

from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "scripts" / "talentpulse_end_to_end.py"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "02_end_to_end_salary_modeling.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


source = SOURCE_PATH.read_text(encoding="utf-8")
source_lines = source.splitlines()
tree = ast.parse(source)


def node_source(node: ast.AST) -> str:
    decorators = getattr(node, "decorator_list", [])
    start_line = min(
        [node.lineno, *(decorator.lineno for decorator in decorators)]
    )
    return "\n".join(source_lines[start_line - 1 : node.end_lineno])


imports = []
definitions = []
functions: dict[str, str] = {}

for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        imports.append(node_source(node))
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        call_name = ast.unparse(node.value.func)
        if call_name == "matplotlib.use":
            imports.append(node_source(node))
    elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.ClassDef)):
        definitions.append(node_source(node))
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        functions[node.name] = node_source(node)


def function_cell(*names: str) -> str:
    return "\n\n\n".join(functions[name] for name in names)


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
}

nb["cells"] = [
    markdown(
        """
        # TalentPulse Salary Prediction
        ## End-to-End Regression Pipeline

        **Role:** Machine Learning Engineer  
        **Primary objective:** build an explainable salary-regression pipeline and determine whether it improves on the legacy **$18,500 MAE**.  
        **Source:** TalentPulse Jobs Dataset, 4,653 job postings across five countries.

        This notebook is self-contained and executes from raw data through cleaning, EDA, feature engineering, model comparison, tuning, final evaluation, scoring, charts, and the HR benchmarking memo.
        """
    ),
    markdown(
        """
        ## tl;dr

        The validated run produces **2,229 model-ready labeled records**, **2,336 unlabeled scoring records**, and a **66-row review queue**. Random Forest is the strongest untuned development model. The selected tuned model uses engineered features, job-title text, and a log target.

        Final locked-holdout performance is approximately:

        - **MAE:** $20,175
        - **RMSE:** $30,194
        - **R2:** 0.694

        The model does **not** beat the legacy $18,500 MAE, so it should not replace the legacy system yet. The main blockers are missing pay-period metadata and weak labeled coverage in Australia, Canada, and India.
        """
    ),
    markdown(
        """
        ## Context & Methods

        The case study requires:

        1. full data validation, cleaning, and EDA;
        2. defensible handling of missing salary targets;
        3. currency standardization and salary-unit review;
        4. salary-range, seniority, experience, and skill engineering;
        5. Pearson correlation, VIF, and log-target analysis;
        6. Linear Regression, Random Forest, and Gradient Boosting comparison;
        7. five-fold cross-validation and tuning;
        8. Random Forest depth and Ridge-alpha diagnostics;
        9. Standard-versus-Premium and subgroup bias evaluation;
        10. an actionable HR salary-benchmarking recommendation.

        ### Key assumptions

        - Missing `salary_avg` values are never imputed. They form the scoring pool.
        - Salary values are treated as local currency and converted using fixed 2024 rates.
        - The dataset has no pay-period field. Questionable values are quarantined rather than silently annualized.
        - Review floors are $20,000 for USA, UK, Canada, and Australia; $3,000 for India. The upper review threshold is $500,000.
        - Salary bounds and salary range are target-derived and excluded from model inputs.
        - Model selection uses development cross-validation. The final holdout is opened only after selection and tuning.
        """
    ),
    markdown("## Setup\n\nImport dependencies, define constants, and configure deterministic behavior."),
    code("\n\n".join(imports) + "\n\nfrom IPython.display import Image, Markdown, display"),
    code("\n\n".join(definitions)),
    markdown(
        """
        ### Concision notes

        The code uses short forms where they give identical results:

        - `df.copy()` is already a deep copy by default.
        - `100 * df.isna().mean()` is equivalent to `df.isna().sum() / len(df) * 100`.
        - `series.gt(0)` already returns `False` for missing values.
        """
    ),
    markdown("### Utility and data-quality functions"),
    code(
        function_cell(
            "configure_output",
            "normalize_text",
            "normalized_key",
            "parse_skills",
            "add_duplicate_audit",
            "clean_and_engineer",
        )
    ),
    markdown("### Modeling and metric functions"),
    code(
        function_cell(
            "make_preprocessor",
            "make_model_pipeline",
            "metric_values",
            "cv_summary",
        )
    ),
    markdown("### Chart and EDA functions"),
    code(
        function_cell(
            "save_figure",
            "add_title",
            "create_eda_outputs",
            "calculate_vif",
        )
    ),
    markdown("### Model-diagnostic and tuning functions"),
    code(
        function_cell(
            "feature_set_ablation",
            "ridge_alpha_analysis",
            "random_forest_depth_analysis",
            "compare_baseline_models",
            "tune_strongest_model",
            "compare_target_scales",
        )
    ),
    markdown("### Evaluation, export, and memo functions"),
    code(
        function_cell(
            "grouped_performance",
            "final_evaluation",
            "export_data_layers",
            "write_benchmarking_memo",
        )
    ),
    markdown(
        """
        ## Data

        ### 1. Resolve project paths

        The notebook works whether Jupyter starts in the project root or in `notebooks/`. Generated outputs are isolated under `artifacts/end_to_end_notebook`.
        """
    ),
    code(
        """
        search_locations = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
        PROJECT_ROOT = next(
            path for path in search_locations
            if (path / "data" / "raw" / "jobs_dataset.csv").exists()
        )
        DATA_PATH = PROJECT_ROOT / "data" / "raw" / "jobs_dataset.csv"
        OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "end_to_end_notebook"
        paths = configure_output(OUTPUT_ROOT)

        warnings.filterwarnings(
            "ignore",
            message=r"Found unknown categories in columns .* during transform.*",
            category=UserWarning,
            module="sklearn[.]preprocessing[.]_encoders",
        )
        sns.set_theme(style="whitegrid", context="notebook")
        pd.set_option("display.max_columns", 40)
        pd.set_option("display.float_format", lambda value: f"{value:,.2f}")

        print(f"Project root: {PROJECT_ROOT}")
        print(f"Source data: {DATA_PATH}")
        print(f"Output root: {OUTPUT_ROOT}")
        """
    ),
    markdown("### 2. Load and inspect the Bronze data"),
    code(
        """
        raw = pd.read_csv(DATA_PATH, encoding="utf-8")
        print(f"Observed shape: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
        print(f"Unique job_id: {raw['job_id'].is_unique}")
        print(f"Missing salary_avg: {raw['salary_avg'].isna().sum():,}")
        display(raw[[
            "job_id", "job_title", "company", "country", "job_level",
            "is_remote", "salary_avg", "has_salary"
        ]].head())
        """
    ),
    markdown(
        """
        ### 3. Clean, deduplicate, engineer features, and export data layers

        Conflicting duplicate groups are preserved for review. Non-conflicting repeats keep one canonical posting. Salary targets with invalid bounds or questionable annual units are excluded from training but retained for audit.
        """
    ),
    code(
        """
        silver, quality_profile = clean_and_engineer(raw)
        training, scoring, review = export_data_layers(
            silver, quality_profile, paths["processed"]
        )

        assert training["salary_avg_usd"].notna().all()
        assert scoring["salary_avg_usd"].isna().all()
        assert training["posting_fingerprint"].is_unique
        assert not set(training["posting_fingerprint"]) & set(scoring["posting_fingerprint"])

        status_summary = (
            silver["salary_quality_status"]
            .value_counts()
            .rename_axis("status")
            .rename("rows")
            .to_frame()
            .assign(pct=lambda frame: 100 * frame["rows"] / len(silver))
        )
        display(status_summary.round(2))
        display(quality_profile.head(10))
        """
    ),
    markdown(
        """
        **Missing-target decision:** missing salaries are not feature-level missingness. Imputing them would fabricate labels. They are therefore modeled separately as the future scoring population.
        """
    ),
    markdown("## Results\n\n### 4. Exploratory analysis and required charts"),
    code(
        """
        coverage, correlations = create_eda_outputs(
            silver,
            training,
            quality_profile,
            paths["figures"],
            paths["processed"],
        )
        display(coverage.round(2))
        display(correlations.round(4))
        """
    ),
    code(
        """
        display(Image(filename=str(paths["figures"] / "01_missing_data_by_column.png"), width=900))
        display(Image(filename=str(paths["figures"] / "02_salary_coverage_by_country.png"), width=950))
        display(Image(filename=str(paths["figures"] / "03_salary_distribution_and_log.png"), width=1100))
        """
    ),
    code(
        """
        display(Image(filename=str(paths["figures"] / "04_salary_qq_before_after_log.png"), width=1050))
        display(Image(filename=str(paths["figures"] / "05_numeric_feature_correlations.png"), width=900))
        """
    ),
    code(
        """
        display(Image(filename=str(paths["figures"] / "06_salary_by_country_and_job_level.png"), width=1100))
        display(Image(filename=str(paths["figures"] / "07_salary_by_role_family_country.png"), width=1100))
        """
    ),
    markdown(
        """
        ### 5. VIF and multicollinearity

        `num_skills` is the sum of the eight individual skill flags, so including all of them together creates perfect linear dependence. The final nonredundant feature candidates therefore compare the count and specific flags separately.
        """
    ),
    code(
        """
        vif = calculate_vif(training, paths["processed"], paths["figures"])
        display(vif)
        display(Image(filename=str(paths["figures"] / "08_vif_results.png"), width=900))
        """
    ),
    markdown("### 6. Create the development and locked final holdout split"),
    code(
        """
        development, test = train_test_split(
            training,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=training["country"],
        )
        y_dev = development["salary_avg_usd"].copy()
        y_test = test["salary_avg_usd"].copy()
        stratified_cv = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        cv_splits = list(stratified_cv.split(development, development["country"]))

        print(f"Development rows: {len(development):,}")
        print(f"Locked holdout rows: {len(test):,}")
        display(pd.crosstab(development["country"], columns="development"))
        display(pd.crosstab(test["country"], columns="holdout"))
        """
    ),
    markdown("### 7. Determine which engineered features improve cross-validated R2"),
    code(
        """
        selected_feature_set, feature_results = feature_set_ablation(
            development,
            y_dev,
            cv_splits,
            paths["processed"],
            paths["figures"],
        )
        selected_spec = FEATURE_SPECS[selected_feature_set]
        print(f"Selected feature set: {selected_feature_set}")
        display(feature_results.round(3))
        display(Image(filename=str(paths["figures"] / "09_feature_set_ablation.png"), width=900))
        """
    ),
    markdown("### 8. Select Ridge alpha and inspect Random Forest depth 3-20"),
    code(
        """
        best_alpha, ridge_results = ridge_alpha_analysis(
            selected_spec,
            development,
            y_dev,
            cv_splits,
            paths["processed"],
            paths["figures"],
        )
        overfit_text, depth_results = random_forest_depth_analysis(
            selected_spec,
            development,
            y_dev,
            paths["processed"],
            paths["figures"],
        )
        print(f"Best Ridge alpha: {best_alpha:g}")
        print(f"Random Forest overfitting result: {overfit_text}")
        display(ridge_results.round(2))
        display(depth_results.round(2))
        """
    ),
    code(
        """
        display(Image(filename=str(paths["figures"] / "10_ridge_alpha_search.png"), width=850))
        display(Image(filename=str(paths["figures"] / "11_random_forest_depth_gap.png"), width=950))
        """
    ),
    markdown("### 9. Compare the three required algorithms"),
    code(
        """
        strongest, baseline_pipelines, baseline_results = compare_baseline_models(
            selected_spec,
            development,
            y_dev,
            cv_splits,
            paths["processed"],
            paths["figures"],
        )
        print(f"Strongest development model: {strongest}")
        display(baseline_results.round(2))
        display(Image(filename=str(paths["figures"] / "12_baseline_model_cv_comparison.png"), width=850))
        """
    ),
    markdown("### 10. Tune the strongest model and compare original versus log target"),
    code(
        """
        tuned_model, tuning_results = tune_strongest_model(
            strongest,
            selected_spec,
            development,
            y_dev,
            cv_splits,
            paths["processed"],
        )
        selected_model, selected_target, target_results = compare_target_scales(
            tuned_model,
            development[selected_spec.columns],
            y_dev,
            cv_splits,
            paths["processed"],
        )
        best_parameters = json.loads(
            (paths["processed"] / "best_hyperparameters.json").read_text(encoding="utf-8")
        )
        print("Best hyperparameters:", best_parameters)
        print(f"Selected target: {selected_target}")
        display(target_results.round(2))
        """
    ),
    markdown("### 11. Open the final holdout once and evaluate algorithms and bias"),
    code(
        """
        _, test_results, audit, country_metrics, segment_metrics = final_evaluation(
            selected_spec,
            development,
            test,
            y_dev,
            y_test,
            baseline_pipelines,
            selected_model,
            selected_target,
            paths["processed"],
            paths["figures"],
        )
        display(test_results.round(3))
        display(country_metrics.round(2))
        display(segment_metrics.round(2))
        """
    ),
    code(
        """
        display(Image(filename=str(paths["figures"] / "13_final_holdout_model_rmse.png"), width=900))
        display(Image(filename=str(paths["figures"] / "14_final_holdout_residual_diagnostics.png"), width=1100))
        display(Image(filename=str(paths["figures"] / "15_residual_bias_by_country_and_level.png"), width=1100))
        display(Image(filename=str(paths["figures"] / "16_permutation_feature_importance.png"), width=900))
        """
    ),
    markdown("### 12. Refit on all approved labels, score unlabeled postings, and save the model"),
    code(
        """
        final_model = clone(selected_model).fit(
            training[selected_spec.columns],
            training["salary_avg_usd"],
        )
        scoring_predictions = scoring[
            ["job_id", "job_title", "company", "country", "job_level", "salary_segment"]
        ].copy()
        scoring_predictions["predicted_salary_usd"] = final_model.predict(
            scoring[selected_spec.columns]
        )
        scoring_predictions.to_csv(
            paths["processed"] / "scoring_predictions.csv",
            index=False,
        )
        joblib.dump(final_model, paths["models"] / "talentpulse_salary_pipeline.joblib")
        with (paths["models"] / "model_contract.json").open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "selected_feature_set": selected_feature_set,
                    "feature_columns": selected_spec.columns,
                    "selected_target": selected_target,
                    "legacy_mae_usd": LEGACY_MAE_USD,
                    "industry_mae_usd": INDUSTRY_MAE_USD,
                    "training_rows": len(training),
                    "scoring_rows": len(scoring),
                    "random_state": RANDOM_STATE,
                },
                stream,
                indent=2,
            )
        display(scoring_predictions.head())
        """
    ),
    markdown("### 13. Generate the HR benchmarking memo"),
    code(
        """
        write_benchmarking_memo(
            paths["reports"] / "benchmarking_memo.md",
            DATA_PATH,
            training,
            scoring,
            review,
            selected_feature_set,
            strongest,
            selected_target,
            overfit_text,
            best_alpha,
            test_results,
            audit,
            country_metrics,
            segment_metrics,
        )
        memo_text = (paths["reports"] / "benchmarking_memo.md").read_text(encoding="utf-8")
        display(Markdown(memo_text))
        """
    ),
    markdown("## Checks"),
    code(
        """
        selected_test = test_results.loc[
            test_results["Model"].str.startswith("Tuned final")
        ].iloc[0]

        assert raw.shape == (4_653, 17)
        assert training["posting_fingerprint"].is_unique
        assert training["salary_avg_usd"].notna().all()
        assert scoring["salary_avg_usd"].isna().all()
        assert len(scoring_predictions) == len(scoring)
        assert np.isfinite(scoring_predictions["predicted_salary_usd"]).all()
        assert (paths["models"] / "talentpulse_salary_pipeline.joblib").exists()
        assert (paths["reports"] / "benchmarking_memo.md").exists()
        assert len(list(paths["figures"].glob("*.png"))) == 16

        print("All notebook data, modeling, export, and chart checks passed.")
        print(f"Final holdout MAE: ${selected_test['MAE']:,.2f}")
        print(f"Final holdout RMSE: ${selected_test['RMSE']:,.2f}")
        print(f"Final holdout R2: {selected_test['R2']:.3f}")
        """
    ),
    markdown(
        """
        ## Takeaways

        - The selected model improves substantially over a simple linear specification but does not beat the $18,500 legacy MAE.
        - Random Forest has the lowest holdout RMSE among the three required algorithms, while the tuned log-target candidate has the lowest selected-candidate MAE.
        - Australia shows the largest mean country underprediction; Lead roles show the largest mean job-level underprediction.
        - Premium roles have slightly higher MAE than Standard roles.
        - The score pool is concentrated in countries with weak salary-label coverage, so aggregate test accuracy is not sufficient for deployment.
        - Production work should prioritize explicit pay-period metadata, better salary coverage in Australia/Canada/India, and a fresh untouched holdout after data improvements.
        """
    ),
    markdown(
        """
        ## Next Steps

        1. Review `talentpulse_review_queue.csv` and confirm salary period/currency rules.
        2. Collect additional labeled salaries for Australia, Canada, and India.
        3. Add explicit hourly/daily/monthly/annual metadata and rebuild the annual target.
        4. Retrain using a new untouched holdout rather than tuning against the current final results.
        5. Deploy only as a controlled HR decision-support pilot after beating the legacy MAE and passing subgroup guardrails.
        """
    ),
]

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
