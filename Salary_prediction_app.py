
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tech Salary Predictor",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       PAGE
       ======================================================== */

    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }


    /* ========================================================
       MAIN TITLE
       ======================================================== */

    .main-title {
        color: #0F172A !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #64748B !important;
        font-size: 1rem;
        margin-bottom: 2rem;
    }


    /* ========================================================
       HEADINGS
       ======================================================== */

    h1, h2, h3 {
        color: #0F172A !important;
    }


    /* ========================================================
       NORMAL TEXT
       ======================================================== */

    p {
        color: #334155;
    }


    /* ========================================================
       INPUT LABELS
       ======================================================== */

    label,
    div[data-testid="stWidgetLabel"] p {
        color: #334155 !important;
        font-weight: 500;
    }


    /* ========================================================
       SELECT BOXES
       ======================================================== */

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        color: #0F172A !important;
    }

    div[data-baseweb="select"] span {
        color: #0F172A !important;
    }

    div[data-baseweb="select"] svg {
        fill: #475569 !important;
    }


    /* ========================================================
       NUMBER INPUT
       ======================================================== */

    div[data-testid="stNumberInput"] input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border-color: #CBD5E1 !important;
    }


    /* ========================================================
       CHECKBOXES
       ======================================================== */

    div[data-testid="stCheckbox"] label p {
        color: #334155 !important;
    }


    /* ========================================================
       PRIMARY BUTTON
       ======================================================== */

    div.stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        min-height: 46px;
        transition: all 0.2s ease;
    }

    div.stButton > button[kind="primary"] p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    div.stButton > button[kind="primary"]:hover p {
        color: #FFFFFF !important;
    }


    /* ========================================================
       SALARY RESULT CARD
       ======================================================== */

    div[data-testid="stMetric"] {
        background-color: #F0FDFA !important;
        padding: 30px;
        border-radius: 14px;
        border: 1px solid #99F6E4;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 500;
    }

    div[data-testid="stMetricValue"] {
        color: #0F766E !important;
        font-size: 2.6rem !important;
        font-weight: 700 !important;
    }


    /* ========================================================
       INFORMATION BOX
       ======================================================== */

    div[data-testid="stAlert"] {
        background-color: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        border-radius: 10px;
    }

    div[data-testid="stAlert"] p {
        color: #1E3A8A !important;
    }


    /* ========================================================
       EXPANDER
       ======================================================== */

    div[data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
    }

    div[data-testid="stExpander"] summary {
        background-color: #EFF6FF !important;
        color: #1E3A8A !important;
        border-radius: 10px !important;
    }

    div[data-testid="stExpander"] summary p {
        color: #1E3A8A !important;
        font-weight: 600 !important;
    }


    /* ========================================================
       TABLE
       ======================================================== */

    div[data-testid="stTable"] {
        background-color: #FFFFFF !important;
    }

    div[data-testid="stTable"] table {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    div[data-testid="stTable"] th {
        background-color: #EFF6FF !important;
        color: #1E3A8A !important;
    }

    div[data-testid="stTable"] td {
        background-color: #FFFFFF !important;
        color: #334155 !important;
    }


    /* ========================================================
       DIVIDER AND FOOTER
       ======================================================== */

    hr {
        border-color: #E2E8F0 !important;
    }

    div[data-testid="stCaptionContainer"] {
        color: #64748B !important;
    }

    div[data-testid="stCaptionContainer"] p {
        color: #64748B !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load(
    "salary_prediction_pipeline.pkl"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💼 Tech Job Salary Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Estimate annual salary from job characteristics, '
    'experience, and required technical skills.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MAIN LAYOUT
# ============================================================

input_col, result_col = st.columns(
    [1.25, 1],
    gap="large"
)


# ============================================================
# LEFT COLUMN — INPUTS
# ============================================================

with input_col:

    st.subheader("Job Details")

    col1, col2 = st.columns(2)

    with col1:

        country = st.selectbox(
            "Country",
            [
                "USA",
                "UK",
                "Canada",
                "India",
                "Australia"
            ]
        )

        job_level = st.selectbox(
            "Job Level",
            [
                "Junior",
                "Mid",
                "Senior",
                "Lead"
            ]
        )

    with col2:

        is_remote = st.selectbox(
            "Remote Job?",
            [False, True]
        )

        experience_required = st.number_input(
            "Years of Experience Required",
            min_value=0.0,
            max_value=40.0,
            value=3.0,
            step=1.0
        )


    # ========================================================
    # DERIVE EXPERIENCE TIER AUTOMATICALLY
    # ========================================================

    if experience_required <= 2:
        experience_tier = "Entry"

    elif experience_required <= 5:
        experience_tier = "Intermediate"

    elif experience_required <= 10:
        experience_tier = "Experienced"

    else:
        experience_tier = "Highly Experienced"

    st.info(
        f"Experience Tier: {experience_tier}"
    )


    # ========================================================
    # REQUIRED SKILLS
    # ========================================================

    st.subheader("Required Skills")

    skill_col1, skill_col2 = st.columns(2)

    with skill_col1:

        skill_python = st.checkbox("Python")
        skill_sql = st.checkbox("SQL")
        skill_spark = st.checkbox("Spark")
        skill_aws = st.checkbox("AWS")

    with skill_col2:

        skill_excel = st.checkbox("Excel")
        skill_machine_learning = st.checkbox(
            "Machine Learning"
        )
        skill_power_bi = st.checkbox("Power BI")
        skill_tableau = st.checkbox("Tableau")


    # ========================================================
    # COUNT SELECTED SKILLS
    # ========================================================

    num_skills = sum([
        skill_python,
        skill_sql,
        skill_spark,
        skill_aws,
        skill_excel,
        skill_machine_learning,
        skill_power_bi,
        skill_tableau
    ])


    # ========================================================
    # BUILD MODEL INPUT
    # ========================================================

    input_data = pd.DataFrame({
        "job_level": [job_level],
        "country": [country],
        "is_remote": [is_remote],
        "experience_tier": [experience_tier],
        "experience_required": [experience_required],
        "num_skills": [num_skills],
        "skill_python": [int(skill_python)],
        "skill_sql": [int(skill_sql)],
        "skill_spark": [int(skill_spark)],
        "skill_aws": [int(skill_aws)],
        "skill_excel": [int(skill_excel)],
        "skill_machine_learning": [
            int(skill_machine_learning)
        ],
        "skill_power_bi": [
            int(skill_power_bi)
        ],
        "skill_tableau": [
            int(skill_tableau)
        ]
    })


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = st.button(
        "Predict Salary",
        type="primary",
        use_container_width=True
    )


# ============================================================
# RIGHT COLUMN — RESULTS
# ============================================================

with result_col:

    st.subheader("Salary Estimate")

    st.write(
        "Your predicted annual salary will appear here "
        "after you submit the job information."
    )

    if predict_button:

        predicted_log_salary = model.predict(
            input_data
        )[0]

        predicted_salary_usd = np.expm1(
            predicted_log_salary
        )

        st.metric(
            label="Estimated Annual Salary",
            value=f"${predicted_salary_usd:,.2f}"
        )

        st.caption(
            "Prediction generated using the final tuned "
            "Random Forest salary model."
        )

        st.info(
            f"The estimate is based on a "
            f"{job_level.lower()} role in {country}, "
            f"with {int(experience_required)} years of "
            f"required experience, classified as "
            f"{experience_tier}, and {num_skills} "
            f"selected technical skills."
        )

    else:

        st.info(
            "Complete the job information on the left "
            "and click Predict Salary."
        )


# ============================================================
# METHODOLOGY AND DATA NOTES
# ============================================================

with st.expander("ℹ️ Methodology and Data Notes"):

    st.markdown(
        """
        **Currency Standardization**

        Salary values from different countries were converted to USD using
        **2024 annual average official exchange rates**.

        **Source:** World Bank World Development Indicators  
        **Indicator:** `PA.NUS.FCRF` — Official exchange rate  
        **Definition:** Local Currency Units (LCU) per US$, period average  
        **Underlying source:** IMF International Financial Statistics
        """
    )

    exchange_rate_table = pd.DataFrame({
        "Country": [
            "USA",
            "UK",
            "Canada",
            "India",
            "Australia"
        ],
        "Currency": [
            "USD",
            "GBP",
            "CAD",
            "INR",
            "AUD"
        ],
        "LCU per USD": [
            1.00,
            0.78,
            1.37,
            83.67,
            1.52
        ],
        "Reference Year": [
            2024,
            2024,
            2024,
            2024,
            2024
        ]
    })

    st.table(
        exchange_rate_table
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Tech Job Salary Prediction Model | "
    "Random Forest Regression | Predictions shown in USD"
)
