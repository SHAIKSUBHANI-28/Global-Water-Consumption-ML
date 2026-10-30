import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Global Water Consumption Prediction",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 48px;
    font-weight: 700;
    color: #30323d;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #243b5a;
    margin-bottom: 30px;
}

.section-title {
    font-size: 32px;
    font-weight: 650;
    color: #243b5a;
    margin-top: 20px;
    margin-bottom: 15px;
}

.prediction-card {
    background-color: #f8fafc;
    border: 1px solid #dce3eb;
    border-radius: 16px;
    padding: 30px 20px;
    text-align: center;
    min-height: 180px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

.prediction-title {
    font-size: 20px;
    font-weight: 600;
    color: #243b5a;
    margin-bottom: 25px;
}

.prediction-value {
    font-size: 30px;
    font-weight: 600;
    color: #243b5a;
}

.success-box {
    background-color: #e9f9ef;
    color: #087f3f;
    padding: 18px;
    border-radius: 12px;
    font-size: 18px;
    margin-top: 10px;
}

.info-box {
    background-color: #eef6ff;
    padding: 20px;
    border-radius: 12px;
    font-size: 16px;
    line-height: 1.7;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATASET
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_FILE = PROJECT_DIR / "cleaned_global_water_consumption.csv"
MODEL_FILE = PROJECT_DIR / "water_models.pkl"

df = pd.read_csv(DATASET_FILE)


# ============================================================
# LOAD SAVED MODELS
# ============================================================

try:

    model_data = joblib.load(MODEL_FILE)

    linear_model = model_data["linear_model"]
    decision_tree_model = model_data["decision_tree_model"]
    random_forest_model = model_data["random_forest_model"]

    scaler = model_data["scaler"]

    feature_columns = model_data["feature_columns"]
    numeric_columns = model_data["numeric_columns"]

    lower_limits = model_data["lower_limits"]
    upper_limits = model_data["upper_limits"]
    model_metrics = model_data["metrics"]

except Exception as e:

    st.error(
        "Unable to load the saved model file. "
        "Please run train_models.py first."
    )

    st.exception(e)
    st.stop()

available_countries = sorted(
    country
    for country in df["Country"].dropna().unique()
    if f"Country_{country}" in feature_columns
)
if not available_countries:
    st.error("No dataset countries match the categories supported by the loaded model.")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌍 Global Water Consumption Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict <b>Per Capita Water Use (Liters per Day)</b> using Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📊 Enter Water Consumption Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    country = st.selectbox(
        "Country",
        available_countries
    )

    year = st.number_input(
        "Year",
        min_value=int(df["Year"].min()),
        max_value=2026,
        value=int(df["Year"].max()),
        step=1
    )

    total_water_consumption = st.number_input(
        "Total Water Consumption (Billion Cubic Meters)",
        min_value=0.0,
        value=100.0,
        step=1.0
    )

    agricultural_water_use = st.number_input(
        "Agricultural Water Use (%)",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    industrial_water_use = st.number_input(
        "Industrial Water Use (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )

    household_water_use = st.number_input(
        "Household Water Use (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )

    rainfall_impact = st.number_input(
        "Rainfall Impact (Annual Precipitation in mm)",
        min_value=0.0,
        value=1000.0,
        step=10.0
    )

    groundwater_depletion = st.number_input(
        "Groundwater Depletion Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=3.0,
        step=0.1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "🔮 Predict Per Capita Water Use",
    width="stretch",
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "Country": [country],

        "Year": [year],

        "Total Water Consumption (Billion Cubic Meters)": [
            total_water_consumption
        ],

        "Agricultural Water Use (%)": [
            agricultural_water_use
        ],

        "Industrial Water Use (%)": [
            industrial_water_use
        ],

        "Household Water Use (%)": [
            household_water_use
        ],

        "Rainfall Impact (Annual Precipitation in mm)": [
            rainfall_impact
        ],

        "Groundwater Depletion Rate (%)": [
            groundwater_depletion
        ]
    })


    # --------------------------------------------------------
    # ONE-HOT ENCODE COUNTRY
    # --------------------------------------------------------

    input_data = pd.get_dummies(
        input_data,
        columns=["Country"],
        dtype=float
    )


    # --------------------------------------------------------
    # MATCH TRAINING FEATURES
    # --------------------------------------------------------

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # IQR OUTLIER CAPPING
    # --------------------------------------------------------

    for column in numeric_columns:

        if column in input_data.columns:

            input_data[column] = input_data[column].clip(
                lower=lower_limits[column],
                upper=upper_limits[column]
            )


    # --------------------------------------------------------
    # STANDARD SCALING
    # --------------------------------------------------------

    input_scaled = scaler.transform(input_data)


    # --------------------------------------------------------
    # MAKE PREDICTIONS
    # --------------------------------------------------------

    linear_prediction = linear_model.predict(
        input_scaled
    )[0]

    decision_tree_prediction = decision_tree_model.predict(
        input_scaled
    )[0]

    random_forest_prediction = random_forest_model.predict(
        input_scaled
    )[0]


# ============================================================
# PREDICTION RESULTS
# ============================================================

if predict_button:

    st.markdown("---")
    st.markdown("## 💧 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Linear Regression")
        st.metric(
            label="Predicted Per Capita Water Use",
            value=f"{linear_prediction:.2f} L/day"
        )

    with col2:
        st.subheader("Decision Tree")
        st.metric(
            label="Predicted Per Capita Water Use",
            value=f"{decision_tree_prediction:.2f} L/day"
        )

    with col3:
        st.subheader("Random Forest")
        st.metric(
            label="Predicted Per Capita Water Use",
            value=f"{random_forest_prediction:.2f} L/day"
        )

    st.success("Prediction completed successfully!")


# ============================================================
# DATASET INFORMATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📋 Dataset Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Records",
        len(df)
    )


with col2:

    st.metric(
        "Original Columns",
        len(df.columns)
    )


with col3:

    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )


with col4:

    st.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">ℹ️ About the Project</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">

    This project uses Machine Learning to predict
    <b>Per Capita Water Use (Liters per Day)</b> based on factors
    such as country, year, total water consumption, agricultural
    water use, industrial water use, household water use,
    rainfall impact, and groundwater depletion rate.

    <br><br>

    <b>Target Variable:</b>
    Per Capita Water Use (Liters per Day)

    <br><br>

    <b>Models Used:</b>
    Linear Regression, Decision Tree Regressor,
    and Random Forest Regressor

    <br><br>

    <b>Preprocessing:</b>
    One-Hot Encoding, IQR Outlier Capping,
    and Standard Scaling

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL PERFORMANCE COMPARISON
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">📊 Model Performance Comparison</div>',
    unsafe_allow_html=True
)

st.write(
    "The following metrics show how the three regression "
    "models performed on the test dataset."
)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

st.subheader("📋 Performance Metrics")

model_names = {
    "linear_model": "Linear Regression",
    "decision_tree_model": "Decision Tree",
    "random_forest_model": "Random Forest"
}
performance_data = pd.DataFrame([
    {"Model": model_names[name], **metrics}
    for name, metrics in model_metrics.items()
])


st.dataframe(
    performance_data,
    width="stretch",
    hide_index=True
)


# ============================================================
# MAE AND RMSE COMPARISON
# ============================================================

st.subheader("📈 MAE and RMSE Comparison")

chart_data = performance_data.set_index("Model")[
    ["MAE", "RMSE"]
]

st.bar_chart(
    chart_data,
    stack=False
)


# ============================================================
# R² SCORE COMPARISON
# ============================================================

st.subheader("📊 R² Score Comparison")

r2_data = performance_data.set_index("Model")[["R²"]]

st.bar_chart(r2_data)


# ============================================================
# MODEL EVALUATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🧠 Model Evaluation</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    **MAE (Mean Absolute Error):**

    Measures the average absolute difference between actual
    and predicted values. Lower MAE indicates smaller
    prediction errors.

    <br>

    **RMSE (Root Mean Squared Error):**

    Measures the square root of the average squared
    prediction error. Lower RMSE indicates lower prediction error.

    <br>

    **R² Score:**

    Shows how well the model explains the variation in the
    target variable. A value closer to 1 indicates a better fit.
    """,
    unsafe_allow_html=True
)


st.info(
    f"{performance_data.loc[performance_data['MAE'].idxmin(), 'Model']} has the lowest MAE, "
    f"{performance_data.loc[performance_data['RMSE'].idxmin(), 'Model']} has the lowest RMSE, "
    f"and {performance_data.loc[performance_data['R²'].idxmax(), 'Model']} has the highest R² score."
)


# ============================================================
# FINAL NOTE
# ============================================================

st.divider()

st.success(
    "Global Water Consumption Prediction application is ready!"
)