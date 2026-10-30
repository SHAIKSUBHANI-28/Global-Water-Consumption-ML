import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WaterSense ML",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - WATERSENSE STYLE
# ============================================================

st.markdown("""
<style>

    /* ---------- MAIN BACKGROUND ---------- */
    .stApp {
        background: #f5f9ff;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #062b58 0%, #083b73 55%, #062b58 100%);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    .sidebar-brand {
        text-align: center;
        padding: 8px 4px 20px 4px;
        color: white;
    }

    .sidebar-logo {
        font-size: 38px;
        margin-bottom: 3px;
    }

    .sidebar-name {
        font-size: 20px;
        font-weight: 800;
        letter-spacing: 0.3px;
    }

    .sidebar-subtitle {
        font-size: 10px;
        color: #b9d8f5;
        margin-top: 2px;
    }

    .sidebar-footer {
        margin-top: 35px;
        background: rgba(255,255,255,0.10);
        border-radius: 12px;
        padding: 13px 10px;
        color: #dff2ff;
        text-align: center;
        font-size: 11px;
    }

    /* ---------- NAVIGATION RADIO ---------- */
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 6px;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        border-radius: 8px;
        padding: 8px 10px;
        color: #dbeeff;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.10);
    }

    /* ---------- TOP HEADER ---------- */
    .top-brand {
        color: #07376c;
        font-size: 16px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .page-title {
        color: #07376c;
        font-size: 32px;
        font-weight: 800;
        margin: 0;
    }

    .page-subtitle {
        color: #55708e;
        font-size: 15px;
        margin-top: 5px;
        margin-bottom: 20px;
    }

    /* ---------- HERO ---------- */
    .hero {
        background: linear-gradient(115deg, #0789df 0%, #12b7d9 100%);
        border-radius: 18px;
        padding: 28px 32px;
        color: white;
        box-shadow: 0 8px 25px rgba(7,137,223,0.18);
        margin-bottom: 22px;
    }

    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 28px;
        color: white;
    }

    .hero p {
        margin: 5px 0;
        color: #e9fbff;
        font-size: 14px;
    }

    .hero-value {
        background: rgba(255,255,255,0.94);
        color: #07376c;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        min-height: 105px;
    }

    .hero-value-label {
        font-size: 12px;
        font-weight: 700;
    }

    .hero-value-number {
        font-size: 28px;
        font-weight: 800;
        margin-top: 8px;
    }

    /* ---------- CARDS ---------- */
    .card {
        background: white;
        border: 1px solid #dce9f5;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 3px 12px rgba(31,78,121,0.06);
    }

    .card-title {
        color: #0a477e;
        font-weight: 800;
        font-size: 16px;
        margin-bottom: 12px;
    }

    .small-muted {
        color: #69809a;
        font-size: 13px;
    }

    .big-number {
        color: #07376c;
        font-size: 28px;
        font-weight: 800;
    }

    /* ---------- SECTION HEADINGS ---------- */
    .section-heading {
        color: #07376c;
        font-size: 23px;
        font-weight: 800;
        margin: 22px 0 12px 0;
    }

    /* ---------- PREDICTION CARDS ---------- */
    .prediction-card {
        background: white;
        border-radius: 16px;
        padding: 22px 16px;
        min-height: 165px;
        text-align: center;
        border: 1px solid #d8e7f5;
        box-shadow: 0 4px 15px rgba(30,80,130,0.08);
    }

    .prediction-card.linear {
        border-top: 5px solid #2186e8;
    }

    .prediction-card.tree {
        border-top: 5px solid #2bb673;
    }

    .prediction-card.forest {
        border-top: 5px solid #8b5cf6;
    }

    .prediction-model {
        color: #244f75;
        font-size: 16px;
        font-weight: 800;
    }

    .prediction-label {
        color: #71859a;
        font-size: 11px;
        margin-top: 12px;
    }

    .prediction-number {
        color: #0b4b86;
        font-size: 29px;
        font-weight: 800;
        margin-top: 5px;
    }

    .success-box {
        background: #e9f9ef;
        color: #078342;
        border: 1px solid #bdebd0;
        border-radius: 12px;
        padding: 13px 16px;
        font-weight: 700;
        margin-top: 15px;
    }

    /* ---------- INFO / ABOUT ---------- */
    .info-card {
        background: #ffffff;
        border: 1px solid #dce9f5;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 3px 12px rgba(31,78,121,0.05);
    }

    .info-row {
        padding: 13px 0;
        border-bottom: 1px solid #edf3f8;
    }

    .info-row:last-child {
        border-bottom: none;
    }

    .info-label {
        color: #69809a;
        font-size: 12px;
        font-weight: 700;
    }

    .info-value {
        color: #123f68;
        font-size: 15px;
        font-weight: 700;
        margin-top: 3px;
    }

    /* ---------- STREAMLIT ELEMENTS ---------- */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dce9f5;
        padding: 15px;
        border-radius: 13px;
        box-shadow: 0 3px 12px rgba(31,78,121,0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #69809a;
    }

    div[data-testid="stMetricValue"] {
        color: #07376c;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 800;
        min-height: 46px;
    }

    /* Make dataframe and charts blend with page */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATASET AND MODELS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_FILE = PROJECT_DIR / "cleaned_global_water_consumption.csv"
MODEL_FILE = PROJECT_DIR / "water_models.pkl"
if not MODEL_FILE.is_file():
    MODEL_FILE = PROJECT_DIR / "Global_Water_Consumption" / "water_models.pkl"

df = pd.read_csv(DATASET_FILE)

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
    st.error("Unable to load the saved model file. Please run train_models.py first.")
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
# PERFORMANCE DATA
# ============================================================

model_names = {
    "linear_model": "Linear Regression",
    "decision_tree_model": "Decision Tree",
    "random_forest_model": "Random Forest"
}
performance_data = pd.DataFrame([
    {"Model": model_names[name], **metrics}
    for name, metrics in model_metrics.items()
])

# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">💧</div>
        <div class="sidebar-name">WaterSense ML</div>
        <div class="sidebar-subtitle">Water Consumption Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    selected_page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📋 Dataset",
            "📊 Model Comparison",
            "🔮 Predictions",
            "ℹ️ About Project"
        ],
        index=[
            "🏠 Home",
            "📋 Dataset",
            "📊 Model Comparison",
            "🔮 Predictions",
            "ℹ️ About Project"
        ].index(
            {
                "Home": "🏠 Home",
                "Dataset": "📋 Dataset",
                "Model Comparison": "📊 Model Comparison",
                "Predictions": "🔮 Predictions",
                "About Project": "ℹ️ About Project"
            }.get(st.session_state.page, "🏠 Home")
        ),
        label_visibility="collapsed"
    )

    page_map = {
        "🏠 Home": "Home",
        "📋 Dataset": "Dataset",
        "📊 Model Comparison": "Model Comparison",
        "🔮 Predictions": "Predictions",
        "ℹ️ About Project": "About Project"
    }

    st.session_state.page = page_map[selected_page]

    st.markdown("""
    <div class="sidebar-footer">
        🌱 Smarter Predictions<br>
        for a Sustainable Future
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TOP HEADER
# ============================================================

st.markdown('<div class="top-brand">💧 WaterSense ML</div>', unsafe_allow_html=True)

# ============================================================
# HOME
# ============================================================

if st.session_state.page == "Home":

    st.markdown('<div class="page-title">Machine Learning Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Predict Per Capita Water Consumption using machine learning.</div>',
        unsafe_allow_html=True
    )

    last_value = "—"
    if st.session_state.last_prediction is not None:
        last_value = f"{st.session_state.last_prediction:.2f}"

    st.markdown(f"""
    <div class="hero">
        <div style="display:flex;align-items:center;gap:25px;">
            <div style="font-size:55px;">💧</div>
            <div style="flex:1;">
                <h1>Predict Per Capita Water Use</h1>
                <p>Use environmental and water-consumption factors to generate machine learning predictions.</p>
                <p>Three regression models are available for comparison.</p>
            </div>
            <div class="hero-value">
                <div class="hero-value-label">Latest Predicted Per Capita Water Use</div>
                <div class="hero-value-number">{last_value} <span style="font-size:14px;">L/day</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">📋 Dataset Information</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records", len(df))
    with c2:
        st.metric("Original Columns", len(df.columns))
    with c3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))
    with c4:
        st.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.markdown('<div class="section-heading">⚙️ Project Overview</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="card">
            <div class="card-title">🎯 Target Variable</div>
            <div class="big-number" style="font-size:20px;">Per Capita Water Use</div>
            <div class="small-muted">Liters per Day</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <div class="card-title">🤖 Models Used</div>
            <div class="small-muted">
                Linear Regression<br>
                Decision Tree Regressor<br>
                Random Forest Regressor
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">🚀 Quick Actions</div>', unsafe_allow_html=True)

    q1, q2 = st.columns(2)
    with q1:
        if st.button("🔮 Go to Predictions", width="stretch"):
            st.session_state.page = "Predictions"
            st.rerun()
    with q2:
        if st.button("📊 View Model Comparison", width="stretch"):
            st.session_state.page = "Model Comparison"
            st.rerun()

# ============================================================
# DATASET
# ============================================================

elif st.session_state.page == "Dataset":

    st.markdown('<div class="page-title">Dataset Information</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Overview of the global water consumption dataset.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records", len(df))
    with c2:
        st.metric("Original Columns", len(df.columns))
    with c3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))
    with c4:
        st.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.markdown('<div class="section-heading">📄 Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.5, 1])

    with c1:
        overview = pd.DataFrame({
            "Dataset Property": [
                "Total Records",
                "Original Columns",
                "Missing Values",
                "Duplicate Rows"
            ],
            "Value": [
                len(df),
                len(df.columns),
                int(df.isnull().sum().sum()),
                int(df.duplicated().sum())
            ]
        })
        st.dataframe(overview, width="stretch", hide_index=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">💡 Dataset Summary</div>
            <div class="small-muted">
                The dataset contains global water-consumption records
                with country, year, water-use, rainfall and groundwater
                information.
                <br><br>
                The target variable is
                <b>Per Capita Water Use (Liters per Day)</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">🧾 Dataset Columns</div>', unsafe_allow_html=True)

    columns_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[c].dtype) for c in df.columns]
    })
    st.dataframe(columns_df, width="stretch", hide_index=True)

    st.markdown('<div class="section-heading">👀 Sample Records</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), width="stretch", hide_index=True)

# ============================================================
# MODEL COMPARISON
# ============================================================

elif st.session_state.page == "Model Comparison":

    st.markdown('<div class="page-title">Model Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Comparison of the three regression models on the test dataset.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-heading">📋 Performance Metrics</div>', unsafe_allow_html=True)

    display_df = performance_data.copy()
    display_df["MAE"] = display_df["MAE"].round(4)
    display_df["MSE"] = display_df["MSE"].round(4)
    display_df["RMSE"] = display_df["RMSE"].round(4)
    display_df["R²"] = display_df["R²"].round(4)

    st.dataframe(display_df, width="stretch", hide_index=True)

    st.markdown('<div class="section-heading">📈 MAE and RMSE Comparison</div>', unsafe_allow_html=True)

    chart_data = performance_data.set_index("Model")[["MAE", "RMSE"]]
    st.bar_chart(chart_data, height=380)

    st.markdown('<div class="section-heading">📊 R² Score Comparison</div>', unsafe_allow_html=True)

    r2_data = performance_data.set_index("Model")[["R²"]]
    st.bar_chart(r2_data, height=340)

    st.markdown('<div class="section-heading">🧠 How to Read the Metrics</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown("""
        <div class="card">
            <div class="card-title">MAE</div>
            <div class="small-muted">
                Mean Absolute Error measures the average absolute
                difference between actual and predicted values.
                Lower values indicate smaller average errors.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="card">
            <div class="card-title">RMSE</div>
            <div class="small-muted">
                Root Mean Squared Error gives more weight to larger
                prediction errors. Lower values indicate lower error.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown("""
        <div class="card">
            <div class="card-title">R² Score</div>
            <div class="small-muted">
                R² indicates how much variation in the target is
                explained by the model. A value closer to 1 indicates
                a better fit.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        f"{performance_data.loc[performance_data['MAE'].idxmin(), 'Model']} has the lowest MAE, "
        f"{performance_data.loc[performance_data['RMSE'].idxmin(), 'Model']} has the lowest RMSE, "
        f"and {performance_data.loc[performance_data['R²'].idxmax(), 'Model']} has the highest R²."
    )

# ============================================================
# PREDICTIONS
# ============================================================

elif st.session_state.page == "Predictions":

    st.markdown('<div class="page-title">Water Consumption Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Enter the water and environmental details to predict Per Capita Water Use.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-heading">📝 Enter Water Consumption Details</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

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

    st.write("")

    if st.button("🔮 Predict Per Capita Water Use", width="stretch", type="primary"):

        input_data = pd.DataFrame({
            "Country": [country],
            "Year": [year],
            "Total Water Consumption (Billion Cubic Meters)": [total_water_consumption],
            "Agricultural Water Use (%)": [agricultural_water_use],
            "Industrial Water Use (%)": [industrial_water_use],
            "Household Water Use (%)": [household_water_use],
            "Rainfall Impact (Annual Precipitation in mm)": [rainfall_impact],
            "Groundwater Depletion Rate (%)": [groundwater_depletion]
        })

        # One-Hot Encoding
        input_data = pd.get_dummies(
            input_data,
            columns=["Country"],
            dtype=float
        )

        # Match training feature order
        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        # IQR outlier capping
        for column in numeric_columns:
            if column in input_data.columns:
                input_data[column] = input_data[column].clip(
                    lower=lower_limits[column],
                    upper=upper_limits[column]
                )

        # Standard scaling
        input_scaled = scaler.transform(input_data)

        # Predictions
        linear_prediction = float(linear_model.predict(input_scaled)[0])
        decision_tree_prediction = float(decision_tree_model.predict(input_scaled)[0])
        random_forest_prediction = float(random_forest_model.predict(input_scaled)[0])

        # Save last Linear Regression prediction for Home
        st.session_state.last_prediction = linear_prediction

        st.markdown("---")
        st.markdown('<div class="section-heading">💧 Prediction Results</div>', unsafe_allow_html=True)

        p1, p2, p3 = st.columns(3)

        with p1:
            st.markdown(f"""
            <div class="prediction-card linear">
                <div class="prediction-model">Linear Regression</div>
                <div class="prediction-label">Predicted Per Capita Water Use</div>
                <div class="prediction-number">{linear_prediction:.2f} <span style="font-size:13px;">L/day</span></div>
            </div>
            """, unsafe_allow_html=True)

        with p2:
            st.markdown(f"""
            <div class="prediction-card tree">
                <div class="prediction-model">Decision Tree</div>
                <div class="prediction-label">Predicted Per Capita Water Use</div>
                <div class="prediction-number">{decision_tree_prediction:.2f} <span style="font-size:13px;">L/day</span></div>
            </div>
            """, unsafe_allow_html=True)

        with p3:
            st.markdown(f"""
            <div class="prediction-card forest">
                <div class="prediction-model">Random Forest</div>
                <div class="prediction-label">Predicted Per Capita Water Use</div>
                <div class="prediction-number">{random_forest_prediction:.2f} <span style="font-size:13px;">L/day</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(
            '<div class="success-box">✓ Prediction completed successfully!</div>',
            unsafe_allow_html=True
        )

# ============================================================
# ABOUT PROJECT
# ============================================================

elif st.session_state.page == "About Project":

    st.markdown('<div class="page-title">About the Project</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Information about the WaterSense ML prediction system.</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">💧 Target Variable</div>
            <div class="info-row">
                <div class="info-label">TARGET</div>
                <div class="info-value">Per Capita Water Use (Liters per Day)</div>
            </div>
            <div class="info-row">
                <div class="info-label">PROBLEM TYPE</div>
                <div class="info-value">Supervised Learning - Regression</div>
            </div>
            <div class="info-row">
                <div class="info-label">PROGRAMMING LANGUAGE</div>
                <div class="info-value">Python</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">🤖 Models Used</div>
            <div class="info-row">
                <div class="info-label">MODEL 1</div>
                <div class="info-value">Linear Regression</div>
            </div>
            <div class="info-row">
                <div class="info-label">MODEL 2</div>
                <div class="info-value">Decision Tree Regressor</div>
            </div>
            <div class="info-row">
                <div class="info-label">MODEL 3</div>
                <div class="info-value">Random Forest Regressor</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">⚙️ Preprocessing</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown("""
        <div class="card">
            <div class="card-title">🔢 One-Hot Encoding</div>
            <div class="small-muted">Converts the categorical Country feature into numerical features.</div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="card">
            <div class="card-title">📌 IQR Outlier Capping</div>
            <div class="small-muted">Limits extreme values in selected numerical predictor columns.</div>
        </div>
        """, unsafe_allow_html=True)

    with a3:
        st.markdown("""
        <div class="card">
            <div class="card-title">📏 Standard Scaling</div>
            <div class="small-muted">Standardizes the input features before prediction.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">🎯 Project Objective</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        The project applies Machine Learning to analyze global water
        consumption patterns and predict Per Capita Water Use.
        The application allows users to enter water-consumption and
        environmental factors and compare predictions from three
        regression models.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">🌱 Key Takeaways</div>', unsafe_allow_html=True)

    st.success("Machine Learning can be applied to analyze water-consumption patterns.")
    st.success("The application compares three regression models.")
    st.success("Better prediction analysis can support data-driven water management.")
